#include "external/ggml/src/ggml-hip/halo/mmvq.cuh"
#include "original_mmvq.cuh"
#include "assets/hip_halo/mmvq_candidates.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { if ((x)!=0) { std::fprintf(stderr,"API failure %s:%d: %s\n",__FILE__,__LINE__,#x);std::exit(1); } } while(0)

static void run(int k,int m,bool gated,int pad=0) {
    const int blocks=k/32,rs=blocks+pad;
    std::vector<block_q8_0> w(size_t(rs)*m),g(w.size());
    std::vector<block_q8_1> y(blocks);
    std::vector<float> bias(m),gb(m),out(m+32,-9876.5f),old(out);
    std::vector<int8_t> packed_w(size_t(k)*m),packed_g(packed_w.size());
    std::vector<__half> scales(size_t(blocks)*m),gate_scales(scales.size());
    for(size_t i=0;i<w.size();++i) {
        w[i].d=__float2half(0.00313f+float(i%11)*0.00041f);
        g[i].d=__float2half(0.00419f+float(i%7)*0.00031f);
        for(int j=0;j<32;++j) {
            w[i].qs[j]=int((i*17+j*31)%255)-127;
            g[i].qs[j]=int((i*29+j*43)%255)-127;
        }
    }
    for(int i=0;i<blocks;++i) {
        int sum=0;
        for(int j=0;j<32;++j) { y[i].qs[j]=((i*43+j*17)%255)-127;sum+=y[i].qs[j]; }
        const float d=0.00717f+float(i%5)*0.00139f;
        y[i].ds=__halves2half2(__float2half(d),__float2half(d*sum));
    }
    for(int i=0;i<m;++i) { bias[i]=std::sin(float(i)*0.31f)/7;gb[i]=std::cos(float(i)*0.23f)/11; }
    for(int row=0;row<m;++row)for(int block=0;block<blocks;++block) {
        int src=row*rs+block,dst=row*blocks+block;
        std::copy(w[src].qs,w[src].qs+32,packed_w.data()+dst*32);
        std::copy(g[src].qs,g[src].qs+32,packed_g.data()+dst*32);
        scales[dst]=w[src].d;gate_scales[dst]=g[src].d;
    }
    block_q8_0 *dw,*dg;block_q8_1 *dy;float *dc,*db,*dgb;
    CHECK(hipMalloc(&dw,w.size()*sizeof(w[0])));CHECK(hipMalloc(&dg,g.size()*sizeof(g[0])));
    CHECK(hipMalloc(&dy,y.size()*sizeof(y[0])));CHECK(hipMalloc(&dc,out.size()*4));
    CHECK(hipMalloc(&db,bias.size()*4));CHECK(hipMalloc(&dgb,gb.size()*4));
    CHECK(hipMemcpy(dw,w.data(),w.size()*sizeof(w[0]),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dg,g.data(),g.size()*sizeof(g[0]),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dy,y.data(),y.size()*sizeof(y[0]),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db,bias.data(),bias.size()*4,hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dgb,gb.data(),gb.size()*4,hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dc,out.data(),out.size()*4,hipMemcpyHostToDevice));
    int8_t *dpw,*dpg;__half *ds,*dgs;
    CHECK(hipMalloc(&dpw,packed_w.size()));CHECK(hipMalloc(&dpg,packed_g.size()));
    CHECK(hipMalloc(&ds,scales.size()*2));CHECK(hipMalloc(&dgs,gate_scales.size()*2));
    CHECK(hipMemcpy(dpw,packed_w.data(),packed_w.size(),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dpg,packed_g.data(),packed_g.size(),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(ds,scales.data(),scales.size()*2,hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dgs,gate_scales.data(),gate_scales.size()*2,hipMemcpyHostToDevice));
    ggml_cuda_mm_fusion_args_device f{};
    if(gated) { f.gate=dg;f.x_bias=db;f.gate_bias=dgb;f.glu_op=GGML_GLU_OP_SWIGLU; }
    const auto fd=init_fastdiv_values(1);
    const int warps=original::calc_nwarps(GGML_TYPE_Q8_0,1,
        original::get_device_table_id(GGML_CUDA_CC_OFFSET_AMD+0x1151));
    auto launch=[&](int mode) {
        if(mode==1)ggml_hip_halo_mmvq_q8_0(dw,dy,dc+16,k,m,rs,f,nullptr);
        else if(mode==2)launch_candidate<true,false>(dw,nullptr,nullptr,dy,dc+16,k,m,rs,f);
        else if(mode==3 || mode==4) {
            auto pf=f;
            pf.gate=gated?dpg:nullptr;
            if(mode==3)launch_candidate<false,true>(dpw,ds,dgs,dy,dc+16,k,m,blocks,pf);
            else launch_candidate<true,true>(dpw,ds,dgs,dy,dc+16,k,m,blocks,pf);
        }
        else if(gated)original::mul_mat_vec_q<GGML_TYPE_Q8_0,1,true><<<dim3(m),dim3(32,warps)>>>(
            dw,dy,nullptr,f,dc+16,k,make_uint3(0,0,0),rs,blocks,m,fd,rs*m,blocks,m,fd,rs*m,blocks,m,0);
        else original::mul_mat_vec_q<GGML_TYPE_Q8_0,1,false><<<dim3(m),dim3(32,warps)>>>(
            dw,dy,nullptr,f,dc+16,k,make_uint3(0,0,0),rs,blocks,m,fd,rs*m,blocks,m,fd,rs*m,blocks,m,0);
    };
    launch(0);CHECK(hipGetLastError());CHECK(hipMemcpy(old.data(),dc,old.size()*4,hipMemcpyDeviceToHost));
    std::vector<double> expected(m);
    for(int row=0;row<m;++row) {
        double value=0,gate=0;
        for(int block=0;block<blocks;++block) {
            int dot=0,gdot=0;
            for(int j=0;j<32;++j) {
                dot+=int(w[row*rs+block].qs[j])*y[block].qs[j];
                gdot+=int(g[row*rs+block].qs[j])*y[block].qs[j];
            }
            const double scale=__half2float(__low2half(y[block].ds));
            value+=dot*double(__half2float(w[row*rs+block].d))*scale;
            gate+=gdot*double(__half2float(g[row*rs+block].d))*scale;
        }
        if(gated) { value+=bias[row];gate+=gb[row];value*=gate/(1+std::exp(-gate)); }
        expected[row]=value;
    }
    double max_error=0;
    for(int mode=0;mode<5;++mode) {
        std::fill(out.begin(),out.end(),-9876.5f);
        CHECK(hipMemcpy(dc,out.data(),out.size()*4,hipMemcpyHostToDevice));
        launch(mode);CHECK(hipGetLastError());CHECK(hipMemcpy(out.data(),dc,out.size()*4,hipMemcpyDeviceToHost));
        for(int i=0;i<16;++i)if(out[i]!=-9876.5f || out[m+16+i]!=-9876.5f)std::exit(2);
        for(int row=0;row<m;++row) {
            double value=expected[row];
            max_error=std::max(max_error,std::abs(value-out[row+16]));
            if(!std::isfinite(out[row+16]) || std::abs(value-out[row+16])>2e-4+2e-5*std::abs(value)) {
                std::fprintf(stderr,"FAIL mode=%d k=%d m=%d row=%d gate=%d ref=%.9g old=%.9g new=%.9g\n",mode,k,m,row,int(gated),value,old[row+16],out[row+16]);std::exit(3);
            }
        }
    }
    hipEvent_t start,end;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&end));
    std::vector<float> times[5];
    for(int round=0;round<7;++round)for(int order=0;order<5;++order) {
        int mode=(round+order)%5;
        for(int i=0;i<8;++i)launch(mode);
        CHECK(hipEventRecord(start));for(int i=0;i<64;++i)launch(mode);
        CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
        CHECK(hipGetLastError());
        float ms;CHECK(hipEventElapsedTime(&ms,start,end));times[mode].push_back(ms/64);
    }
    for(auto &v:times)std::sort(v.begin(),v.end());
    std::printf("MMVQ,%d,%d,%d,%d",k,m,int(gated),pad);
    for(auto &v:times)std::printf(",%.8f",v[3]);
    for(int mode=1;mode<5;++mode)std::printf(",%.4f",times[0][3]/times[mode][3]);
    std::printf(",max_error=%.8g\n",max_error);
    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(end));
    CHECK(hipFree(dw));CHECK(hipFree(dg));CHECK(hipFree(dy));CHECK(hipFree(dc));CHECK(hipFree(db));CHECK(hipFree(dgb));
    CHECK(hipFree(dpw));CHECK(hipFree(dpg));CHECK(hipFree(ds));CHECK(hipFree(dgs));
}

int main() {
    for(bool gate:{false,true}) {
        for(auto shape:std::vector<std::vector<int>>{{32,1},{96,3},{512,67},{512,512},{512,1024},{512,2048},
            {1024,512},{1024,1024},{1024,2048},{1024,3072},{1024,4096},
            {2048,512},{2048,1024},{2048,2048},{2048,3072},{2048,6144},
            {3072,1024},{4096,1024},{6144,2048}})run(shape[0],shape[1],gate);
        run(512,65,gate,3);
    }
    std::puts("PASS MMVQ original-kernel and independent integer/double reference; guards intact");
}
