#include "external/ggml/src/ggml-hip/halo/mmvq.cuh"
#include "original_mmvq.cuh"
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
    ggml_cuda_mm_fusion_args_device f{};
    if(gated) { f.gate=dg;f.x_bias=db;f.gate_bias=dgb;f.glu_op=GGML_GLU_OP_SWIGLU; }
    const auto fd=init_fastdiv_values(1);
    auto launch=[&](bool halo) {
        if(halo)ggml_hip_halo_mmvq_q8_0(dw,dy,dc+16,k,m,rs,f,nullptr);
        else if(gated)original::mul_mat_vec_q<GGML_TYPE_Q8_0,1,true><<<dim3(m),dim3(32,4)>>>(
            dw,dy,nullptr,f,dc+16,k,make_uint3(0,0,0),rs,blocks,m,fd,rs*m,blocks,m,fd,rs*m,blocks,m,0);
        else original::mul_mat_vec_q<GGML_TYPE_Q8_0,1,false><<<dim3(m),dim3(32,4)>>>(
            dw,dy,nullptr,f,dc+16,k,make_uint3(0,0,0),rs,blocks,m,fd,rs*m,blocks,m,fd,rs*m,blocks,m,0);
    };
    launch(false);CHECK(hipMemcpy(old.data(),dc,old.size()*4,hipMemcpyDeviceToHost));
    launch(true);CHECK(hipMemcpy(out.data(),dc,out.size()*4,hipMemcpyDeviceToHost));
    for(int i=0;i<16;++i)if(out[i]!=-9876.5f || out[m+16+i]!=-9876.5f)std::exit(2);
    double max_error=0;
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
        max_error=std::max(max_error,std::abs(value-out[row+16]));
        const double tolerance=2e-4+2e-5*std::abs(value);
        if(!std::isfinite(out[row+16]) || std::abs(value-out[row+16])>tolerance || std::abs(value-old[row+16])>tolerance) {
            std::fprintf(stderr,"FAIL k=%d m=%d row=%d gate=%d ref=%.9g old=%.9g new=%.9g\n",k,m,row,int(gated),value,old[row+16],out[row+16]);std::exit(3);
        }
    }
    hipEvent_t start,end;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&end));
    std::vector<float> times[2];
    for(int round=0;round<7;++round)for(int order=0;order<2;++order) {
        int mode=(round+order)%2;
        for(int i=0;i<8;++i)launch(mode);
        CHECK(hipEventRecord(start));for(int i=0;i<64;++i)launch(mode);
        CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
        float ms;CHECK(hipEventElapsedTime(&ms,start,end));times[mode].push_back(ms/64);
    }
    for(auto &v:times)std::sort(v.begin(),v.end());
    std::printf("MMVQ,%d,%d,%d,%d,%.8f,%.8f,%.4f,max_error=%.8g\n",k,m,int(gated),pad,times[0][3],times[1][3],times[0][3]/times[1][3],max_error);
    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(end));
    CHECK(hipFree(dw));CHECK(hipFree(dg));CHECK(hipFree(dy));CHECK(hipFree(dc));CHECK(hipFree(db));CHECK(hipFree(dgb));
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
