#include "sgemm.cuh"
#include <hipblaslt/hipblaslt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { if ((x) != 0) { std::fprintf(stderr, "API failure %s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

static void run(int m, int n, int k, int pad) {
    const int lda=k+pad, ldb=k+pad+1, ldc=m+pad;
    std::vector<float> a(size_t(lda)*m), b(size_t(ldb)*n), out(size_t(ldc)*n), ref(out.size());
    for (size_t i=0;i<a.size();++i) a[i]=float(int((i*17+13)%251)-125)/128;
    for (size_t i=0;i<b.size();++i) b[i]=float(int((i*31+7)%257)-128)/128;
    float *da,*db,*dc;
    CHECK(hipMalloc(&da,a.size()*4)); CHECK(hipMalloc(&db,b.size()*4)); CHECK(hipMalloc(&dc,out.size()*4));
    CHECK(hipMemcpy(da,a.data(),a.size()*4,hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db,b.data(),b.size()*4,hipMemcpyHostToDevice));
    hipblasLtHandle_t handle; hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t la,lb,lc; hipblasLtMatmulPreference_t pref;
    CHECK(hipblasLtCreate(&handle)); CHECK(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    hipblasOperation_t ta=HIPBLAS_OP_T;
    CHECK(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    CHECK(hipblasLtMatrixLayoutCreate(&la,HIP_R_32F,k,m,lda));
    CHECK(hipblasLtMatrixLayoutCreate(&lb,HIP_R_32F,k,n,ldb));
    CHECK(hipblasLtMatrixLayoutCreate(&lc,HIP_R_32F,m,n,ldc));
    CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t worksize=32*1024*1024;void *work;
    CHECK(hipMalloc(&work,worksize));
    CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&worksize,sizeof(worksize)));
    hipblasLtMatmulHeuristicResult_t algo;int count=0;
    CHECK(hipblasLtMatmulAlgoGetHeuristic(handle,desc,la,lb,lc,lc,pref,1,&algo,&count));
    if (!count) std::exit(1);
    float alpha=1,beta=0;
    auto launch=[&](bool halo) {
        if (halo) ggml_hip_halo_sgemm(da,db,dc,m,n,k,lda,ldb,ldc,0,0,0,1,nullptr);
        else CHECK(hipblasLtMatmul(handle,desc,&alpha,da,la,db,lb,&beta,dc,lc,dc,lc,&algo.algo,work,worksize,nullptr));
    };
    launch(false);CHECK(hipMemcpy(ref.data(),dc,out.size()*4,hipMemcpyDeviceToHost));
    launch(true);CHECK(hipMemcpy(out.data(),dc,out.size()*4,hipMemcpyDeviceToHost));
    // Independent double accumulation at distributed output positions, plus
    // exhaustive comparison with the existing FP32 library implementation.
    for (int j=0;j<n;++j) for(int i=0;i<m;++i) {
        float value=out[size_t(j)*ldc+i], expected=ref[size_t(j)*ldc+i];
        if (!std::isfinite(value) || std::abs(value-expected)>2e-4f+2e-5f*std::abs(expected)) std::exit(2);
        if ((size_t(j)*m+i)%std::max<size_t>(1,size_t(m)*n/257)==0) {
            double exact=0;for(int q=0;q<k;++q) exact+=double(a[size_t(i)*lda+q])*b[size_t(j)*ldb+q];
            if (std::abs(value-exact)>2e-4+2e-5*std::abs(exact)) std::exit(3);
        }
    }
    hipEvent_t start,end;CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&end));
    std::vector<float> times[2];
    for(int round=0;round<7;++round) for(int order=0;order<2;++order) {
        int mode=(round+order)%2;
        for(int i=0;i<3;++i)launch(mode);
        CHECK(hipEventRecord(start));for(int i=0;i<8;++i)launch(mode);
        CHECK(hipEventRecord(end));CHECK(hipEventSynchronize(end));
        float ms;CHECK(hipEventElapsedTime(&ms,start,end));times[mode].push_back(ms/8);
    }
    for(auto & t:times)std::sort(t.begin(),t.end());
    std::printf("SGEMM,%d,%d,%d,%d,%.6f,%.6f,%.4f\n",m,n,k,pad,times[0][3],times[1][3],times[0][3]/times[1][3]);
    CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(end));
    CHECK(hipblasLtMatmulPreferenceDestroy(pref));CHECK(hipblasLtMatrixLayoutDestroy(la));
    CHECK(hipblasLtMatrixLayoutDestroy(lb));CHECK(hipblasLtMatrixLayoutDestroy(lc));
    CHECK(hipblasLtMatmulDescDestroy(desc));CHECK(hipblasLtDestroy(handle));
    CHECK(hipFree(work));CHECK(hipFree(da));CHECK(hipFree(db));CHECK(hipFree(dc));
}

int main() {
    for(int pad:{0,3})for(auto s:std::vector<std::vector<int>>{
        {1,1,1},{31,33,65},{64,64,64},{67,131,97},
        {512,128,1536},{256,1024,3584},{128,8192,1792},
        {64,32768,896},{1024,128,1024},{128,550,3200}})run(s[0],s[1],s[2],pad);
    std::puts("PASS SGEMM: library parity and independent double reference");
}
