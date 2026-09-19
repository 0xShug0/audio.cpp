#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <stdexcept>
struct Case {
 ggml_context *ctx=ggml_init({32*1024*1024,nullptr,true});
 ggml_backend_buffer_t buf=nullptr;
 ggml_cgraph *g=nullptr;
 ~Case(){if(buf)ggml_backend_buffer_free(buf);ggml_free(ctx);}
 void alloc(ggml_backend_t b,ggml_tensor*out){g=ggml_new_graph_custom(ctx,2048,false);ggml_build_forward_expand(g,out);buf=ggml_backend_alloc_ctx_tensors(ctx,b);if(!buf)throw std::runtime_error("alloc");}
 void run(ggml_backend_t b){if(ggml_backend_graph_compute(b,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute");ggml_backend_synchronize(b);}
};
void mm(ggml_backend_t backend,int K,int M,int N,int broadcast,int mode,bool f32,int reps=5,ggml_type dtype=GGML_TYPE_F32){
 Case c; int B=2,BA=broadcast?1:B;
 auto*a=ggml_new_tensor_3d(c.ctx,dtype,K,M,BA);auto*b=ggml_new_tensor_3d(c.ctx,GGML_TYPE_F32,K,N,B);
 auto*out=ggml_mul_mat(c.ctx,a,b);if(f32)ggml_mul_mat_set_prec(out,GGML_PREC_F32);
 c.alloc(backend,out);
 std::vector<float> av(K*M*BA),bv(K*N*B),got(M*N*B);
 for(size_t i=0;i<av.size();i++)av[i]=mode==1?70000.f+float(i%31):mode==2?1.f+float(i%23)*0.00001f:std::sin(float(i)*0.013f);
 for(size_t i=0;i<bv.size();i++)bv[i]=mode==2?(i%K%2?-1.f:1.f):std::cos(float(i)*0.017f);
 if(dtype==GGML_TYPE_F16){std::vector<ggml_fp16_t>h(av.size());ggml_fp32_to_fp16_row(av.data(),h.data(),h.size());ggml_backend_tensor_set(a,h.data(),0,h.size()*2);ggml_fp16_to_fp32_row(h.data(),av.data(),h.size());}
 else ggml_backend_tensor_set(a,av.data(),0,av.size()*4);
 ggml_backend_tensor_set(b,bv.data(),0,bv.size()*4);c.run(backend);
 auto t0=std::chrono::steady_clock::now();for(int r=0;r<reps;r++)c.run(backend);
 double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/reps;
 ggml_backend_tensor_get(out,got.data(),0,got.size()*4);
 double maxerr=0,se=0,ss=0,maxscaled=0;int nonfinite=0;
 for(int z=0;z<B;z++)for(int n=0;n<N;n++)for(int m=0;m<M;m++){
  double ref=0,sumabs=0;for(int k=0;k<K;k++){double v=double(av[((broadcast?0:z)*M+m)*K+k])*bv[(z*N+n)*K+k];ref+=v;sumabs+=std::abs(v);}
  float v=got[(z*N+n)*M+m];if(!std::isfinite(v)){nonfinite++;continue;}
  double e=std::abs(v-ref);maxerr=std::max(maxerr,e);maxscaled=std::max(maxscaled,e/(1+sumabs));se+=e*e;ss+=ref*ref;
 }
 printf("MM K=%d M=%d N=%d bc=%d mode=%d prec=%s dtype=%s nonfinite=%d maxerr=%.9g relrmse=%.9g scaled=%.9g ms=%.4f\n",K,M,N,broadcast,mode,f32?"f32":"default",ggml_type_name(dtype),nonfinite,maxerr,std::sqrt(se/(ss+1e-30)),maxscaled,ms);fflush(stdout);
 if(nonfinite || (f32 && dtype==GGML_TYPE_F32 && maxscaled>2e-6))
  throw std::runtime_error("explicit F32 matmul precision regression");
}
int main() {
 auto b=ggml_backend_metal_init();if(!b)return 77;
 try {
 for(int k:{31,32,33,48,63,64,65,128})for(int bc:{0,1})mm(b,k,67,35,bc,0,true);
 for(int mode:{1,2})for(int k:{32,48,64,128})mm(b,k,67,35,0,mode,true);
 for(int m:{63,64,67})for(int n:{8,9,31,32,35})mm(b,32,m,n,0,0,true);
 for(int k:{32,48})mm(b,k,512,512,0,0,true,20);
 for(bool prec:{false,true}){mm(b,128,67,35,0,0,prec);mm(b,128,67,35,0,0,prec,5,GGML_TYPE_F16);}
 } catch (const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());ggml_backend_free(b);return 1;}
 ggml_backend_free(b);
}
