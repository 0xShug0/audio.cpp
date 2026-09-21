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
void copy(ggml_backend_t backend,int K,int R){
 Case c;const int C=5,B=2,guard=8192;int count=K*R*C*B;
 // Permute [C,K,R,B] into [K,R,C,B], a real non-contiguous ggml view.
 auto *storage=ggml_new_tensor_1d(c.ctx,GGML_TYPE_F32,count+guard);
 auto *source=ggml_view_4d(c.ctx,storage,C,K,R,B,C*4,C*K*4,C*K*R*4,0);
 source=ggml_permute(c.ctx,source,2,0,1,3);
 auto *dest=ggml_new_tensor_1d(c.ctx,GGML_TYPE_F32,count+guard);
 auto *view=ggml_view_4d(c.ctx,dest,K,R,C,B,K*4,K*R*4,K*R*C*4,0);
 auto *out=ggml_cpy(c.ctx,source,view);c.alloc(backend,out);
 std::vector<float>in(count+guard),init(count+guard,-1234567.f),got(count+guard);
 for(size_t i=0;i<in.size();i++)in[i]=float(i+1);
 ggml_backend_tensor_set(storage,in.data(),0,in.size()*4);
 int worst=0,tail=0;
 for(int t=0;t<5;t++){
  ggml_backend_tensor_set(dest,init.data(),0,init.size()*4);c.run(backend);ggml_backend_tensor_get(dest,got.data(),0,got.size()*4);
  int bad=0,over=0;
  for(int b=0;b<B;b++)for(int ch=0;ch<C;ch++)for(int r=0;r<R;r++)for(int k=0;k<K;k++)
   bad+=got[((b*C+ch)*R+r)*K+k]!=in[((b*R+r)*K+k)*C+ch];
  for(int i=count;i<count+guard;i++)over+=got[i]!=init[i];
  worst=std::max(worst,bad);tail=std::max(tail,over);
 }
 printf("COPY K=%d rows=%d channels=%d batch=%d wrong=%d guard_overwrites=%d\n",K,R,C,B,worst,tail);fflush(stdout);
 if(worst || tail)throw std::runtime_error("copy output or guard mismatch");
}
int main() {
 auto b=ggml_backend_metal_init();if(!b)return 77;
 try {
 for(int k:{7,32,48,64,129,256})for(int r:{1,3,7,16})copy(b,k,r);
 } catch (const std::exception & e) { std::fprintf(stderr,"%s\n",e.what());ggml_backend_free(b);return 1; }
 ggml_backend_free(b);
}
