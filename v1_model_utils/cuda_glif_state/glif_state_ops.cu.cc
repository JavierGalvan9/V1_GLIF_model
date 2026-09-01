#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
template <typename T> __device__ float F(T x) { return static_cast<float>(x); }
template <typename T> __device__ T V(float x) { return static_cast<T>(x); }

template <typename T, typename R, int Basis>
__global__ void ForwardKernel(int64 count, int neurons,
    const T* z, const T* v, const R* r, const T* asc, const T* rise,
    const T* psc, const T* inputs, const T* syn_decay, const T* initial,
    const T* asc_decay, const T* asc_amps, const T* decay,
    const T* current_factor, const R* t_ref, const T* dt, const T* v_reset,
    bool hard_reset, T* new_v, R* new_r, T* new_asc, T* new_rise, T* new_psc) {
  GPU_1D_KERNEL_LOOP(i, count) {
    const int neuron = i % neurons;
    const int parameter_base = neuron * Basis;
    const int64 state_base = i * Basis;
    const float reset = F(z[i]);
    float current = F(asc[2*i]) + F(asc[2*i+1]);
    #pragma unroll
    for (int k = 0; k < Basis; ++k) {
      const int p = parameter_base + k; const int64 j = state_base + k;
      const float d = F(syn_decay[p]); const float old_rise = F(rise[j]);
      current += F(psc[j]);
      new_rise[j] = V<T>(old_rise*d + F(inputs[j])*F(initial[p]));
      new_psc[j] = V<T>(F(psc[j])*d + F(*dt)*d*old_rise);
    }
    const int refractory = max(static_cast<int>(r[i]) +
        static_cast<int>(reset)*static_cast<int>(t_ref[neuron])-1, 0);
    float voltage = F(decay[neuron])*F(v[i]) +
        F(current_factor[neuron])*current - reset;
    if (hard_reset && refractory > 0) voltage = F(*v_reset);
    new_v[i] = V<T>(voltage); new_r[i] = static_cast<R>(refractory);
    new_asc[2*i] = V<T>(F(asc_decay[2*neuron])*F(asc[2*i]) + reset*F(asc_amps[2*neuron]));
    new_asc[2*i+1] = V<T>(F(asc_decay[2*neuron+1])*F(asc[2*i+1]) + reset*F(asc_amps[2*neuron+1]));
  }
}

template <typename T, typename R, int Basis>
__global__ void BackwardKernel(int64 count, int neurons,
    const T* z, const R* r, const T* syn_decay, const T* initial,
    const T* asc_decay, const T* asc_amps, const T* decay,
    const T* current_factor, const R* t_ref, const T* dt, const T* gv,
    const T* ga, const T* grise, const T* gpsc, bool hard_reset,
    bool detach_reset, bool detach_asc_reset,
    T* z_grad, T* v_grad, T* asc_grad, T* rise_grad, T* psc_grad,
    T* input_grad) {
  GPU_1D_KERNEL_LOOP(i, count) {
    const int neuron = i % neurons;
    const int refractory = max(static_cast<int>(r[i]) +
        static_cast<int>(F(z[i]))*static_cast<int>(t_ref[neuron])-1, 0);
    const float active_gv = hard_reset && refractory > 0 ? 0.0f : F(gv[i]);
    const float current_g = active_gv*F(current_factor[neuron]);
    // The membrane reset is detached; ASC injection remains differentiable.
    const float asc_reset_g = detach_asc_reset ? 0.0f :
        F(ga[2*i])*F(asc_amps[2*neuron]) +
        F(ga[2*i+1])*F(asc_amps[2*neuron+1]);
    z_grad[i] = V<T>((detach_reset ? 0.0f : -active_gv) + asc_reset_g);
    v_grad[i] = V<T>(active_gv*F(decay[neuron]));
    asc_grad[2*i] = V<T>(current_g + F(ga[2*i])*F(asc_decay[2*neuron]));
    asc_grad[2*i+1] = V<T>(current_g + F(ga[2*i+1])*F(asc_decay[2*neuron+1]));
    const int parameter_base = neuron*Basis; const int64 state_base = i*Basis;
    #pragma unroll
    for (int k = 0; k < Basis; ++k) {
      const int p = parameter_base+k; const int64 j = state_base+k;
      const float d = F(syn_decay[p]);
      rise_grad[j] = V<T>(F(grise[j])*d + F(gpsc[j])*F(*dt)*d);
      psc_grad[j] = V<T>(F(gpsc[j])*d + current_g);
      input_grad[j] = V<T>(F(grise[j])*F(initial[p]));
    }
  }
}

template <typename T, typename R> class ForwardOp : public OpKernel {
 public: explicit ForwardOp(OpKernelConstruction* c):OpKernel(c){OP_REQUIRES_OK(c,c->GetAttr("hard_reset",&hard_));}
 void Compute(OpKernelContext* c) override {
  const Tensor& v=c->input(1); const Tensor& psc=c->input(5); const int neurons=v.dim_size(1), basis=psc.dim_size(1)/neurons;
  OP_REQUIRES(c,basis==4,errors::InvalidArgument("single-kernel candidate requires four synaptic bases"));
  Tensor *ov,*orr,*oa,*orise,*opsc; OP_REQUIRES_OK(c,c->allocate_output(0,v.shape(),&ov)); OP_REQUIRES_OK(c,c->allocate_output(1,c->input(2).shape(),&orr)); OP_REQUIRES_OK(c,c->allocate_output(2,c->input(3).shape(),&oa)); OP_REQUIRES_OK(c,c->allocate_output(3,c->input(4).shape(),&orise)); OP_REQUIRES_OK(c,c->allocate_output(4,psc.shape(),&opsc));
  auto& d=c->eigen_device<GPUDevice>(); auto cfg=GetGpuLaunchConfig(v.NumElements(),d); OP_REQUIRES_OK(c,GpuLaunchKernel(ForwardKernel<T,R,4>,cfg.block_count,cfg.thread_per_block,0,d.stream(),v.NumElements(),neurons,c->input(0).flat<T>().data(),v.flat<T>().data(),c->input(2).flat<R>().data(),c->input(3).flat<T>().data(),c->input(4).flat<T>().data(),psc.flat<T>().data(),c->input(6).flat<T>().data(),c->input(7).flat<T>().data(),c->input(8).flat<T>().data(),c->input(9).flat<T>().data(),c->input(10).flat<T>().data(),c->input(11).flat<T>().data(),c->input(12).flat<T>().data(),c->input(13).flat<R>().data(),c->input(14).flat<T>().data(),c->input(15).flat<T>().data(),hard_,ov->flat<T>().data(),orr->flat<R>().data(),oa->flat<T>().data(),orise->flat<T>().data(),opsc->flat<T>().data()));
 } private: bool hard_;
};

template <typename T, typename R> class BackwardOp : public OpKernel {
 public: explicit BackwardOp(OpKernelConstruction* c):OpKernel(c){OP_REQUIRES_OK(c,c->GetAttr("hard_reset",&hard_)); OP_REQUIRES_OK(c,c->GetAttr("detach_reset",&detach_)); OP_REQUIRES_OK(c,c->GetAttr("detach_asc_reset",&detach_asc_));}
 void Compute(OpKernelContext* c) override {
  const Tensor& gv=c->input(12); const Tensor& rise=c->input(3); const int neurons=gv.dim_size(1), basis=rise.dim_size(1)/neurons;
  OP_REQUIRES(c,basis==4,errors::InvalidArgument("single-kernel candidate requires four synaptic bases"));
  Tensor *zg,*vg,*ag,*rg,*pg,*ig; OP_REQUIRES_OK(c,c->allocate_output(0,gv.shape(),&zg)); OP_REQUIRES_OK(c,c->allocate_output(1,gv.shape(),&vg)); OP_REQUIRES_OK(c,c->allocate_output(2,c->input(2).shape(),&ag)); OP_REQUIRES_OK(c,c->allocate_output(3,rise.shape(),&rg)); OP_REQUIRES_OK(c,c->allocate_output(4,rise.shape(),&pg)); OP_REQUIRES_OK(c,c->allocate_output(5,rise.shape(),&ig));
  auto& d=c->eigen_device<GPUDevice>(); auto cfg=GetGpuLaunchConfig(gv.NumElements(),d); OP_REQUIRES_OK(c,GpuLaunchKernel(BackwardKernel<T,R,4>,cfg.block_count,cfg.thread_per_block,0,d.stream(),gv.NumElements(),neurons,c->input(0).flat<T>().data(),c->input(1).flat<R>().data(),c->input(4).flat<T>().data(),c->input(5).flat<T>().data(),c->input(6).flat<T>().data(),c->input(10).flat<T>().data(),c->input(7).flat<T>().data(),c->input(8).flat<T>().data(),c->input(9).flat<R>().data(),c->input(11).flat<T>().data(),gv.flat<T>().data(),c->input(13).flat<T>().data(),c->input(14).flat<T>().data(),c->input(15).flat<T>().data(),hard_,detach_,detach_asc_,zg->flat<T>().data(),vg->flat<T>().data(),ag->flat<T>().data(),rg->flat<T>().data(),pg->flat<T>().data(),ig->flat<T>().data()));
 } private: bool hard_; bool detach_; bool detach_asc_;
};

#define REG(T,R) REGISTER_KERNEL_BUILDER(Name("FusedGlifSingleForward").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<R>("R"),ForwardOp<T,R>); REGISTER_KERNEL_BUILDER(Name("FusedGlifSingleBackward").Device(DEVICE_GPU).TypeConstraint<T>("T").TypeConstraint<R>("R"),BackwardOp<T,R>);
REG(float,int8); REG(float,int16); REG(Eigen::half,int8); REG(Eigen::half,int16);
#undef REG
#endif
