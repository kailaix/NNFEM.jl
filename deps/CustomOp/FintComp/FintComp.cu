#define GOOGLE_CUDA 1
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"


namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;

  __global__ void forward_(const int nthreads, double *Fint, const double *fints, const int32*el_eqns, int32 ngs, int32 neqns_per_elem, int32 neqs){
    for(int i : CudaGridRangeX(nthreads)) 
      for(int32 j=0;j<neqns_per_elem; j++){
          auto fint = fints + i*neqns_per_elem;
          auto el_eqns_row = el_eqns + i*neqns_per_elem;
          if(el_eqns_row[j] > 0)
            Fint[el_eqns_row[j]-1] += fint[j];
      }
  }

  __global__ void zero_(const int nthreads, double *Fint){
    for(int i: CudaGridRangeX(nthreads))
      Fint[i] = 0.0;
  }

  int32 getInt32(const int32* i){
    int32 I;
    cudaMemcpy(&I, i, sizeof(int32), cudaMemcpyDeviceToHost);
    return I;
  }

  void forwardGPU(double *Fint, const double *fints, const int32*el_eqns, int32 ngs, 
      int32 neqns_per_elem, int32 NEQS, const GPUDevice& d){
  
      GpuLaunchConfig config1 = GetGpuLaunchConfig(NEQS, d);
      GpuLaunchConfig config2 = GetGpuLaunchConfig(ngs, d);

      TF_CHECK_OK(GpuLaunchKernel(
        zero_, config1.block_count, config1.thread_per_block, 0,
        d.stream(), config1.virtual_thread_count, Fint));

      TF_CHECK_OK(GpuLaunchKernel(
          forward_, config2.block_count, config2.thread_per_block, 0,
          d.stream(), config2.virtual_thread_count, Fint, fints, el_eqns,
          ngs, neqns_per_elem, NEQS));
  }


  __global__ void backward_(const int nthreads, double *fint_grad, const double *Fint_grad, const double *Fint, 
    const double *fints, const int32*el_eqns, int32 ngs, int32 neqns_per_elem){
      for(int i : CudaGridRangeX(nthreads)) {
        auto el_eqns_row = el_eqns + i*neqns_per_elem;
        auto fint_grad_ = fint_grad + i*neqns_per_elem;
        for(int32 j=0;j<neqns_per_elem; j++){    
            if(el_eqns_row[j] > 0){
              fint_grad_[j] += Fint_grad[el_eqns_row[j]-1];
            }
        }
      }
  }
  

  void backwardGPU(double *fint_grad, const double *Fint_grad, const double *Fint, const double *fints, 
          const int32*el_eqns, int32 ngs, int32 neqns_per_elem, const int32 *neqs, const GPUDevice& d){
    int32 NEQS;
    cudaMemcpy(&NEQS, neqs, sizeof(int32), cudaMemcpyDeviceToHost);
    GpuLaunchConfig config1 = GetGpuLaunchConfig(ngs*neqns_per_elem, d);
    GpuLaunchConfig config2 = GetGpuLaunchConfig(ngs, d);

    TF_CHECK_OK(GpuLaunchKernel(
      zero_, config1.block_count, config1.thread_per_block, 0,
      d.stream(), config1.virtual_thread_count, fint_grad));

    TF_CHECK_OK(GpuLaunchKernel(
      backward_, config2.block_count, config2.thread_per_block, 0,
      d.stream(), config2.virtual_thread_count, fint_grad, Fint_grad,
      Fint, fints, el_eqns, ngs, neqns_per_elem));
    
  }

}