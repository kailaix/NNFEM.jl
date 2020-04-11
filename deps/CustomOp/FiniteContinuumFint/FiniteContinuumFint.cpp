#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>


#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif
using namespace tensorflow;
#include "FiniteContinuumFint.h"


REGISTER_OP("FiniteContinuumFint")

.Input("stress : double")
.Input("state : double")
.Output("fint : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle stress_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &stress_shape));
        shape_inference::ShapeHandle state_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &state_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("FiniteContinuumFintGrad")

.Input("grad_fint : double")
.Input("fint : double")
.Input("stress : double")
.Input("state : double")
.Output("grad_stress : double")
.Output("grad_state : double");


class FiniteContinuumFintOp : public OpKernel {
private:
  
public:
  explicit FiniteContinuumFintOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& stress = context->input(0);
    const Tensor& state = context->input(1);
    
    
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& state_shape = state.shape();
    
    
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(state_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape fint_shape({domain.neqs});
            
    // create output tensor
    
    Tensor* fint = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, fint_shape, &fint));
    
    // get the corresponding Eigen tensors for data access
    
    auto stress_tensor = stress.flat<double>().data();
    auto state_tensor = state.flat<double>().data();
    auto fint_tensor = fint->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(fint_tensor, stress_tensor, state_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("FiniteContinuumFint").Device(DEVICE_CPU), FiniteContinuumFintOp);



class FiniteContinuumFintGradOp : public OpKernel {
private:
  
public:
  explicit FiniteContinuumFintGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_fint = context->input(0);
    const Tensor& fint = context->input(1);
    const Tensor& stress = context->input(2);
    const Tensor& state = context->input(3);
    
    
    const TensorShape& grad_fint_shape = grad_fint.shape();
    const TensorShape& fint_shape = fint.shape();
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& state_shape = state.shape();
    
    
    DCHECK_EQ(grad_fint_shape.dims(), 1);
    DCHECK_EQ(fint_shape.dims(), 1);
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(state_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_stress_shape(stress_shape);
    TensorShape grad_state_shape(state_shape);
            
    // create output tensor
    
    Tensor* grad_stress = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_stress_shape, &grad_stress));
    Tensor* grad_state = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_state_shape, &grad_state));
    
    // get the corresponding Eigen tensors for data access
    
    auto stress_tensor = stress.flat<double>().data();
    auto state_tensor = state.flat<double>().data();
    auto grad_fint_tensor = grad_fint.flat<double>().data();
    auto fint_tensor = fint.flat<double>().data();
    auto grad_stress_tensor = grad_stress->flat<double>().data();
    auto grad_state_tensor = grad_state->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_stress_tensor, grad_state_tensor, grad_fint_tensor, fint_tensor, stress_tensor, state_tensor);

    
  }
};
REGISTER_KERNEL_BUILDER(Name("FiniteContinuumFintGrad").Device(DEVICE_CPU), FiniteContinuumFintGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class FiniteContinuumFintOpGPU : public OpKernel {
private:
  
public:
  explicit FiniteContinuumFintOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& stress = context->input(0);
    const Tensor& state = context->input(1);
    
    
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& state_shape = state.shape();
    
    
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(state_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape fint_shape({-1});
            
    // create output tensor
    
    Tensor* fint = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, fint_shape, &fint));
    
    // get the corresponding Eigen tensors for data access
    
    auto stress_tensor = stress.flat<double>().data();
    auto state_tensor = state.flat<double>().data();
    auto fint_tensor = fint->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FiniteContinuumFint").Device(DEVICE_GPU), FiniteContinuumFintOpGPU);

class FiniteContinuumFintGradOpGPU : public OpKernel {
private:
  
public:
  explicit FiniteContinuumFintGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_fint = context->input(0);
    const Tensor& fint = context->input(1);
    const Tensor& stress = context->input(2);
    const Tensor& state = context->input(3);
    
    
    const TensorShape& grad_fint_shape = grad_fint.shape();
    const TensorShape& fint_shape = fint.shape();
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& state_shape = state.shape();
    
    
    DCHECK_EQ(grad_fint_shape.dims(), 1);
    DCHECK_EQ(fint_shape.dims(), 1);
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(state_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_stress_shape(stress_shape);
    TensorShape grad_state_shape(state_shape);
            
    // create output tensor
    
    Tensor* grad_stress = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_stress_shape, &grad_stress));
    Tensor* grad_state = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_state_shape, &grad_state));
    
    // get the corresponding Eigen tensors for data access
    
    auto stress_tensor = stress.flat<double>().data();
    auto state_tensor = state.flat<double>().data();
    auto grad_fint_tensor = grad_fint.flat<double>().data();
    auto fint_tensor = fint.flat<double>().data();
    auto grad_stress_tensor = grad_stress->flat<double>().data();
    auto grad_state_tensor = grad_state->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FiniteContinuumFintGrad").Device(DEVICE_GPU), FiniteContinuumFintGradOpGPU);

#endif