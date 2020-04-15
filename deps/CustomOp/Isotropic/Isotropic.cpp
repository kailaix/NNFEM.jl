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
#include "Isotropic.h"


REGISTER_OP("Isotropic")

.Input("coef : double")
.Input("strain : double")
.Output("stress : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle coef_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &coef_shape));
        shape_inference::ShapeHandle strain_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &strain_shape));

        c->set_output(0, c->Matrix(-1,3));
    return Status::OK();
  });

REGISTER_OP("IsotropicGrad")

.Input("grad_stress : double")
.Input("stress : double")
.Input("coef : double")
.Input("strain : double")
.Output("grad_coef : double")
.Output("grad_strain : double");


class IsotropicOp : public OpKernel {
private:
  
public:
  explicit IsotropicOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& coef = context->input(0);
    const Tensor& strain = context->input(1);
    
    
    const TensorShape& coef_shape = coef.shape();
    const TensorShape& strain_shape = strain.shape();
    
    
    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);

    // extra check
        
    // create output shape
    int N = coef_shape.dim_size(0);
    
    TensorShape stress_shape({N,3});
            
    // create output tensor
    
    Tensor* stress = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, stress_shape, &stress));
    
    // get the corresponding Eigen tensors for data access
    
    auto coef_tensor = coef.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto stress_tensor = stress->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(stress_tensor, coef_tensor, strain_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("Isotropic").Device(DEVICE_CPU), IsotropicOp);



class IsotropicGradOp : public OpKernel {
private:
  
public:
  explicit IsotropicGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_stress = context->input(0);
    const Tensor& stress = context->input(1);
    const Tensor& coef = context->input(2);
    const Tensor& strain = context->input(3);
    
    
    const TensorShape& grad_stress_shape = grad_stress.shape();
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& coef_shape = coef.shape();
    const TensorShape& strain_shape = strain.shape();
    
    
    DCHECK_EQ(grad_stress_shape.dims(), 2);
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_coef_shape(coef_shape);
    TensorShape grad_strain_shape(strain_shape);
            
    // create output tensor
    
    Tensor* grad_coef = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_coef_shape, &grad_coef));
    Tensor* grad_strain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_strain_shape, &grad_strain));
    
    // get the corresponding Eigen tensors for data access
    
    auto coef_tensor = coef.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto grad_stress_tensor = grad_stress.flat<double>().data();
    auto stress_tensor = stress.flat<double>().data();
    auto grad_coef_tensor = grad_coef->flat<double>().data();
    auto grad_strain_tensor = grad_strain->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = coef_shape.dim_size(0);
    backward(grad_coef_tensor, grad_strain_tensor, grad_stress_tensor, stress_tensor, coef_tensor, strain_tensor, N);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("IsotropicGrad").Device(DEVICE_CPU), IsotropicGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class IsotropicOpGPU : public OpKernel {
private:
  
public:
  explicit IsotropicOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& coef = context->input(0);
    const Tensor& strain = context->input(1);
    
    
    const TensorShape& coef_shape = coef.shape();
    const TensorShape& strain_shape = strain.shape();
    
    
    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape stress_shape({-1,3});
            
    // create output tensor
    
    Tensor* stress = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, stress_shape, &stress));
    
    // get the corresponding Eigen tensors for data access
    
    auto coef_tensor = coef.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto stress_tensor = stress->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Isotropic").Device(DEVICE_GPU), IsotropicOpGPU);

class IsotropicGradOpGPU : public OpKernel {
private:
  
public:
  explicit IsotropicGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_stress = context->input(0);
    const Tensor& stress = context->input(1);
    const Tensor& coef = context->input(2);
    const Tensor& strain = context->input(3);
    
    
    const TensorShape& grad_stress_shape = grad_stress.shape();
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& coef_shape = coef.shape();
    const TensorShape& strain_shape = strain.shape();
    
    
    DCHECK_EQ(grad_stress_shape.dims(), 2);
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_coef_shape(coef_shape);
    TensorShape grad_strain_shape(strain_shape);
            
    // create output tensor
    
    Tensor* grad_coef = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_coef_shape, &grad_coef));
    Tensor* grad_strain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_strain_shape, &grad_strain));
    
    // get the corresponding Eigen tensors for data access
    
    auto coef_tensor = coef.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto grad_stress_tensor = grad_stress.flat<double>().data();
    auto stress_tensor = stress.flat<double>().data();
    auto grad_coef_tensor = grad_coef->flat<double>().data();
    auto grad_strain_tensor = grad_strain->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("IsotropicGrad").Device(DEVICE_GPU), IsotropicGradOpGPU);

#endif