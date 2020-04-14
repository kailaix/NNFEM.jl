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
#include "IsotropicTwo.h"


REGISTER_OP("IsotropicTwo")

.Input("coef : double")
.Input("strain : double")
.Input("strainrate : double")
.Output("stress : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle coef_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &coef_shape));
        shape_inference::ShapeHandle strain_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &strain_shape));
        shape_inference::ShapeHandle strainrate_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &strainrate_shape));

        c->set_output(0, c->Matrix(-1,3));
    return Status::OK();
  });

REGISTER_OP("IsotropicTwoGrad")

.Input("grad_stress : double")
.Input("stress : double")
.Input("coef : double")
.Input("strain : double")
.Input("strainrate : double")
.Output("grad_coef : double")
.Output("grad_strain : double")
.Output("grad_strainrate : double");


class IsotropicTwoOp : public OpKernel {
private:
  
public:
  explicit IsotropicTwoOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& coef = context->input(0);
    const Tensor& strain = context->input(1);
    const Tensor& strainrate = context->input(2);
    
    
    const TensorShape& coef_shape = coef.shape();
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& strainrate_shape = strainrate.shape();
    
    
    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);
    DCHECK_EQ(strainrate_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n = strain_shape.dim_size(0);
    TensorShape stress_shape({n,3});
            
    // create output tensor
    
    Tensor* stress = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, stress_shape, &stress));
    
    // get the corresponding Eigen tensors for data access
    
    auto coef_tensor = coef.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto strainrate_tensor = strainrate.flat<double>().data();
    auto stress_tensor = stress->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(stress_tensor, strain_tensor, strainrate_tensor, coef_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("IsotropicTwo").Device(DEVICE_CPU), IsotropicTwoOp);



class IsotropicTwoGradOp : public OpKernel {
private:
  
public:
  explicit IsotropicTwoGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_stress = context->input(0);
    const Tensor& stress = context->input(1);
    const Tensor& coef = context->input(2);
    const Tensor& strain = context->input(3);
    const Tensor& strainrate = context->input(4);
    
    
    const TensorShape& grad_stress_shape = grad_stress.shape();
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& coef_shape = coef.shape();
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& strainrate_shape = strainrate.shape();
    
    
    DCHECK_EQ(grad_stress_shape.dims(), 2);
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);
    DCHECK_EQ(strainrate_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_coef_shape(coef_shape);
    TensorShape grad_strain_shape(strain_shape);
    TensorShape grad_strainrate_shape(strainrate_shape);
            
    // create output tensor
    
    Tensor* grad_coef = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_coef_shape, &grad_coef));
    Tensor* grad_strain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_strain_shape, &grad_strain));
    Tensor* grad_strainrate = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_strainrate_shape, &grad_strainrate));
    
    // get the corresponding Eigen tensors for data access
    
    auto coef_tensor = coef.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto strainrate_tensor = strainrate.flat<double>().data();
    auto grad_stress_tensor = grad_stress.flat<double>().data();
    auto stress_tensor = stress.flat<double>().data();
    auto grad_coef_tensor = grad_coef->flat<double>().data();
    auto grad_strain_tensor = grad_strain->flat<double>().data();
    auto grad_strainrate_tensor = grad_strainrate->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int n = strain_shape.dim_size(0);
    backward(
      grad_strain_tensor, grad_strainrate_tensor, grad_coef_tensor, grad_stress_tensor,
      stress_tensor, strain_tensor, strainrate_tensor, coef_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("IsotropicTwoGrad").Device(DEVICE_CPU), IsotropicTwoGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class IsotropicTwoOpGPU : public OpKernel {
private:
  
public:
  explicit IsotropicTwoOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& coef = context->input(0);
    const Tensor& strain = context->input(1);
    const Tensor& strainrate = context->input(2);
    
    
    const TensorShape& coef_shape = coef.shape();
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& strainrate_shape = strainrate.shape();
    
    
    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);
    DCHECK_EQ(strainrate_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape stress_shape({-1,3});
            
    // create output tensor
    
    Tensor* stress = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, stress_shape, &stress));
    
    // get the corresponding Eigen tensors for data access
    
    auto coef_tensor = coef.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto strainrate_tensor = strainrate.flat<double>().data();
    auto stress_tensor = stress->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("IsotropicTwo").Device(DEVICE_GPU), IsotropicTwoOpGPU);

class IsotropicTwoGradOpGPU : public OpKernel {
private:
  
public:
  explicit IsotropicTwoGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_stress = context->input(0);
    const Tensor& stress = context->input(1);
    const Tensor& coef = context->input(2);
    const Tensor& strain = context->input(3);
    const Tensor& strainrate = context->input(4);
    
    
    const TensorShape& grad_stress_shape = grad_stress.shape();
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& coef_shape = coef.shape();
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& strainrate_shape = strainrate.shape();
    
    
    DCHECK_EQ(grad_stress_shape.dims(), 2);
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(coef_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);
    DCHECK_EQ(strainrate_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_coef_shape(coef_shape);
    TensorShape grad_strain_shape(strain_shape);
    TensorShape grad_strainrate_shape(strainrate_shape);
            
    // create output tensor
    
    Tensor* grad_coef = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_coef_shape, &grad_coef));
    Tensor* grad_strain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_strain_shape, &grad_strain));
    Tensor* grad_strainrate = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_strainrate_shape, &grad_strainrate));
    
    // get the corresponding Eigen tensors for data access
    
    auto coef_tensor = coef.flat<double>().data();
    auto strain_tensor = strain.flat<double>().data();
    auto strainrate_tensor = strainrate.flat<double>().data();
    auto grad_stress_tensor = grad_stress.flat<double>().data();
    auto stress_tensor = stress.flat<double>().data();
    auto grad_coef_tensor = grad_coef->flat<double>().data();
    auto grad_strain_tensor = grad_strain->flat<double>().data();
    auto grad_strainrate_tensor = grad_strainrate->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("IsotropicTwoGrad").Device(DEVICE_GPU), IsotropicTwoGradOpGPU);

#endif