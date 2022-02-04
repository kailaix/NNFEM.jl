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
#include "RivlinSaunders.h"


REGISTER_OP("RivlinSaunders")

.Input("strain : double")
.Input("c1 : double")
.Input("c2 : double")
.Output("stress : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle strain_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &strain_shape));
        shape_inference::ShapeHandle c1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &c1_shape));
        shape_inference::ShapeHandle c2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &c2_shape));

        c->set_output(0, c->Matrix(-1,3));
    return Status::OK();
  });

REGISTER_OP("RivlinSaundersGrad")

.Input("grad_stress : double")
.Input("stress : double")
.Input("strain : double")
.Input("c1 : double")
.Input("c2 : double")
.Output("grad_strain : double")
.Output("grad_c1 : double")
.Output("grad_c2 : double");


class RivlinSaundersOp : public OpKernel {
private:
  
public:
  explicit RivlinSaundersOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& strain = context->input(0);
    const Tensor& c1 = context->input(1);
    const Tensor& c2 = context->input(2);
    
    
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& c1_shape = c1.shape();
    const TensorShape& c2_shape = c2.shape();
    
    
    DCHECK_EQ(strain_shape.dims(), 2);
    DCHECK_EQ(c1_shape.dims(), 0);
    DCHECK_EQ(c2_shape.dims(), 0);

    // extra check
        
    // create output shape
    int n = strain_shape.dim_size(0);
    TensorShape stress_shape({n,3});
            
    // create output tensor
    
    Tensor* stress = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, stress_shape, &stress));
    
    // get the corresponding Eigen tensors for data access
    
    auto strain_tensor = strain.flat<double>().data();
    auto c1_tensor = c1.flat<double>().data();
    auto c2_tensor = c2.flat<double>().data();
    auto stress_tensor = stress->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward_RivlinSaunders(stress_tensor, strain_tensor, *c1_tensor, *c2_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("RivlinSaunders").Device(DEVICE_CPU), RivlinSaundersOp);



class RivlinSaundersGradOp : public OpKernel {
private:
  
public:
  explicit RivlinSaundersGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_stress = context->input(0);
    const Tensor& stress = context->input(1);
    const Tensor& strain = context->input(2);
    const Tensor& c1 = context->input(3);
    const Tensor& c2 = context->input(4);
    
    
    const TensorShape& grad_stress_shape = grad_stress.shape();
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& c1_shape = c1.shape();
    const TensorShape& c2_shape = c2.shape();
    
    
    DCHECK_EQ(grad_stress_shape.dims(), 2);
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);
    DCHECK_EQ(c1_shape.dims(), 0);
    DCHECK_EQ(c2_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_strain_shape(strain_shape);
    TensorShape grad_c1_shape(c1_shape);
    TensorShape grad_c2_shape(c2_shape);
            
    // create output tensor
    
    Tensor* grad_strain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_strain_shape, &grad_strain));
    Tensor* grad_c1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_c1_shape, &grad_c1));
    Tensor* grad_c2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c2_shape, &grad_c2));
    
    // get the corresponding Eigen tensors for data access
    
    auto strain_tensor = strain.flat<double>().data();
    auto c1_tensor = c1.flat<double>().data();
    auto c2_tensor = c2.flat<double>().data();
    auto grad_stress_tensor = grad_stress.flat<double>().data();
    auto stress_tensor = stress.flat<double>().data();
    auto grad_strain_tensor = grad_strain->flat<double>().data();
    auto grad_c1_tensor = grad_c1->flat<double>().data();
    auto grad_c2_tensor = grad_c2->flat<double>().data();   

    // implement your backward function here 

    int n = strain_shape.dim_size(0);
    // TODO:
    forward_RivlinSaunders(
      grad_strain_tensor, grad_c1_tensor, grad_c2_tensor,
      grad_stress_tensor,
      stress_tensor, strain_tensor, *c1_tensor, *c2_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("RivlinSaundersGrad").Device(DEVICE_CPU), RivlinSaundersGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class RivlinSaundersOpGPU : public OpKernel {
private:
  
public:
  explicit RivlinSaundersOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& strain = context->input(0);
    const Tensor& c1 = context->input(1);
    const Tensor& c2 = context->input(2);
    
    
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& c1_shape = c1.shape();
    const TensorShape& c2_shape = c2.shape();
    
    
    DCHECK_EQ(strain_shape.dims(), 2);
    DCHECK_EQ(c1_shape.dims(), 0);
    DCHECK_EQ(c2_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape stress_shape({-1,3});
            
    // create output tensor
    
    Tensor* stress = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, stress_shape, &stress));
    
    // get the corresponding Eigen tensors for data access
    
    auto strain_tensor = strain.flat<double>().data();
    auto c1_tensor = c1.flat<double>().data();
    auto c2_tensor = c2.flat<double>().data();
    auto stress_tensor = stress->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("RivlinSaunders").Device(DEVICE_GPU), RivlinSaundersOpGPU);

class RivlinSaundersGradOpGPU : public OpKernel {
private:
  
public:
  explicit RivlinSaundersGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_stress = context->input(0);
    const Tensor& stress = context->input(1);
    const Tensor& strain = context->input(2);
    const Tensor& c1 = context->input(3);
    const Tensor& c2 = context->input(4);
    
    
    const TensorShape& grad_stress_shape = grad_stress.shape();
    const TensorShape& stress_shape = stress.shape();
    const TensorShape& strain_shape = strain.shape();
    const TensorShape& c1_shape = c1.shape();
    const TensorShape& c2_shape = c2.shape();
    
    
    DCHECK_EQ(grad_stress_shape.dims(), 2);
    DCHECK_EQ(stress_shape.dims(), 2);
    DCHECK_EQ(strain_shape.dims(), 2);
    DCHECK_EQ(c1_shape.dims(), 0);
    DCHECK_EQ(c2_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_strain_shape(strain_shape);
    TensorShape grad_c1_shape(c1_shape);
    TensorShape grad_c2_shape(c2_shape);
            
    // create output tensor
    
    Tensor* grad_strain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_strain_shape, &grad_strain));
    Tensor* grad_c1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_c1_shape, &grad_c1));
    Tensor* grad_c2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c2_shape, &grad_c2));
    
    // get the corresponding Eigen tensors for data access
    
    auto strain_tensor = strain.flat<double>().data();
    auto c1_tensor = c1.flat<double>().data();
    auto c2_tensor = c2.flat<double>().data();
    auto grad_stress_tensor = grad_stress.flat<double>().data();
    auto stress_tensor = stress.flat<double>().data();
    auto grad_strain_tensor = grad_strain->flat<double>().data();
    auto grad_c1_tensor = grad_c1->flat<double>().data();
    auto grad_c2_tensor = grad_c2->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("RivlinSaundersGrad").Device(DEVICE_GPU), RivlinSaundersGradOpGPU);

#endif