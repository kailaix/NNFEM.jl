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
#include "Plasticity.h"


REGISTER_OP("Plasticity")

.Input("val : double")
.Input("h : double")
.Output("de : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle val_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &val_shape));
        shape_inference::ShapeHandle h_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &h_shape));

        c->set_output(0, c->MakeShape({-1,3,3}));
    return Status::OK();
  });

REGISTER_OP("PlasticityGrad")

.Input("grad_de : double")
.Input("de : double")
.Input("val : double")
.Input("h : double")
.Output("grad_val : double")
.Output("grad_h : double");


class PlasticityOp : public OpKernel {
private:
  
public:
  explicit PlasticityOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& val = context->input(0);
    const Tensor& h = context->input(1);
    
    
    const TensorShape& val_shape = val.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(val_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 2);

    // extra check
        
    // create output shape
    int N = val_shape.dim_size(0);
    TensorShape de_shape({N,3,3});
            
    // create output tensor
    
    Tensor* de = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, de_shape, &de));
    
    // get the corresponding Eigen tensors for data access
    
    auto val_tensor = val.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto de_tensor = de->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(de_tensor, h_tensor, val_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("Plasticity").Device(DEVICE_CPU), PlasticityOp);



class PlasticityGradOp : public OpKernel {
private:
  
public:
  explicit PlasticityGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_de = context->input(0);
    const Tensor& de = context->input(1);
    const Tensor& val = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& grad_de_shape = grad_de.shape();
    const TensorShape& de_shape = de.shape();
    const TensorShape& val_shape = val.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_de_shape.dims(), 3);
    DCHECK_EQ(de_shape.dims(), 3);
    DCHECK_EQ(val_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_val_shape(val_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_val = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_val_shape, &grad_val));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto val_tensor = val.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_de_tensor = grad_de.flat<double>().data();
    auto de_tensor = de.flat<double>().data();
    auto grad_val_tensor = grad_val->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = val_shape.dim_size(0);
    backward(grad_val_tensor, grad_h_tensor, grad_de_tensor, de_tensor, h_tensor, val_tensor, N);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PlasticityGrad").Device(DEVICE_CPU), PlasticityGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class PlasticityOpGPU : public OpKernel {
private:
  
public:
  explicit PlasticityOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& val = context->input(0);
    const Tensor& h = context->input(1);
    
    
    const TensorShape& val_shape = val.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(val_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape de_shape({-1,3,3});
            
    // create output tensor
    
    Tensor* de = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, de_shape, &de));
    
    // get the corresponding Eigen tensors for data access
    
    auto val_tensor = val.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto de_tensor = de->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Plasticity").Device(DEVICE_GPU), PlasticityOpGPU);

class PlasticityGradOpGPU : public OpKernel {
private:
  
public:
  explicit PlasticityGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_de = context->input(0);
    const Tensor& de = context->input(1);
    const Tensor& val = context->input(2);
    const Tensor& h = context->input(3);
    
    
    const TensorShape& grad_de_shape = grad_de.shape();
    const TensorShape& de_shape = de.shape();
    const TensorShape& val_shape = val.shape();
    const TensorShape& h_shape = h.shape();
    
    
    DCHECK_EQ(grad_de_shape.dims(), 3);
    DCHECK_EQ(de_shape.dims(), 3);
    DCHECK_EQ(val_shape.dims(), 2);
    DCHECK_EQ(h_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_val_shape(val_shape);
    TensorShape grad_h_shape(h_shape);
            
    // create output tensor
    
    Tensor* grad_val = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_val_shape, &grad_val));
    Tensor* grad_h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_h_shape, &grad_h));
    
    // get the corresponding Eigen tensors for data access
    
    auto val_tensor = val.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_de_tensor = grad_de.flat<double>().data();
    auto de_tensor = de.flat<double>().data();
    auto grad_val_tensor = grad_val->flat<double>().data();
    auto grad_h_tensor = grad_h->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PlasticityGrad").Device(DEVICE_GPU), PlasticityGradOpGPU);

#endif