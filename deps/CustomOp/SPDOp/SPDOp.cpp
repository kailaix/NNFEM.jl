#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include<cmath>
#include<string> 
using std::string;

namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(double *out, const double *y, const double *H0, int n, const GPUDevice &d);
  void backwardGPU(double *d_y, const double *d_out, const double *y, const double *H0, int n, const GPUDevice &d);
}

using namespace tensorflow;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "SPDOp.h"



REGISTER_OP("SPDOp")
.Input("h0 : double")
  .Input("y : double")
  .Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle h0_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &h0_shape));
        shape_inference::ShapeHandle y_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &y_shape));

        c->set_output(0, c->MakeShape({-1, 3, 3}));
    return Status::OK();
  });



REGISTER_OP("SPDOpGrad")
  
  .Input("grad_out : double")
  .Input("out : double")
  .Input("h0 : double")
  .Input("y : double")
  .Output("grad_h0 : double")
  .Output("grad_y : double");




 class SPDOpOp : public OpKernel {
private:
  
public:
  explicit SPDOpOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& h0 = context->input(0);
    const Tensor& y = context->input(1);
    
    
    const TensorShape& h0_shape = h0.shape();
    const TensorShape& y_shape = y.shape();
    
    
    DCHECK_EQ(h0_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n = y_shape.dim_size(0);
    TensorShape out_shape({n,3,3});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto h0_tensor = h0.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(out_tensor, y_tensor, h0_tensor, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("SPDOp").Device(DEVICE_CPU), SPDOpOp);


class SPDOpGradOp : public OpKernel {
private:
  
public:
  explicit SPDOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& h0 = context->input(2);
    const Tensor& y = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& h0_shape = h0.shape();
    const TensorShape& y_shape = y.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 3);
    DCHECK_EQ(out_shape.dims(), 3);
    DCHECK_EQ(h0_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
    int n = y_shape.dim_size(0);
    // create output shape
    
    TensorShape grad_h0_shape(h0_shape);
    TensorShape grad_y_shape(y_shape);
            
    // create output tensor
    
    Tensor* grad_h0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_h0_shape, &grad_h0));
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_y_shape, &grad_y));
    
    // get the corresponding Eigen tensors for data access
    
    auto h0_tensor = h0.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_h0_tensor = grad_h0->flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_y_tensor, grad_out_tensor, y_tensor, h0_tensor, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("SPDOpGrad").Device(DEVICE_CPU), SPDOpGradOp);



#ifndef NOGPU

 class SPDOpOpGPU : public OpKernel {
private:
  
public:
  explicit SPDOpOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& h0 = context->input(0);
    const Tensor& y = context->input(1);
    
    
    const TensorShape& h0_shape = h0.shape();
    const TensorShape& y_shape = y.shape();
    
    
    DCHECK_EQ(h0_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n = y_shape.dim_size(0);
    TensorShape out_shape({n,3,3});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto h0_tensor = h0.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forwardGPU(out_tensor, y_tensor, h0_tensor, n, context->eigen_gpu_device());
  }
};
REGISTER_KERNEL_BUILDER(Name("SPDOp").Device(DEVICE_GPU), SPDOpOpGPU);


class SPDOpGradOpGPU : public OpKernel {
private:
  
public:
  explicit SPDOpGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& h0 = context->input(2);
    const Tensor& y = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& h0_shape = h0.shape();
    const TensorShape& y_shape = y.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 3);
    DCHECK_EQ(out_shape.dims(), 3);
    DCHECK_EQ(h0_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
    int n = y_shape.dim_size(0);
    // create output shape
    
    TensorShape grad_h0_shape(h0_shape);
    TensorShape grad_y_shape(y_shape);
            
    // create output tensor
    
    Tensor* grad_h0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_h0_shape, &grad_h0));
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_y_shape, &grad_y));
    
    // get the corresponding Eigen tensors for data access
    
    auto h0_tensor = h0.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_h0_tensor = grad_h0->flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backwardGPU(grad_y_tensor, grad_out_tensor, y_tensor, h0_tensor, n, context->eigen_gpu_device());
  }
};
REGISTER_KERNEL_BUILDER(Name("SPDOpGrad").Device(DEVICE_GPU), SPDOpGradOpGPU);


#endif 