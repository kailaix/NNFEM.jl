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
#include "TensorRep.h"


REGISTER_OP("TensorRep")

.Input("inp : double")
.Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle inp_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &inp_shape));

        c->set_output(0, c->MakeShape({-1,2,2}));
    return Status::OK();
  });

REGISTER_OP("TensorRepGrad")

.Input("grad_out : double")
.Input("out : double")
.Input("inp : double")
.Output("grad_inp : double");


class TensorRepOp : public OpKernel {
private:
  
public:
  explicit TensorRepOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& inp = context->input(0);
    
    
    const TensorShape& inp_shape = inp.shape();
    
    
    DCHECK_EQ(inp_shape.dims(), 2);

    // extra check
        
    // create output shape
    int N = inp_shape.dim_size(0);
    TensorShape out_shape({N,2,2});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto inp_tensor = inp.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward_TensorRep(out_tensor, inp_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("TensorRep").Device(DEVICE_CPU), TensorRepOp);



class TensorRepGradOp : public OpKernel {
private:
  
public:
  explicit TensorRepGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& inp = context->input(2);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& inp_shape = inp.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 3);
    DCHECK_EQ(out_shape.dims(), 3);
    DCHECK_EQ(inp_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_inp_shape(inp_shape);
            
    // create output tensor
    
    Tensor* grad_inp = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_inp_shape, &grad_inp));
    
    // get the corresponding Eigen tensors for data access
    
    auto inp_tensor = inp.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_inp_tensor = grad_inp->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    int N = inp_shape.dim_size(0);
    forward_TensorRep(grad_inp_tensor, grad_out_tensor, out_tensor, inp_tensor, N);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("TensorRepGrad").Device(DEVICE_CPU), TensorRepGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class TensorRepOpGPU : public OpKernel {
private:
  
public:
  explicit TensorRepOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& inp = context->input(0);
    
    
    const TensorShape& inp_shape = inp.shape();
    
    
    DCHECK_EQ(inp_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1,2,2});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto inp_tensor = inp.flat<double>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("TensorRep").Device(DEVICE_GPU), TensorRepOpGPU);

class TensorRepGradOpGPU : public OpKernel {
private:
  
public:
  explicit TensorRepGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& inp = context->input(2);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& inp_shape = inp.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 3);
    DCHECK_EQ(out_shape.dims(), 3);
    DCHECK_EQ(inp_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_inp_shape(inp_shape);
            
    // create output tensor
    
    Tensor* grad_inp = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_inp_shape, &grad_inp));
    
    // get the corresponding Eigen tensors for data access
    
    auto inp_tensor = inp.flat<double>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_inp_tensor = grad_inp->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("TensorRepGrad").Device(DEVICE_GPU), TensorRepGradOpGPU);

#endif