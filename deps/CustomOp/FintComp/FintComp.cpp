#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using namespace tensorflow;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "FintComp.h"

REGISTER_OP("FintComp")

.Input("fints : double")
  .Input("el : int")
  .Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle fints_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &fints_shape));
        shape_inference::ShapeHandle el_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &el_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });
class FintCompOp : public OpKernel {
private:
  
public:
  explicit FintCompOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& fints = context->input(0);
    const Tensor& el = context->input(1);
    
    
    const TensorShape& fints_shape = fints.shape();
    const TensorShape& el_shape = el.shape();
    
    
    DCHECK_EQ(fints_shape.dims(), 2);
    DCHECK_EQ(el_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    int ngs = fints_shape.dim_size(0);
    int neqns_per_elem = fints_shape.dim_size(1);
    TensorShape out_shape({ngs*neqns_per_elem});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto fints_tensor = fints.flat<double>().data();
    auto el_tensor = el.flat<int>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
  forward(out_tensor, fints_tensor, el_tensor, ngs, neqns_per_elem);
  }
};
REGISTER_KERNEL_BUILDER(Name("FintComp").Device(DEVICE_CPU), FintCompOp);


REGISTER_OP("FintCompGrad")
  
  .Input("grad_out : double")
  .Input("out : double")
  .Input("fints : double")
  .Input("el : int")
  .Output("grad_fints : double")
  .Output("grad_el : int");
class FintCompGradOp : public OpKernel {
private:
  
public:
  explicit FintCompGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& fints = context->input(2);
    const Tensor& el = context->input(3);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& fints_shape = fints.shape();
    const TensorShape& el_shape = el.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(fints_shape.dims(), 2);
    DCHECK_EQ(el_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
    int ngs = fints_shape.dim_size(0);
    int neqns_per_elem = fints_shape.dim_size(1);

    // create output shape
    
    TensorShape grad_fints_shape(fints_shape);
    TensorShape grad_el_shape(el_shape);
            
    // create output tensor
    
    Tensor* grad_fints = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_fints_shape, &grad_fints));
    Tensor* grad_el = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_el_shape, &grad_el));
    
    // get the corresponding Eigen tensors for data access
    
    auto fints_tensor = fints.flat<double>().data();
    auto el_tensor = el.flat<int>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_fints_tensor = grad_fints->flat<double>().data();
    auto grad_el_tensor = grad_el->flat<int>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_fints_tensor, grad_out_tensor, out_tensor, fints_tensor, el_tensor, ngs, neqns_per_elem);
  }
};
REGISTER_KERNEL_BUILDER(Name("FintCompGrad").Device(DEVICE_CPU), FintCompGradOp);

