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
#include "CholOp.h"

REGISTER_OP("CholOp")

.Input("x : double")
  .Output("y : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle x_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x_shape));

        c->set_output(0, c->Matrix(-1,9));
    return Status::OK();
  });
class CholOpOp : public OpKernel {
private:
  
public:
  explicit CholOpOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& x = context->input(0);
    
    
    const TensorShape& x_shape = x.shape();
    
    
    DCHECK_EQ(x_shape.dims(), 2);

    // extra check
        
    // create output shape
    int n = x_shape.dim_size(0);
    TensorShape y_shape({n,9});
            
    // create output tensor
    
    Tensor* y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_shape, &y));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto y_tensor = y->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward_CholOp(y_tensor, x_tensor, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("CholOp").Device(DEVICE_CPU), CholOpOp);


REGISTER_OP("CholOpGrad")
  
  .Input("grad_y : double")
  .Input("y : double")
  .Input("x : double")
  .Output("grad_x : double");
class CholOpGradOp : public OpKernel {
private:
  
public:
  explicit CholOpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_y = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& x = context->input(2);
    
    
    const TensorShape& grad_y_shape = grad_y.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& x_shape = x.shape();
    
    
    DCHECK_EQ(grad_y_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);
    DCHECK_EQ(x_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    int n = x_shape.dim_size(0);
    TensorShape grad_x_shape(x_shape);
            
    // create output tensor
    
    Tensor* grad_x = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape, &grad_x));
    
    // get the corresponding Eigen tensors for data access
    
    auto x_tensor = x.flat<double>().data();
    auto grad_y_tensor = grad_y.flat<double>().data();
    auto y_tensor = y.flat<double>().data();
    auto grad_x_tensor = grad_x->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    forward_CholOp(grad_x_tensor, grad_y_tensor, x_tensor, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("CholOpGrad").Device(DEVICE_CPU), CholOpGradOp);

