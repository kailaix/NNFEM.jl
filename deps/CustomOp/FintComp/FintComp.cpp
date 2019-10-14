#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "FintComp.h"

namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void backward(double *fint_grad, const double *Fint_grad, const double *Fint, const double *fints, 
          const int32*el_eqns, int32 ngs, int32 neqns_per_elem, int32 *neqs);
  void forward(double *Fint, const double *fints, const int32*el_eqns, int32 ngs, 
      int32 neqns_per_elem, int32 *neqs, const GPUDevice& d);
}

using namespace tensorflow;

REGISTER_OP("FintComp")

.Input("fints : double")
  .Input("el : int32")
  .Input("neqs : int32")
  .Output("out : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle fints_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &fints_shape));
        shape_inference::ShapeHandle el_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &el_shape));
        shape_inference::ShapeHandle neqs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &neqs_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("FintCompGrad")

.Input("grad_out : double")
  .Input("out : double")
  .Input("fints : double")
  .Input("el : int32")
  .Input("neqs : int32")
  .Output("grad_fints : double")
  .Output("grad_el : int32")
  .Output("grad_neqs : int32");


class FintCompOp : public OpKernel {
private:
  
public:
  explicit FintCompOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& fints = context->input(0);
    const Tensor& el = context->input(1);
    const Tensor& neqs = context->input(2);
    
    
    const TensorShape& fints_shape = fints.shape();
    const TensorShape& el_shape = el.shape();
    const TensorShape& neqs_shape = neqs.shape();
    
    
    DCHECK_EQ(fints_shape.dims(), 2);
    DCHECK_EQ(el_shape.dims(), 2);
    DCHECK_EQ(neqs_shape.dims(), 0);

    // extra check
        
    // create output shape
    int32 ngs = fints_shape.dim_size(0);
    int32 neqns_per_elem = fints_shape.dim_size(1);
    TensorShape out_shape({*neqs.flat<int32>().data()});

    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto fints_tensor = fints.flat<double>().data();
    auto el_tensor = el.flat<int32>().data();
    auto neqs_tensor = neqs.flat<int32>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(out_tensor, fints_tensor, el_tensor, ngs, neqns_per_elem, *neqs_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("FintComp").Device(DEVICE_CPU), FintCompOp);



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
    const Tensor& neqs = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& fints_shape = fints.shape();
    const TensorShape& el_shape = el.shape();
    const TensorShape& neqs_shape = neqs.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(fints_shape.dims(), 2);
    DCHECK_EQ(el_shape.dims(), 2);
    DCHECK_EQ(neqs_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
    int32 ngs = fints_shape.dim_size(0);
    int32 neqns_per_elem = fints_shape.dim_size(1);

    // create output shape
    
    TensorShape grad_fints_shape(fints_shape);
    TensorShape grad_el_shape(el_shape);
    TensorShape grad_neqs_shape(neqs_shape);
            
    // create output tensor
    
    Tensor* grad_fints = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_fints_shape, &grad_fints));
    Tensor* grad_el = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_el_shape, &grad_el));
    Tensor* grad_neqs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_neqs_shape, &grad_neqs));
    
    // get the corresponding Eigen tensors for data access
    
    auto fints_tensor = fints.flat<double>().data();
    auto el_tensor = el.flat<int32>().data();
    auto neqs_tensor = neqs.flat<int32>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_fints_tensor = grad_fints->flat<double>().data();
    auto grad_el_tensor = grad_el->flat<int32>().data();
    auto grad_neqs_tensor = grad_neqs->flat<int32>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_fints_tensor, grad_out_tensor, out_tensor, fints_tensor, el_tensor, ngs, neqns_per_elem, *neqs_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("FintCompGrad").Device(DEVICE_CPU), FintCompGradOp);

#ifdef USE_GPU
class FintCompOpGPU : public OpKernel {
private:
  
public:
  explicit FintCompOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& fints = context->input(0);
    const Tensor& el = context->input(1);
    const Tensor& neqs = context->input(2);
    
    
    const TensorShape& fints_shape = fints.shape();
    const TensorShape& el_shape = el.shape();
    const TensorShape& neqs_shape = neqs.shape();
    
    
    DCHECK_EQ(fints_shape.dims(), 2);
    DCHECK_EQ(el_shape.dims(), 2);
    DCHECK_EQ(neqs_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape out_shape({-1});
            
    // create output tensor
    
    Tensor* out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    
    // get the corresponding Eigen tensors for data access
    
    auto fints_tensor = fints.flat<double>().data();
    auto el_tensor = el.flat<int32>().data();
    auto neqs_tensor = neqs.flat<int32>().data();
    auto out_tensor = out->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("FintComp").Device(DEVICE_GPU), FintCompOpGPU);



class FintCompGradOpGPU : public OpKernel {
private:
  
public:
  explicit FintCompGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_out = context->input(0);
    const Tensor& out = context->input(1);
    const Tensor& fints = context->input(2);
    const Tensor& el = context->input(3);
    const Tensor& neqs = context->input(4);
    
    
    const TensorShape& grad_out_shape = grad_out.shape();
    const TensorShape& out_shape = out.shape();
    const TensorShape& fints_shape = fints.shape();
    const TensorShape& el_shape = el.shape();
    const TensorShape& neqs_shape = neqs.shape();
    
    
    DCHECK_EQ(grad_out_shape.dims(), 1);
    DCHECK_EQ(out_shape.dims(), 1);
    DCHECK_EQ(fints_shape.dims(), 2);
    DCHECK_EQ(el_shape.dims(), 2);
    DCHECK_EQ(neqs_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_fints_shape(fints_shape);
    TensorShape grad_el_shape(el_shape);
    TensorShape grad_neqs_shape(neqs_shape);
            
    // create output tensor
    
    Tensor* grad_fints = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_fints_shape, &grad_fints));
    Tensor* grad_el = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_el_shape, &grad_el));
    Tensor* grad_neqs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_neqs_shape, &grad_neqs));
    
    // get the corresponding Eigen tensors for data access
    
    auto fints_tensor = fints.flat<double>().data();
    auto el_tensor = el.flat<int32>().data();
    auto neqs_tensor = neqs.flat<int32>().data();
    auto grad_out_tensor = grad_out.flat<double>().data();
    auto out_tensor = out.flat<double>().data();
    auto grad_fints_tensor = grad_fints->flat<double>().data();
    auto grad_el_tensor = grad_el->flat<int32>().data();
    auto grad_neqs_tensor = grad_neqs->flat<int32>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FintCompGrad").Device(DEVICE_GPU), FintCompGradOpGPU);

#endif