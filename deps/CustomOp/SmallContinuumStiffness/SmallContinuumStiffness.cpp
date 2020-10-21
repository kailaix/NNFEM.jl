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
#include "SmallContinuumStiffness.h"


REGISTER_OP("SmallContinuumStiffness")

.Input("k : double")
.Output("ii : int64")
.Output("jj : int64")
.Output("vv : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle k_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &k_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("SmallContinuumStiffnessGrad")

.Input("grad_vv : double")
.Input("ii : int64")
.Input("jj : int64")
.Input("vv : double")
.Input("k : double")
.Output("grad_k : double");


class SmallContinuumStiffnessOp : public OpKernel {
private:
  
public:
  explicit SmallContinuumStiffnessOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& k = context->input(0);
    
    
    const TensorShape& k_shape = k.shape();
    
    
    DCHECK_EQ(k_shape.dims(), 3);

    // extra check
        
    // create output shape
    
    int N = forward_count0();
    TensorShape ii_shape({N});
    TensorShape jj_shape({N});
    TensorShape vv_shape({N});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto k_tensor = k.flat<double>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward_SmallContinuumStiffness(ii_tensor, jj_tensor, vv_tensor, k_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("SmallContinuumStiffness").Device(DEVICE_CPU), SmallContinuumStiffnessOp);



class SmallContinuumStiffnessGradOp : public OpKernel {
private:
  
public:
  explicit SmallContinuumStiffnessGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& k = context->input(4);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& k_shape = k.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(k_shape.dims(), 3);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_k_shape(k_shape);
            
    // create output tensor
    
    Tensor* grad_k = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_k_shape, &grad_k));
    
    // get the corresponding Eigen tensors for data access
    
    auto k_tensor = k.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_k_tensor = grad_k->flat<double>().data();   

    // implement your backward function here 

    // TODO:

    forward_SmallContinuumStiffness(
      grad_k_tensor, grad_vv_tensor, vv_tensor, k_tensor);

    
  }
};
REGISTER_KERNEL_BUILDER(Name("SmallContinuumStiffnessGrad").Device(DEVICE_CPU), SmallContinuumStiffnessGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class SmallContinuumStiffnessOpGPU : public OpKernel {
private:
  
public:
  explicit SmallContinuumStiffnessOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& k = context->input(0);
    
    
    const TensorShape& k_shape = k.shape();
    
    
    DCHECK_EQ(k_shape.dims(), 3);

    // extra check
        
    // create output shape
    
    TensorShape ii_shape({-1});
    TensorShape jj_shape({-1});
    TensorShape vv_shape({-1});
            
    // create output tensor
    
    Tensor* ii = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii));
    Tensor* jj = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj));
    Tensor* vv = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv));
    
    // get the corresponding Eigen tensors for data access
    
    auto k_tensor = k.flat<double>().data();
    auto ii_tensor = ii->flat<int64>().data();
    auto jj_tensor = jj->flat<int64>().data();
    auto vv_tensor = vv->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("SmallContinuumStiffness").Device(DEVICE_GPU), SmallContinuumStiffnessOpGPU);

class SmallContinuumStiffnessGradOpGPU : public OpKernel {
private:
  
public:
  explicit SmallContinuumStiffnessGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_vv = context->input(0);
    const Tensor& ii = context->input(1);
    const Tensor& jj = context->input(2);
    const Tensor& vv = context->input(3);
    const Tensor& k = context->input(4);
    
    
    const TensorShape& grad_vv_shape = grad_vv.shape();
    const TensorShape& ii_shape = ii.shape();
    const TensorShape& jj_shape = jj.shape();
    const TensorShape& vv_shape = vv.shape();
    const TensorShape& k_shape = k.shape();
    
    
    DCHECK_EQ(grad_vv_shape.dims(), 1);
    DCHECK_EQ(ii_shape.dims(), 1);
    DCHECK_EQ(jj_shape.dims(), 1);
    DCHECK_EQ(vv_shape.dims(), 1);
    DCHECK_EQ(k_shape.dims(), 3);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_k_shape(k_shape);
            
    // create output tensor
    
    Tensor* grad_k = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_k_shape, &grad_k));
    
    // get the corresponding Eigen tensors for data access
    
    auto k_tensor = k.flat<double>().data();
    auto grad_vv_tensor = grad_vv.flat<double>().data();
    auto ii_tensor = ii.flat<int64>().data();
    auto jj_tensor = jj.flat<int64>().data();
    auto vv_tensor = vv.flat<double>().data();
    auto grad_k_tensor = grad_k->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("SmallContinuumStiffnessGrad").Device(DEVICE_GPU), SmallContinuumStiffnessGradOpGPU);

#endif