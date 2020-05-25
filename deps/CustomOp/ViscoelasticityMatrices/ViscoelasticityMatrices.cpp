#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>

// Signatures for GPU kernels here 


using namespace tensorflow;
#include "ViscoelasticityMatrices.h"


REGISTER_OP("ViscoelasticityMatrices")
.Input("mu : double")
.Input("eta : double")
.Input("lambda : double")
.Input("dt : double")
.Output("s : double")
.Output("h : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle mu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &mu_shape));
        shape_inference::ShapeHandle eta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &eta_shape));
        shape_inference::ShapeHandle lambda_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &lambda_shape));
        shape_inference::ShapeHandle dt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &dt_shape));

        c->set_output(0, c->MakeShape({-1,3,3}));
        c->set_output(1, c->MakeShape({-1,3,3}));
    return Status::OK();
  });

REGISTER_OP("ViscoelasticityMatricesGrad")
.Input("grad_s : double")
.Input("grad_h : double")
.Input("s : double")
.Input("h : double")
.Input("mu : double")
.Input("eta : double")
.Input("lambda : double")
.Input("dt : double")
.Output("grad_mu : double")
.Output("grad_eta : double")
.Output("grad_lambda : double")
.Output("grad_dt : double");

/*-------------------------------------------------------------------------------------*/

class ViscoelasticityMatricesOp : public OpKernel {
private:
  
public:
  explicit ViscoelasticityMatricesOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& mu = context->input(0);
    const Tensor& eta = context->input(1);
    const Tensor& lambda = context->input(2);
    const Tensor& dt = context->input(3);
    
    
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& eta_shape = eta.shape();
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& dt_shape = dt.shape();
    
    
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 1);
    DCHECK_EQ(lambda_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);

    // extra check
        
    // create output shape
    int N = domain.ngauss;
    TensorShape s_shape({N,3,3});
    TensorShape h_shape({N,3,3});
            
    // create output tensor
    
    Tensor* s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, s_shape, &s));
    Tensor* h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, h_shape, &h));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto lambda_tensor = lambda.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto s_tensor = s->flat<double>().data();
    auto h_tensor = h->flat<double>().data();   

    // implement your forward function here 

    forward(s_tensor, h_tensor, mu_tensor, eta_tensor, lambda_tensor, *dt_tensor);

  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoelasticityMatrices").Device(DEVICE_CPU), ViscoelasticityMatricesOp);



class ViscoelasticityMatricesGradOp : public OpKernel {
private:
  
public:
  explicit ViscoelasticityMatricesGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_s = context->input(0);
    const Tensor& grad_h = context->input(1);
    const Tensor& s = context->input(2);
    const Tensor& h = context->input(3);
    const Tensor& mu = context->input(4);
    const Tensor& eta = context->input(5);
    const Tensor& lambda = context->input(6);
    const Tensor& dt = context->input(7);
    
    
    const TensorShape& grad_s_shape = grad_s.shape();
    const TensorShape& grad_h_shape = grad_h.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& eta_shape = eta.shape();
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& dt_shape = dt.shape();
    
    
    DCHECK_EQ(grad_s_shape.dims(), 3);
    DCHECK_EQ(grad_h_shape.dims(), 3);
    DCHECK_EQ(s_shape.dims(), 3);
    DCHECK_EQ(h_shape.dims(), 3);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 1);
    DCHECK_EQ(lambda_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_eta_shape(eta_shape);
    TensorShape grad_lambda_shape(lambda_shape);
    TensorShape grad_dt_shape(dt_shape);
            
    // create output tensor
    
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_mu_shape, &grad_mu));
    Tensor* grad_eta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_eta_shape, &grad_eta));
    Tensor* grad_lambda = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_lambda_shape, &grad_lambda));
    Tensor* grad_dt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_dt_shape, &grad_dt));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto lambda_tensor = lambda.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto grad_s_tensor = grad_s.flat<double>().data();
    auto grad_h_tensor = grad_h.flat<double>().data();
    auto s_tensor = s.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_eta_tensor = grad_eta->flat<double>().data();
    auto grad_lambda_tensor = grad_lambda->flat<double>().data();
    auto grad_dt_tensor = grad_dt->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    grad_mu->flat<double>().setZero();
    grad_eta->flat<double>().setZero();
    grad_lambda->flat<double>().setZero();
        backward(
          grad_mu_tensor, grad_eta_tensor, grad_lambda_tensor,
          grad_s_tensor, grad_h_tensor,
          s_tensor, h_tensor, mu_tensor, eta_tensor, lambda_tensor, *dt_tensor);

    
  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoelasticityMatricesGrad").Device(DEVICE_CPU), ViscoelasticityMatricesGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef GOOGLE_CUDA
class ViscoelasticityMatricesOpGPU : public OpKernel {
private:
  
public:
  explicit ViscoelasticityMatricesOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& mu = context->input(0);
    const Tensor& eta = context->input(1);
    const Tensor& lambda = context->input(2);
    const Tensor& dt = context->input(3);
    
    
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& eta_shape = eta.shape();
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& dt_shape = dt.shape();
    
    
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 1);
    DCHECK_EQ(lambda_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);

    // extra check
        
    // create output shape
    
    TensorShape s_shape({-1,3,3});
    TensorShape h_shape({-1,3,3});
            
    // create output tensor
    
    Tensor* s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, s_shape, &s));
    Tensor* h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, h_shape, &h));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto lambda_tensor = lambda.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto s_tensor = s->flat<double>().data();
    auto h_tensor = h->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoelasticityMatrices").Device(DEVICE_GPU), ViscoelasticityMatricesOpGPU);

class ViscoelasticityMatricesGradOpGPU : public OpKernel {
private:
  
public:
  explicit ViscoelasticityMatricesGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_s = context->input(0);
    const Tensor& grad_h = context->input(1);
    const Tensor& s = context->input(2);
    const Tensor& h = context->input(3);
    const Tensor& mu = context->input(4);
    const Tensor& eta = context->input(5);
    const Tensor& lambda = context->input(6);
    const Tensor& dt = context->input(7);
    
    
    const TensorShape& grad_s_shape = grad_s.shape();
    const TensorShape& grad_h_shape = grad_h.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& mu_shape = mu.shape();
    const TensorShape& eta_shape = eta.shape();
    const TensorShape& lambda_shape = lambda.shape();
    const TensorShape& dt_shape = dt.shape();
    
    
    DCHECK_EQ(grad_s_shape.dims(), 3);
    DCHECK_EQ(grad_h_shape.dims(), 3);
    DCHECK_EQ(s_shape.dims(), 3);
    DCHECK_EQ(h_shape.dims(), 3);
    DCHECK_EQ(mu_shape.dims(), 1);
    DCHECK_EQ(eta_shape.dims(), 1);
    DCHECK_EQ(lambda_shape.dims(), 1);
    DCHECK_EQ(dt_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_mu_shape(mu_shape);
    TensorShape grad_eta_shape(eta_shape);
    TensorShape grad_lambda_shape(lambda_shape);
    TensorShape grad_dt_shape(dt_shape);
            
    // create output tensor
    
    Tensor* grad_mu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_mu_shape, &grad_mu));
    Tensor* grad_eta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_eta_shape, &grad_eta));
    Tensor* grad_lambda = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_lambda_shape, &grad_lambda));
    Tensor* grad_dt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_dt_shape, &grad_dt));
    
    // get the corresponding Eigen tensors for data access
    
    auto mu_tensor = mu.flat<double>().data();
    auto eta_tensor = eta.flat<double>().data();
    auto lambda_tensor = lambda.flat<double>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto grad_s_tensor = grad_s.flat<double>().data();
    auto grad_h_tensor = grad_h.flat<double>().data();
    auto s_tensor = s.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_mu_tensor = grad_mu->flat<double>().data();
    auto grad_eta_tensor = grad_eta->flat<double>().data();
    auto grad_lambda_tensor = grad_lambda->flat<double>().data();
    auto grad_dt_tensor = grad_dt->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("ViscoelasticityMatricesGrad").Device(DEVICE_GPU), ViscoelasticityMatricesGradOpGPU);

#endif