#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include<string> 
using std::string;
using tensorflow::int64;
using tensorflow::Status;
using tensorflow::OpKernelContext;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernel;

// If you want to use the PyTorch feature, uncomment the following line
// #include "la.h" 
#include "FEM.h"


REGISTER_OP("FemOp")

.Input("theta : double")
  .Input("d : double")
  .Input("v : double")
  .Input("a : double")
  .Input("fext : double")
  .Input("eps : double")
  .Input("sigma : double")
  .Input("m : double")
  .Input("neqs : int64")
  .Input("neqns_per_elem : int64")
  .Input("nelems : int64")
  .Input("ngps_per_elem : int64")
  .Input("ngp : int64")
  .Input("dt : double")
  .Input("el_eqns_row : int64")
  .Input("dhdx : double")
  .Input("weights : double")
  .Input("max_iter : int64")
  .Input("tol : double")
  .Output("oa : double")
  .Output("ov : double")
  .Output("od : double")
  .Output("osigma : double")
  .Output("oeps : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        tensorflow::shape_inference::ShapeHandle theta_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &theta_shape));
        tensorflow::shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &d_shape));
        tensorflow::shape_inference::ShapeHandle v_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &v_shape));
        tensorflow::shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &a_shape));
        tensorflow::shape_inference::ShapeHandle fext_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &fext_shape));
        tensorflow::shape_inference::ShapeHandle eps_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &eps_shape));
        tensorflow::shape_inference::ShapeHandle sigma_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &sigma_shape));
        tensorflow::shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 2, &m_shape));
        tensorflow::shape_inference::ShapeHandle neqs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &neqs_shape));
        tensorflow::shape_inference::ShapeHandle neqns_per_elem_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &neqns_per_elem_shape));
        tensorflow::shape_inference::ShapeHandle nelems_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 0, &nelems_shape));
        tensorflow::shape_inference::ShapeHandle ngps_per_elem_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(11), 0, &ngps_per_elem_shape));
        tensorflow::shape_inference::ShapeHandle ngp_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(12), 0, &ngp_shape));
        tensorflow::shape_inference::ShapeHandle dt_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(13), 0, &dt_shape));
        tensorflow::shape_inference::ShapeHandle el_eqns_row_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(14), 1, &el_eqns_row_shape));
        tensorflow::shape_inference::ShapeHandle dhdx_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(15), 1, &dhdx_shape));
        tensorflow::shape_inference::ShapeHandle weights_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(16), 1, &weights_shape));
        tensorflow::shape_inference::ShapeHandle max_iter_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(17), 0, &max_iter_shape));
        tensorflow::shape_inference::ShapeHandle tol_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(18), 0, &tol_shape));

        c->set_output(0, c->Vector(-1));
        c->set_output(1, c->Vector(-1));
        c->set_output(2, c->Vector(-1));
        c->set_output(3, c->Matrix(-1,3));
        c->set_output(4, c->Matrix(-1,3));
    return Status::OK();
  });
class FemOpOp : public OpKernel {
private:
    FEM fem;
public:
  explicit FemOpOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(19, context->num_inputs());
    
    printf("%d\n", __LINE__);
    const tensorflow::Tensor& theta = context->input(0);
    const tensorflow::Tensor& d = context->input(1);
    const tensorflow::Tensor& v = context->input(2);
    const tensorflow::Tensor& a = context->input(3);
    printf("%d\n", __LINE__);
    const tensorflow::Tensor& fext = context->input(4);
    const tensorflow::Tensor& eps = context->input(5);
    const tensorflow::Tensor& sigma = context->input(6);
    const tensorflow::Tensor& m = context->input(7);
    const tensorflow::Tensor& neqs = context->input(8);
    printf("%d\n", __LINE__);
    const tensorflow::Tensor& neqns_per_elem = context->input(9);
    const tensorflow::Tensor& nelems = context->input(10);
    const tensorflow::Tensor& ngps_per_elem = context->input(11);
    const tensorflow::Tensor& ngp = context->input(12);
    const tensorflow::Tensor& dt = context->input(13);
    printf("%d\n", __LINE__);
    const tensorflow::Tensor& el_eqns_row = context->input(14);
    const tensorflow::Tensor& dhdx = context->input(15);
    const tensorflow::Tensor& weights = context->input(16);
    const tensorflow::Tensor& max_iter = context->input(17);
    const tensorflow::Tensor& tol = context->input(18);
    printf("%d\n", __LINE__);
    
    const tensorflow::TensorShape& theta_shape = theta.shape();
    const tensorflow::TensorShape& d_shape = d.shape();
    const tensorflow::TensorShape& v_shape = v.shape();
    const tensorflow::TensorShape& a_shape = a.shape();
    printf("%d\n", __LINE__);
    const tensorflow::TensorShape& fext_shape = fext.shape();
    const tensorflow::TensorShape& eps_shape = eps.shape();
    const tensorflow::TensorShape& sigma_shape = sigma.shape();
    const tensorflow::TensorShape& m_shape = m.shape();
    const tensorflow::TensorShape& neqs_shape = neqs.shape();
    printf("%d\n", __LINE__);
    const tensorflow::TensorShape& neqns_per_elem_shape = neqns_per_elem.shape();
    const tensorflow::TensorShape& nelems_shape = nelems.shape();
    const tensorflow::TensorShape& ngps_per_elem_shape = ngps_per_elem.shape();
    const tensorflow::TensorShape& ngp_shape = ngp.shape();
    const tensorflow::TensorShape& dt_shape = dt.shape();
    printf("%d\n", __LINE__);
    const tensorflow::TensorShape& el_eqns_row_shape = el_eqns_row.shape();
    const tensorflow::TensorShape& dhdx_shape = dhdx.shape();
    const tensorflow::TensorShape& weights_shape = weights.shape();
    const tensorflow::TensorShape& max_iter_shape = max_iter.shape();
    const tensorflow::TensorShape& tol_shape = tol.shape();
    printf("%d\n", __LINE__);
    
    DCHECK_EQ(theta_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(fext_shape.dims(), 1);
    printf("%d\n", __LINE__);
    DCHECK_EQ(eps_shape.dims(), 2);
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 2);
    DCHECK_EQ(neqs_shape.dims(), 0);
    DCHECK_EQ(neqns_per_elem_shape.dims(), 0);
    DCHECK_EQ(nelems_shape.dims(), 0);
    printf("%d\n", __LINE__);
    DCHECK_EQ(ngps_per_elem_shape.dims(), 0);
    DCHECK_EQ(ngp_shape.dims(), 0);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(el_eqns_row_shape.dims(), 1);
    DCHECK_EQ(dhdx_shape.dims(), 1);
    DCHECK_EQ(weights_shape.dims(), 1);
    DCHECK_EQ(max_iter_shape.dims(), 0);
    DCHECK_EQ(tol_shape.dims(), 0);
    printf("%d\n", __LINE__);

    // extra check
        
    // create output shape
    int n = a_shape.dim_size(0);
    int p = sigma_shape.dim_size(0);
    tensorflow::TensorShape oa_shape({n});
    tensorflow::TensorShape ov_shape({n});
    tensorflow::TensorShape od_shape({n});
    tensorflow::TensorShape osigma_shape({p,3});
    tensorflow::TensorShape oeps_shape({p,3});
    printf("%d\n", __LINE__);
            
    // create output tensor
    
    tensorflow::Tensor* oa = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, oa_shape, &oa));
    tensorflow::Tensor* ov = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, ov_shape, &ov));
    tensorflow::Tensor* od = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, od_shape, &od));
    tensorflow::Tensor* osigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, osigma_shape, &osigma));
    tensorflow::Tensor* oeps = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, oeps_shape, &oeps));
    printf("%d\n", __LINE__);
    // get the corresponding Eigen tensors for data access
    
    auto theta_tensor = theta.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto a_tensor = a.flat<double>().data();
    auto fext_tensor = fext.flat<double>().data();
    auto eps_tensor = eps.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto m_tensor = m.flat<double>().data();
    auto neqs_tensor = neqs.flat<int64>().data();
    auto neqns_per_elem_tensor = neqns_per_elem.flat<int64>().data();
    auto nelems_tensor = nelems.flat<int64>().data();
    auto ngps_per_elem_tensor = ngps_per_elem.flat<int64>().data();
    auto ngp_tensor = ngp.flat<int64>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto el_eqns_row_tensor = el_eqns_row.flat<int64>().data();
    auto dhdx_tensor = dhdx.flat<double>().data();
    auto weights_tensor = weights.flat<double>().data();
    auto max_iter_tensor = max_iter.flat<int64>().data();
    auto tol_tensor = tol.flat<double>().data();
    auto oa_tensor = oa->flat<double>().data();
    auto ov_tensor = ov->flat<double>().data();
    auto od_tensor = od->flat<double>().data();
    auto osigma_tensor = osigma->flat<double>().data();
    auto oeps_tensor = oeps->flat<double>().data();   

    // implement your forward function here 
printf("%d\n", __LINE__);
    // TODO:
    fem.initialization(
        *neqs_tensor, 
        *neqns_per_elem_tensor, 
        *nelems_tensor, 
        *ngps_per_elem_tensor, 
        *ngp_tensor,
        *dt_tensor,
        theta_tensor, 
        el_eqns_row_tensor,
        dhdx_tensor, 
        weights_tensor,
        *max_iter_tensor, 
        *tol_tensor,
         d_tensor, 
        v_tensor, 
        a_tensor, 
        eps_tensor, 
        sigma_tensor,
         fext_tensor, 
        m_tensor,
         od_tensor, 
        ov_tensor, 
        oa_tensor, 
        oeps_tensor, 
        osigma_tensor);
        printf("%d\n", __LINE__);
    fem.forward(oa_tensor, ov_tensor, od_tensor, osigma_tensor, oeps_tensor);
    printf("%d\n", __LINE__);

  }
};
REGISTER_KERNEL_BUILDER(Name("FemOp").Device(tensorflow::DEVICE_CPU), FemOpOp);


REGISTER_OP("FemOpGrad")
  
  .Input("grad_oa : double")
.Input("grad_ov : double")
.Input("grad_od : double")
.Input("grad_osigma : double")
.Input("grad_oeps : double")
  .Input("oa : double")
  .Input("ov : double")
  .Input("od : double")
  .Input("osigma : double")
  .Input("oeps : double")
  .Input("theta : double")
  .Input("d : double")
  .Input("v : double")
  .Input("a : double")
  .Input("fext : double")
  .Input("eps : double")
  .Input("sigma : double")
  .Input("m : double")
  .Input("neqs : int64")
  .Input("neqns_per_elem : int64")
  .Input("nelems : int64")
  .Input("ngps_per_elem : int64")
  .Input("ngp : int64")
  .Input("dt : double")
  .Input("el_eqns_row : int64")
  .Input("dhdx : double")
  .Input("weights : double")
  .Input("max_iter : int64")
  .Input("tol : double")
  .Output("grad_theta : double")
  .Output("grad_d : double")
  .Output("grad_v : double")
  .Output("grad_a : double")
  .Output("grad_fext : double")
  .Output("grad_eps : double")
  .Output("grad_sigma : double")
  .Output("grad_m : double")
  .Output("grad_neqs : int64")
  .Output("grad_neqns_per_elem : int64")
  .Output("grad_nelems : int64")
  .Output("grad_ngps_per_elem : int64")
  .Output("grad_ngp : int64")
  .Output("grad_dt : double")
  .Output("grad_el_eqns_row : int64")
  .Output("grad_dhdx : double")
  .Output("grad_weights : double")
  .Output("grad_max_iter : int64")
  .Output("grad_tol : double");
class FemOpGradOp : public OpKernel {
private:
  FEM fem;
public:
  explicit FemOpGradOp(tensorflow::OpKernelConstruction* context) : OpKernel(context) {
      fem = FEM(3);
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const tensorflow::Tensor& grad_oa = context->input(0);
    const tensorflow::Tensor& grad_ov = context->input(1);
    const tensorflow::Tensor& grad_od = context->input(2);
    const tensorflow::Tensor& grad_osigma = context->input(3);
    const tensorflow::Tensor& grad_oeps = context->input(4);
    const tensorflow::Tensor& oa = context->input(5);
    const tensorflow::Tensor& ov = context->input(6);
    const tensorflow::Tensor& od = context->input(7);
    const tensorflow::Tensor& osigma = context->input(8);
    const tensorflow::Tensor& oeps = context->input(9);
    const tensorflow::Tensor& theta = context->input(10);
    const tensorflow::Tensor& d = context->input(11);
    const tensorflow::Tensor& v = context->input(12);
    const tensorflow::Tensor& a = context->input(13);
    const tensorflow::Tensor& fext = context->input(14);
    const tensorflow::Tensor& eps = context->input(15);
    const tensorflow::Tensor& sigma = context->input(16);
    const tensorflow::Tensor& m = context->input(17);
    const tensorflow::Tensor& neqs = context->input(18);
    const tensorflow::Tensor& neqns_per_elem = context->input(19);
    const tensorflow::Tensor& nelems = context->input(20);
    const tensorflow::Tensor& ngps_per_elem = context->input(21);
    const tensorflow::Tensor& ngp = context->input(22);
    const tensorflow::Tensor& dt = context->input(23);
    const tensorflow::Tensor& el_eqns_row = context->input(24);
    const tensorflow::Tensor& dhdx = context->input(25);
    const tensorflow::Tensor& weights = context->input(26);
    const tensorflow::Tensor& max_iter = context->input(27);
    const tensorflow::Tensor& tol = context->input(28);
    
    
    const tensorflow::TensorShape& grad_oa_shape = grad_oa.shape();
    const tensorflow::TensorShape& grad_ov_shape = grad_ov.shape();
    const tensorflow::TensorShape& grad_od_shape = grad_od.shape();
    const tensorflow::TensorShape& grad_osigma_shape = grad_osigma.shape();
    const tensorflow::TensorShape& grad_oeps_shape = grad_oeps.shape();
    const tensorflow::TensorShape& oa_shape = oa.shape();
    const tensorflow::TensorShape& ov_shape = ov.shape();
    const tensorflow::TensorShape& od_shape = od.shape();
    const tensorflow::TensorShape& osigma_shape = osigma.shape();
    const tensorflow::TensorShape& oeps_shape = oeps.shape();
    const tensorflow::TensorShape& theta_shape = theta.shape();
    const tensorflow::TensorShape& d_shape = d.shape();
    const tensorflow::TensorShape& v_shape = v.shape();
    const tensorflow::TensorShape& a_shape = a.shape();
    const tensorflow::TensorShape& fext_shape = fext.shape();
    const tensorflow::TensorShape& eps_shape = eps.shape();
    const tensorflow::TensorShape& sigma_shape = sigma.shape();
    const tensorflow::TensorShape& m_shape = m.shape();
    const tensorflow::TensorShape& neqs_shape = neqs.shape();
    const tensorflow::TensorShape& neqns_per_elem_shape = neqns_per_elem.shape();
    const tensorflow::TensorShape& nelems_shape = nelems.shape();
    const tensorflow::TensorShape& ngps_per_elem_shape = ngps_per_elem.shape();
    const tensorflow::TensorShape& ngp_shape = ngp.shape();
    const tensorflow::TensorShape& dt_shape = dt.shape();
    const tensorflow::TensorShape& el_eqns_row_shape = el_eqns_row.shape();
    const tensorflow::TensorShape& dhdx_shape = dhdx.shape();
    const tensorflow::TensorShape& weights_shape = weights.shape();
    const tensorflow::TensorShape& max_iter_shape = max_iter.shape();
    const tensorflow::TensorShape& tol_shape = tol.shape();
    
    
    DCHECK_EQ(grad_oa_shape.dims(), 1);
    DCHECK_EQ(grad_ov_shape.dims(), 1);
    DCHECK_EQ(grad_od_shape.dims(), 1);
    DCHECK_EQ(grad_osigma_shape.dims(), 2);
    DCHECK_EQ(grad_oeps_shape.dims(), 2);
    DCHECK_EQ(oa_shape.dims(), 1);
    DCHECK_EQ(ov_shape.dims(), 1);
    DCHECK_EQ(od_shape.dims(), 1);
    DCHECK_EQ(osigma_shape.dims(), 2);
    DCHECK_EQ(oeps_shape.dims(), 2);
    DCHECK_EQ(theta_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);
    DCHECK_EQ(v_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(fext_shape.dims(), 1);
    DCHECK_EQ(eps_shape.dims(), 2);
    DCHECK_EQ(sigma_shape.dims(), 2);
    DCHECK_EQ(m_shape.dims(), 2);
    DCHECK_EQ(neqs_shape.dims(), 0);
    DCHECK_EQ(neqns_per_elem_shape.dims(), 0);
    DCHECK_EQ(nelems_shape.dims(), 0);
    DCHECK_EQ(ngps_per_elem_shape.dims(), 0);
    DCHECK_EQ(ngp_shape.dims(), 0);
    DCHECK_EQ(dt_shape.dims(), 0);
    DCHECK_EQ(el_eqns_row_shape.dims(), 1);
    DCHECK_EQ(dhdx_shape.dims(), 1);
    DCHECK_EQ(weights_shape.dims(), 1);
    DCHECK_EQ(max_iter_shape.dims(), 0);
    DCHECK_EQ(tol_shape.dims(), 0);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    tensorflow::TensorShape grad_theta_shape(theta_shape);
    tensorflow::TensorShape grad_d_shape(d_shape);
    tensorflow::TensorShape grad_v_shape(v_shape);
    tensorflow::TensorShape grad_a_shape(a_shape);
    tensorflow::TensorShape grad_fext_shape(fext_shape);
    tensorflow::TensorShape grad_eps_shape(eps_shape);
    tensorflow::TensorShape grad_sigma_shape(sigma_shape);
    tensorflow::TensorShape grad_m_shape(m_shape);
    tensorflow::TensorShape grad_neqs_shape(neqs_shape);
    tensorflow::TensorShape grad_neqns_per_elem_shape(neqns_per_elem_shape);
    tensorflow::TensorShape grad_nelems_shape(nelems_shape);
    tensorflow::TensorShape grad_ngps_per_elem_shape(ngps_per_elem_shape);
    tensorflow::TensorShape grad_ngp_shape(ngp_shape);
    tensorflow::TensorShape grad_dt_shape(dt_shape);
    tensorflow::TensorShape grad_el_eqns_row_shape(el_eqns_row_shape);
    tensorflow::TensorShape grad_dhdx_shape(dhdx_shape);
    tensorflow::TensorShape grad_weights_shape(weights_shape);
    tensorflow::TensorShape grad_max_iter_shape(max_iter_shape);
    tensorflow::TensorShape grad_tol_shape(tol_shape);
            
    // create output tensor
    
    tensorflow::Tensor* grad_theta = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_theta_shape, &grad_theta));
    tensorflow::Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_d_shape, &grad_d));
    tensorflow::Tensor* grad_v = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_v_shape, &grad_v));
    tensorflow::Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_a_shape, &grad_a));
    tensorflow::Tensor* grad_fext = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_fext_shape, &grad_fext));
    tensorflow::Tensor* grad_eps = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_eps_shape, &grad_eps));
    tensorflow::Tensor* grad_sigma = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_sigma_shape, &grad_sigma));
    tensorflow::Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_m_shape, &grad_m));
    tensorflow::Tensor* grad_neqs = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_neqs_shape, &grad_neqs));
    tensorflow::Tensor* grad_neqns_per_elem = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_neqns_per_elem_shape, &grad_neqns_per_elem));
    tensorflow::Tensor* grad_nelems = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(10, grad_nelems_shape, &grad_nelems));
    tensorflow::Tensor* grad_ngps_per_elem = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(11, grad_ngps_per_elem_shape, &grad_ngps_per_elem));
    tensorflow::Tensor* grad_ngp = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(12, grad_ngp_shape, &grad_ngp));
    tensorflow::Tensor* grad_dt = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(13, grad_dt_shape, &grad_dt));
    tensorflow::Tensor* grad_el_eqns_row = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(14, grad_el_eqns_row_shape, &grad_el_eqns_row));
    tensorflow::Tensor* grad_dhdx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(15, grad_dhdx_shape, &grad_dhdx));
    tensorflow::Tensor* grad_weights = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(16, grad_weights_shape, &grad_weights));
    tensorflow::Tensor* grad_max_iter = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(17, grad_max_iter_shape, &grad_max_iter));
    tensorflow::Tensor* grad_tol = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(18, grad_tol_shape, &grad_tol));
    
    // get the corresponding Eigen tensors for data access
    
    auto theta_tensor = theta.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto v_tensor = v.flat<double>().data();
    auto a_tensor = a.flat<double>().data();
    auto fext_tensor = fext.flat<double>().data();
    auto eps_tensor = eps.flat<double>().data();
    auto sigma_tensor = sigma.flat<double>().data();
    auto m_tensor = m.flat<double>().data();
    auto neqs_tensor = neqs.flat<int64>().data();
    auto neqns_per_elem_tensor = neqns_per_elem.flat<int64>().data();
    auto nelems_tensor = nelems.flat<int64>().data();
    auto ngps_per_elem_tensor = ngps_per_elem.flat<int64>().data();
    auto ngp_tensor = ngp.flat<int64>().data();
    auto dt_tensor = dt.flat<double>().data();
    auto el_eqns_row_tensor = el_eqns_row.flat<int64>().data();
    auto dhdx_tensor = dhdx.flat<double>().data();
    auto weights_tensor = weights.flat<double>().data();
    auto max_iter_tensor = max_iter.flat<int64>().data();
    auto tol_tensor = tol.flat<double>().data();
    auto grad_oa_tensor = grad_oa.flat<double>().data();
    auto grad_ov_tensor = grad_ov.flat<double>().data();
    auto grad_od_tensor = grad_od.flat<double>().data();
    auto grad_osigma_tensor = grad_osigma.flat<double>().data();
    auto grad_oeps_tensor = grad_oeps.flat<double>().data();
    auto oa_tensor = oa.flat<double>().data();
    auto ov_tensor = ov.flat<double>().data();
    auto od_tensor = od.flat<double>().data();
    auto osigma_tensor = osigma.flat<double>().data();
    auto oeps_tensor = oeps.flat<double>().data();
    auto grad_theta_tensor = grad_theta->flat<double>().data();
    auto grad_d_tensor = grad_d->flat<double>().data();
    auto grad_v_tensor = grad_v->flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_fext_tensor = grad_fext->flat<double>().data();
    auto grad_eps_tensor = grad_eps->flat<double>().data();
    auto grad_sigma_tensor = grad_sigma->flat<double>().data();
    auto grad_m_tensor = grad_m->flat<double>().data();
    auto grad_neqs_tensor = grad_neqs->flat<int64>().data();
    auto grad_neqns_per_elem_tensor = grad_neqns_per_elem->flat<int64>().data();
    auto grad_nelems_tensor = grad_nelems->flat<int64>().data();
    auto grad_ngps_per_elem_tensor = grad_ngps_per_elem->flat<int64>().data();
    auto grad_ngp_tensor = grad_ngp->flat<int64>().data();
    auto grad_dt_tensor = grad_dt->flat<double>().data();
    auto grad_el_eqns_row_tensor = grad_el_eqns_row->flat<int64>().data();
    auto grad_dhdx_tensor = grad_dhdx->flat<double>().data();
    auto grad_weights_tensor = grad_weights->flat<double>().data();
    auto grad_max_iter_tensor = grad_max_iter->flat<int64>().data();
    auto grad_tol_tensor = grad_tol->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    fem.initialization(
        *neqs_tensor, 
        *neqns_per_elem_tensor, 
        *nelems_tensor, 
        *ngps_per_elem_tensor, 
        *ngp_tensor,
        *dt_tensor,
        theta_tensor, 
        el_eqns_row_tensor,
        dhdx_tensor, 
        weights_tensor,
        *max_iter_tensor, 
        *tol_tensor,
         d_tensor, 
        v_tensor, 
        a_tensor, 
        eps_tensor, 
        sigma_tensor,
         fext_tensor, 
        m_tensor,
         od_tensor, 
        ov_tensor, 
        oa_tensor, 
        oeps_tensor, 
        osigma_tensor);
    fem.backward(
      grad_oa_tensor, 
      grad_a_tensor, grad_v_tensor, grad_d_tensor, grad_sigma_tensor, grad_eps_tensor, grad_theta_tensor);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("FemOpGrad").Device(tensorflow::DEVICE_CPU), FemOpGradOp);

