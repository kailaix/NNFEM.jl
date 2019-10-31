#include "nnconstitutivelaw.h"


void setv(Array1D& v,  const double *val){
    int n = v.size(0);
    for(int i=0;i<n;i++) v(i).set_value(val[i]);
}

void setv(Array2D& v, const double *val){
    int k = 0;
    int m = v.size(0), n = v.size(1);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++)
            v(i,j).set_value(val[k++]);
    }
}

void getg(Array1D& v, double *val){
    int n = v.size(0);
    auto gv = v.get_gradient();
    for(int i=0;i<n;i++) val[i] = value(gv(i));
}

void getg(Array2D& v, double *val){
    int k = 0;
    int m = v.size(0), n = v.size(1);
    auto gv = v.get_gradient();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++)
            val[k++] = value(gv(i, j));
    }
}

void cholorthop(Array2D& out, Array2D&x, Array2D& y){
  /*     y0 y1 0
    L =     y2 0             out = L^T L
              y3
  */
  int n = out.size(0);
  for(int i=0;i<n;i++){
    out(i,0) = y(i,0)*y(i,0)*(x(i,0)-x(i,3)) + y(i,0)*y(i,1)*(x(i,1)-x(i,4)) + x(i,6);
    out(i,1) = y(i,0)*y(i,1)*(x(i,0)-x(i,3)) +
                 ( y(i,2)*y(i,2) + y(i,1)*y(i,1) ) * (x(i,1)-x(i,4)) + x(i,7);
    out(i,2) = y(i,3)*y(i,3)*(x(i,2)-x(i,5)) + x(i,8);
  }
}

void constitutive_law(
    double *osigma,
    const double*input,
    const double*theta,
    const double*g, 
    double*dinput,
    double*dtheta,
    int n,
    int grad_input,
    int grad_theta
){
    Stack stack;
    Array2D X(n,9), W1(9,20), W2(20,20), W3(20,4);
    Array1D b1(20), b2(20), b3(4);
    int k = 0;
    setv(X, input);
    setv(W1, theta);
    setv(b1, theta+180);
    setv(W2, theta+180+20);
    setv(b2, theta+180+20+400);
    setv(W3, theta+180+20+400+20);
    setv(b3, theta+180+20+400+20+80);
    stack.new_recording();

    auto x = X**W1;
    Array2D y1(x.size(0),x.size(1));
    for(int i=0;i<x.size(0);i++) 
        for(int j=0;j<x.size(1);j++)
            y1(i,j) = tanh(x(i,j)+b1(j));
        
    auto w = y1**W2;
    Array2D y2(w.size(0),w.size(1));
    for(int i=0;i<w.size(0);i++) 
        for(int j=0;j<w.size(1);j++)
            y2(i,j) = tanh(w(i,j)+b2(j));

    auto z = y2**W3;
    Array2D y3(z.size(0),z.size(1));
    for(int i=0;i<z.size(0);i++) 
        for(int j=0;j<z.size(1);j++)
            y3(i,j) = z(i,j)+b3(j);

    Array2D out(n, 3);
    cholorthop(out, X, y3);

    for(int i=0;i<n;i++){
        osigma[3*i] = out(i,0).value();
        osigma[3*i+1] = out(i,1).value();
        osigma[3*i+2] = out(i,2).value();
    }

    // auto out = sum(y3, 0);
    // out[0].set_gradient(1.0);
    // stack.compute_adjoint();

    if(grad_theta){
        Array2D G(n, 3);
        setv(G, g);
        adouble L = sum(G * out);
        L.set_gradient(1.0);
        stack.compute_adjoint();
        getg(W1, dtheta);
        getg(b1, dtheta+180);
        getg(W2, dtheta+180+20);
        getg(b2, dtheta+180+20+400);
        getg(W3, dtheta+180+20+400+20);
        getg(b3, dtheta+180+20+400+20+80);
    }

    if(grad_input){
        auto out_ = sum(out, 0);
        // cout << out_ << endl;
        for(int i=0;i<3;i++){
            stack.clear_gradients();
            out_[i].set_gradient(1.0);
            stack.compute_adjoint();
            getg(X, dinput + i*n*9);
        }
    }
}



void constitutive_law_generic(
    double *osigma,
    const double*input,
    const double*theta,
    const double*g, 
    double*dinput,
    double*dtheta,
    const int *config,
    int n_layer,
    int n,
    int grad_input,
    int grad_theta
){
    Stack stack;
    std::vector<Array2D> weight(n_layer-1);
    std::vector<Array1D> bias(n_layer-1);
    std::vector<Array2D> x(n_layer);
    std::vector<Array2D> y(n_layer-1);

    // for(int i=0;i<n_layer;i++){
    //     printf("%d : %d\n", i, config[i]);
    // }

    Array2D X(n,9);
    setv(X, input);
    for(int i=0;i<n_layer-1;i++){
        weight[i] = Array2D(config[i], config[i+1]);
        bias[i] = Array1D(config[i+1]);
        setv(weight[i], theta); theta += config[i]*config[i+1];
        setv(bias[i], theta); theta += config[i+1];
    }    
    if (grad_input!=0 || grad_theta!=0)
        stack.new_recording();

    x[0] = X;
    for(int l=0;l<n_layer-1;l++){
        x[l+1] = x[l]**weight[l];
        y[l] = Array2D(x[l+1].size(0),x[l+1].size(1));
        for(int i=0;i<x[l+1].size(0);i++) 
            for(int j=0;j<x[l+1].size(1);j++){
                if(l==n_layer-2)
                    y[l](i,j) = x[l+1](i,j)+bias[l](j);
                else
                    y[l](i,j) = tanh(x[l+1](i,j)+bias[l](j));
            }           
    }

    Array2D out(n, 3);
    cholorthop(out, X, y[n_layer-2]);

    for(int i=0;i<n;i++){
        osigma[3*i] = out(i,0).value();
        osigma[3*i+1] = out(i,1).value();
        osigma[3*i+2] = out(i,2).value();
    }

    // auto out = sum(y3, 0);
    // out[0].set_gradient(1.0);
    // stack.compute_adjoint();

    if(grad_theta){
        Array2D G(n, 3);
        setv(G, g);
        adouble L = sum(G * out);
        stack.clear_gradients();
        L.set_gradient(1.0);
        stack.compute_adjoint();
        for(int i=0;i<n_layer-1;i++){
            getg(weight[i], dtheta); dtheta += config[i]*config[i+1];
            getg(bias[i], dtheta); dtheta += config[i+1];
        }    
    }

    if(grad_input){
        auto out_ = sum(out, 0);
        // cout << out_ << endl;
        for(int i=0;i<3;i++){
            stack.clear_gradients();
            out_[i].set_gradient(1.0);
            stack.compute_adjoint();
            getg(X, dinput + i*n*9);
        }
    }
}
