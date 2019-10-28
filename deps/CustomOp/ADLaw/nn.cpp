#include "adept.h"
#include "adept_arrays.h"
#include <iostream>
using namespace adept;
using namespace std;

int main()
{
  // a two layer neural network
  // 3 -- 20 -- 20 -- 4
  Stack stack;
  Array<2, double, true> X(100,3), W1(3,20), W2(20,20), W3(20,4);
  Array<1, double, true> b1(20), b2(20), b3(4);
  double V[400];
  for(int i=0;i<300;i++) X[i] = 0.01*i;
  for(int i=0;i<60;i++) W1[i] = 0.01*i;
  for(int i=0;i<400;i++) W2[i] = 0.01*i;
  for(int i=0;i<80;i++) W3[i] = 0.01*i;
  for(int i=0;i<20;i++) b1[i] = 0.01*i;
  for(int i=0;i<20;i++) b2[i] = 0.01*i;
  for(int i=0;i<4;i++) b3[i] = 0.01*i;

  stack.new_recording();
  auto x = X**W1;
  Array<2, double, true> y1(x.size(0),x.size(1));
  for(int i=0;i<x.size(0);i++) 
    for(int j=0;j<x.size(1);j++)
        y1(i,j) = tanh(x(i,j)+b1(j));
    
  auto w = y1**W2;
  Array<2, double, true> y2(w.size(0),w.size(1));
  for(int i=0;i<w.size(0);i++) 
    for(int j=0;j<w.size(1);j++)
        y2(i,j) = tanh(w(i,j)+b2(j));

  auto z = y2**W3;
  Array<2, double, true> y3(z.size(0),z.size(1));
  for(int i=0;i<z.size(0);i++) 
    for(int j=0;j<z.size(1);j++)
        y3(i,j) = z(i,j)+b3(j);
  
  auto out = sum(y3, 0);
  
  out[0].set_gradient(1.0);
  stack.compute_adjoint();
  auto g1 = X.get_gradient();
  cout << g1 << endl;

  auto g2 = W3.get_gradient();
  cout << g2 << endl;



  out[1].set_gradient(1.0);
  stack.compute_adjoint();
  auto g1_ = X.get_gradient();
  cout << g1_ << endl;

  auto g2_ = W3.get_gradient();
  cout << g2_ << endl;


  y3(0,0).set_gradient(1.0);
  stack.compute_adjoint();
  auto g1__ = X.get_gradient();
  cout << g1__ << endl;
  auto g2__ = W3.get_gradient();
  cout << g2__ << endl;

  // y[0].set_gradient(1.0);
  // stack.compute_adjoint();
  // auto z = M.get_gradient();
  // cout << z << endl;
}
