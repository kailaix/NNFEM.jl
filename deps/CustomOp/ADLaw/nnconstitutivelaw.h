#ifndef __NNCONSTITUTIVELAW__
#define __NNCONSTITUTIVELAW__
#include "adept.h"
#include "adept_arrays.h"
#include <iostream>
#include <vector>
#include<functional>

using namespace adept;
using namespace std;

typedef Array<2, double, true> Array2D;
typedef Array<1, double, true> Array1D;

extern "C" void constitutive_law(
    double *osigma,
    const double*input,
    const double*theta,
    const double*g, 
    double*dinput,
    double*dtheta,
    int n,
    int grad_input,
    int grad_theta
);

extern "C" void constitutive_law_generic(
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
);

#endif