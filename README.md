# NNFEM

## Install `NNFEM`

If you intend to develop the package (add new features, modify current functions, etc.), we suggest developing the package (in the current directory)
```
julia> ]
pkg> dev .
```

If you only want to use the package and do not want to install the dependencies manually, do
```
julia> ]
pkg> activate .
(NNFEM) pkg> instantiate
```


# Quick Start

There are several test cases in `test/Plate` and `test/Truss/Case1D`

* `Data_*` generates the test data and produces `Data/1.dat` and `Data/domain.jld2`

* `NNLearn.jl` learns an ANN with end-to-end strain-to-stress data generated previously. It produces `learned_nn.mat`

* `Train_NN*` learns an ANN from displacement data only.

* `Test_NN*` substitutes the constitutive law with the learned NN from last step and computes the displacement. 


# Applications

The applications are located in the `test` folder. 

