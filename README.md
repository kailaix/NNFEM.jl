# NNFEM

## Install `NNFEM`

If you intend to develop the package (add new features, modify current functions, etc.), we suggest developing the package (in the current directory (NNFEM.jl))
```
julia> ]
pkg> dev .
```

When necessary, you can delete the package (in the current directory (NNFEM.jl))
```
julia> ]
pkg> rm NNFEM
```


If you only want to use the package and do not want to install the dependencies manually, do
```
julia> ]
pkg> activate .
(NNFEM) pkg> instantiate
```


## Build customized operators
There are several customized operators in deps, you need to compile them first
```
cd deps
julia build.jl
```


## Applications


There are several applications in `test/Plate` and `test/Truss/Case1D`

* `Data_*` generates the test data and produces `Data/1.dat` and `Data/domain.jld2`

* `NNLearn.jl` learns an ANN with strain-to-stress data generated previously (extracted from each Gaussian quadrature points of the train sets). It produces `learned_nn.mat`. This is refered as direct training.

* `Train_NN*` learns an ANN from displacement data and associated loading condition. This is refered as indirect training.

* `Test_NN*` substitutes the constitutive law with the learned NN and test the hybrid model (NN + FEM) on the test sets.

* `NN_Test_All*` substitutes the constitutive law with the learned NN and test the hybrid model (NN + FEM) on the all test cases, and visualize the time-histories of the displacement and von-Mises stress fields.




## Installation issues


NNFEM is based on ADCME, you need to first install ADCME


PyCall relies on the python version installed in 'XXX/.julia/conda/3/bin/python', you can check the path with

```
julia> using PyCall
PyCall.python
```
To install python packages, i.e. 'tikzplotlib'  
```
XXX/.julia/conda/3/bin/python -m pip install tikzplotlib
```


