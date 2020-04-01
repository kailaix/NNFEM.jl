# NNFEM

* This is a lightweight educational 2D finite element library with truss and 2D quad elements. Different constitutive relations are supported, including plane stress/strain, hyperelasticity, elasto-plasticity, etc. 

* This is also a research code to explore nerual network-enabled finite element library, which supports learning a nerual network-based constitutive relations with both direct data (i.e, strain-stress pairs) and indirect data (i.e. full displacement field) via automatic differentiation, and solving finite element problems with network-based constitutive relations.




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




## Code structure

### Basic finite element library

* elements are in /src/elements, including finite/small strain 2D quad and 1D truss elements.

* constitutive relations are in /src/materials, including plane stress/strain, hyperelasticity, elasto-plasticity, etc.

* solvers are in /src/solvers/Solver.jl, including generalized-alpha solver, etc.

* finite element domain, and core functions are in /src/fem.

### Nerual network enablers 

* nerual network based constitutive relations are in /src/materials/NeuralNetwork1D.jl and src/materials/NeuralNetwork2D.jl.

* nerual network based finite element solvers are in /src/solvers/NNSolver.jl, which compute the loss for indirect data training.

* different customized neural networks are in /deps/CustomOp, which enables designing constraint-embedded neural networks.


### Applications


There are several applications in `test/Plate` and `test/Truss/Case1D`

* `Data_*` runs the finite element solver to generate the test data and produces `Data/1.dat` and `Data/domain.jld2` 

* `NNLearn.jl` learns an ANN with strain-to-stress data generated previously (extracted from each Gaussian quadrature points of the train sets). It produces `learned_nn.mat`. This is refered as direct training.

* `Train_NN*` learns an ANN from displacement data and associated loading condition. This is refered as indirect training.

* `Test_NN*` substitutes the constitutive law with the learned NN and test the hybrid model (NN + FEM) on the test sets.

* `NN_Test_All*` substitutes the constitutive law with the learned NN and test the hybrid model (NN + FEM) on the all test cases, and visualize the time-histories of the displacement and von-Mises stress fields.





## Installation issues


NNFEM is based on ADCME, you need to first install ADCME


PyCall relies on the python version installed in 'XXX/.julia/conda/3/bin/python', you can check the path with

```
julia> using PyCall
julia> PyCall.python
```
To install python packages, i.e. 'tikzplotlib'  
```
XXX/.julia/conda/3/bin/python -m pip install tikzplotlib
```


