<p align="center">
  <img src="docs/src/assets/nnfem.gif" alt="NNFEM"/>
</p>

NNFEM is a
* lightweight educational 2D finite element library with **truss and 2D quadrilateral elements**. Different constitutive relations are supported, including **plane stress/strain**, **hyperelasticity**, **elasto-plasticity**, etc. It supports **unstructured grid**. 
* neural network-enabled finite element library, which supports learning a neural network-based constitutive relations with both direct data (i.e, strain-stress pairs) and indirect data (i.e. full displacement field) via **automatic differentiation**, and solving finite element problems with **network-based constitutive relations**. In principle, it allows you to insert and learn a neural network anywhere in your finite element codes. 


⚠️ NNFEM.jl is now superseded by [AdFem.jl](https://github.com/kailaix/AdFem.jl), a computational-graph-based finite element library for inverse modeling. NNFEM.jl will be no longer actively developed. 


| Documentation                                                |
| ------------------------------------------------------------ |
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://kailaix.github.io/NNFEM.jl/dev) |


## Install `NNFEM`

Install via Julia registery
```julia
using Pkg; Pkg.add("NNFEM")
```

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



## Code structure

### Basic finite element library

* elements are in /src/elements, including finite/small strain 2D quad and 1D truss elements.

* constitutive relations are in /src/materials, including plane stress/strain, hyperelasticity, elasto-plasticity, etc.

* solvers are in /src/solvers/Solver.jl, including generalized-alpha solver, etc.

* finite element domain, and core functions are in /src/fem.

### Neural network based constitutive relations 

* neural network based constitutive relations are in /src/materials/NeuralNetwork1D.jl and src/materials/NeuralNetwork2D.jl.

* neural network based finite element solvers are in /src/solvers/NNSolver.jl, which compute the loss for indirect data training.

* different customized neural networks are in /deps/CustomOp, which enables designing constraint-embedded neural networks.


### Applications


There are several applications in `research/ConstitutiveRelations/Plate` and `research/ConstitutiveRelations/Truss/Case1D`

* `Data_*` runs the finite element solver to generate the test data and produces `Data/1.dat` and `Data/domain.jld2` 

* `NNLearn.jl` learns an ANN with strain-to-stress data generated previously (extracted from each Gaussian quadrature points of the train sets). It produces `learned_nn.mat`. This is refered as direct training.

* `Train_NN*` learns an ANN from displacement data and associated loading condition. This is refered as indirect training.

* `Test_NN*` substitutes the constitutive law with the learned NN and test the hybrid model (NN + FEM) on the test sets.

* `NN_Test_All*` substitutes the constitutive law with the learned NN and test the hybrid model (NN + FEM) on the all test cases, and visualize the time-histories of the displacement and von-Mises stress fields.





## Troubleshooting 

### Python dependencies 
NNFEM is based on ADCME, you need to first install ADCME.jl, which will install a private Python environment for you. Take it easy, it will NOT mess your local environment!

A bit more about what is under the hood: PyCall relies on the python version installed in `$HOME/.julia/conda/3/bin/python`, you can check the path with

```
julia> using PyCall
julia> PyCall.python
```

If you want to use Python packages via PyCall, install python packages, e.g., `tikzplotlib`, via
```
$HOME/.julia/conda/3/bin/python -m pip install tikzplotlib
```

### Build customized operators

NNFEM includes some custom operators. Those operators are implemented in C++ and will be compiled automatically when you invoke `Pkg.build("NNFEM")`. However, in the case you encounter any compilation issue, you can go into the `deps` directory, and run `build.jl`
```
cd deps
julia build.jl
```

### Submit an issue
You are welcome to submit an issue for any questions related to NNFEM. 


## Research 

1. Huang, Daniel Z., Kailai Xu, Charbel Farhat, and Eric Darve. "[Learning Constitutive Relations from Indirect Observations Using Deep Neural Networks](https://arxiv.org/pdf/1905.12530.pdf)"

2. Kailai Xu, Huang, Daniel Z., and Eric Darve. "[Learning Constitutive Relations using  Symmetric Positive Definite Neural Networks](https://arxiv.org/pdf/2004.00265.pdf)"


