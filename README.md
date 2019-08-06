## Install NNFEM.jl Development Package
```
$ julia
julia> ]
pkg> dev .
```
Now you have access to NNFEM.jl through `using NNFEM`

## Development
```
julia> using Revise
julia> using NNFEM
```
Run tests in the Julia console, modify source codes without exiting Julia every time.

## Test and Commit
```
julia> ]
pkg> test NNFEM
```
For every function, a test snippet should go into `test` folder. Be sure to include the new files in `runtests.jl`.


# Quick Start

There are several test cases in `test/Plate` and `test/Truss/Case1D`

* `Data_*` generates the test data and produces `Data/1.dat` and `Data/domain.jld2`

* `NNLearn.jl` learns an ANN with end-to-end strain-to-stress data generated previously. It produces `learned_nn.mat`

* `Train_NN*` learns an ANN from displacement data only.

* `Test_NN*` substitutes the constitutive law with the learned NN from last step and computes the displacement. 