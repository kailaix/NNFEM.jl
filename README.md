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

```
include("test/Benchmark/Data_NNPlatePull_Plasticity.jl")  # generate data
include("test/Benchmark/Train_NNPlatePull_Plasticity.jl") # training
include("test/Benchmark/Test_NNPlatePull_Plasticity.jl")  # testing
```