# Solvers

There are two types of solvers: **AD-free** solvers, which does not support automatic differentiation but in general more efficient due to less bookkeeping; and **AD-capable** solvers, which has automatic differentiation so you can use them to solve inverse problems.

## AD-free Solvers

AD-free solvers are located in `src/fem/solvers/Solver.jl` and `src/fem/solver/SolverV2.jl`. These solvers are implemented in pure Julia. 

* [`LinearStaticSolver`](@ref)

Consider a static linear elasticity problem 

$$\begin{aligned}
\text{div}\sigma &= f & (x,y) \in \Omega\\ 
\sigma &= H\epsilon \\ 
u(x,y) &= u_0(x,y) & (x,y) \in \partial \Omega
\end{aligned}$$

on a unit square domain $\Omega$. We can compare the result with PoreFlow.jl. 

```@eval 
using Markdown
Markdown.parse("```julia\\n"*String(read("../../test/solvers/linearstatic.jl"))*"```")
```

Here shows the result for PoreFlow (left: PoreFlow, right: Exact)

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/PoreFlow_static_linear.png?raw=true)

Here shows the result for NNFEM (left: NNFEM, right: Exact)

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/NNFEM_static_linear.png?raw=true)

* [`ExplicitSolverStep`](@ref)

* [`ImplicitStaticSolver`](@ref)

* [`GeneralizedAlphaSolverStep`](@ref)

## AD-capable Solvers

AD-capable solver are located in `src/adutils/solvers.jl`. These solvers are implemented with highly optimized C++ kernels. The usage of these solvers are different from AD-free solvers, where all data structures are wrapped in `domain` and `globaldata`. Here users need to provide variables such as displacement, velocity, acceleration, stress, strain, etc., explicitly to the solvers. To this end, the following utiltity functions are provided: 



* [`compute_boundary_info`](@ref)

* [`compute_external_force`](@ref)

* [`ExplicitSolverTime`](@ref)

* [`GeneralizedAlphaSolverTime`](@ref)

A list of available:

* [`LinearStaticSolver`](@ref)


We consider the same linear elasticity problem problem used in AD-free solvers.
```@eval 
using Markdown
Markdown.parse("```julia\\n"*String(read("../../test/solvers/linearstatic_ad.jl"))*"```")
```


![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/AD_static_linear.png?raw=true)


* [`ExplicitSolver`](@ref)

* [`GeneralizedAlphaSolver`](@ref)

* [`ExplicitStaticSolver`](@ref)

* [`ImplicitStaticSolver`](@ref)





