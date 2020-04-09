# Simulation 

We use the generalized $\alpha$ scheme to solve the dynamics equation numerically. 

$$m\ddot{ \mathbf{u}} + \gamma\dot{\mathbf{u}} + k\mathbf{u} = \mathbf{f}$$

The discretized form is 

$$M\mathbf{a} + C\mathbf v + K \mathbf d = \mathbf F$$

For a detailed description of the generalized $\alpha$ scheme, see [this post](https://kailaix.github.io/ADCME.jl/dev/alphascheme/). 


NNFEM.jl supports two types of boundary conditions, the Dirichlet boundary condition and the Neumann boundary condition. Both boundary conditions can be time independent or dependent. 