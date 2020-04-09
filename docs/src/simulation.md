# Simulation 

We use the Newmark method or the generalized $\alpha$ scheme to solve the dynamics equation numerically. 

$$m\ddot{ \mathbf{u}} + \gamma\dot{\mathbf{u}} + k\mathbf{u} = \mathbf{f}$$

The discretized form is 

$$M\mathbf{a} + C\mathbf v + K \mathbf d = \mathbf F$$

For a detailed description of the generalized $\alpha$ scheme, see [this post](https://kailaix.github.io/ADCME.jl/dev/alphascheme/). 


NNFEM.jl supports two types of boundary conditions, the Dirichlet boundary condition and the Neumann boundary condition. Both boundary conditions can be time independent or dependent. Read [] for how to specify time dependent boundary conditions and [] for how to specify time independent boundary conditions. 


```julia
prop = Dict("name"=> "PlaneStress", "rho"=> 0.0876584, "E"=>0.07180760098, "nu"=>0.4)

NT = 1000
Δt = 1.0e-3  
T = NT * dt

m, n =  20, 10
h = 0.1

# TODO: EBC, g, FBC, fext, gt, ft 
# Create a very simple mesh
coords = zeros((m+1)*(n+1), 2)
elements = 


ndims = 2
domain = Domain(nodes, elements, ndims, EBC, g, FBC, f)
# setGeometryPoints!(domain, npoints, node_to_point)


Dstate = zeros(domain.neqs)
state = zeros(domain.neqs)
velo = zeros(domain.neqs)
acce = zeros(domain.neqs)
globdat = GlobalData(state, Dstate, velo, acce, domain.neqs, gt, ft)


assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)

for i = 1:NT
    globdat, domain = ExplicitSolverStep(Δt, globdat, domain)
end
```