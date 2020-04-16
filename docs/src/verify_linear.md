# Linear Elasticity

In this example, we consider the linear elasticity problem. The strain stress relation is given by 

$$\sigma = H \epsilon$$

where the linear elasticity matrix $H$ is given by 

```julia
H = elements[1].mat[1].H
```

## Analytical Solution

We can use `SymPy.jl` to generate the analytical solution

```julia
using SymPy
H = elements[1].mat[1].H
x, y, t = @vars x y t
u = 0.1*(1-y^2)*(x^2+y^2)*exp(-t)
v = 0.1* (1-y^2)*(x^2+y^2)*exp(-t)
ux = diff(u,x)
uy = diff(u,y)
vx = diff(v,x)
vy = diff(v,y)
ϵ = [ux;vy;uy+vx]
σ = H * ϵ
f1 = u - (div(σ[1], x) + div(σ[3], y))
f2 = v - (div(σ[3], x) + div(σ[2], y))
println(replace(replace(sympy.julia_code(f1), ".*"=>"*"), ".^"=>"^"))
println(replace(replace(sympy.julia_code(f2), ".*"=>"*"), ".^"=>"^"))

S = [σ[1] σ[3]
	σ[3] σ[2]]
# edge function on domain 0
t1 = S * [1;0.0]
# edge function on domain 1
t2 = S * [0;-1.0]

println(replace(replace(sympy.julia_code(t1[1]), ".*"=>"*"), ".^"=>"^"))
println(replace(replace(sympy.julia_code(t1[2]), ".*"=>"*"), ".^"=>"^"))

println(replace(replace(sympy.julia_code(t2[1]), ".*"=>"*"), ".^"=>"^"))
println(replace(replace(sympy.julia_code(t2[2]), ".*"=>"*"), ".^"=>"^"))


```

## Forward Computation

To conduct forward computation using AD-enabled kernels, we need to precompute some data. This includes the boundary conditions and external force

```julia
# linear elasticity matrix at each Gauss point
Hs = zeros(domain.neles*length(domain.elements[1].weights), 3, 3)
for i = 1:size(Hs,1)
    Hs[i,:,:] = elements[1].mat[1].H
end

# Construct Edge_func
function Edge_func_linear_elasticity(x, y, t, idx)
  if idx==0
      f1 = @. 0.307746861342857*x*(0.1 - 0.1*y^2)*exp(-t) + 0.205164574228572*y*(0.1 - 0.1*y^2)*exp(-t) - 0.0205164574228572*y*(x^2 + y^2)*exp(-t)
      f2 = @. 0.0512911435571429*x*(0.1 - 0.1*y^2)*exp(-t) + 0.0512911435571429*y*(0.1 - 0.1*y^2)*exp(-t) - 0.00512911435571429*y*(x^2 + y^2)*exp(-t)
    elseif idx==1
      f1 = @. -0.0512911435571429*x*(0.1 - 0.1*y^2)*exp(-t) - 0.0512911435571429*y*(0.1 - 0.1*y^2)*exp(-t) + 0.00512911435571429*y*(x^2 + y^2)*exp(-t)
      f2 = @. -0.205164574228572*x*(0.1 - 0.1*y^2)*exp(-t) - 0.307746861342857*y*(0.1 - 0.1*y^2)*exp(-t) + 0.0307746861342857*y*(x^2 + y^2)*exp(-t)
    end
    return [f1 f2]
end

globaldata.Edge_func = Edge_func_linear_elasticity
  
ts = ExplicitSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts)
```

Finally, we can carry out forward computation

```julia
d, v, a= ExplicitSolver(globaldata, domain, d0, v0, a0, Δt, NT, Hs, Fext, ubd, abd)

sess = Session(); init(sess)
d_, v_, a_ = run(sess, [d,v,a])
```

The computation can be verified by comparing with exact solutions 

```julia
for i = 1:5
    i = rand(1:m+1)
    j = rand(1:n+1)
    plot(d_[:,(j-1)*(m+1)+i], color = "C$i", label="Computed")
    x0 = (i-1)*h 
    y0 = (j-1)*h
    plot((@. (1-y0^2)*(x0^2+y0^2)*exp(-ts))*0.1,"--", color="C$i", label="Reference")
end
legend()
```



