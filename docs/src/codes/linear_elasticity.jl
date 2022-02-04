using Revise
using NNFEM 
using PyPlot
using LinearAlgebra
using ADCME

NT = 200
Δt = 1/NT 

n = 10
m = 2n 
h = 1/n

# Create a very simple mesh
elements = SmallStrainContinuum[]
prop = Dict("name"=> "PlaneStrain", "rho"=> 1.0, "E"=>2.0, "nu"=>0.35)
coords = zeros((m+1)*(n+1), 2)
for j = 1:n
    for i = 1:m
        idx = (m+1)*(j-1)+i 
        elnodes = [idx; idx+1; idx+1+m+1; idx+m+1]
        ngp = 3
        nodes = [
            (i-1)*h (j-1)*h
            i*h (j-1)*h
            i*h j*h
            (i-1)*h j*h
        ]
        coords[elnodes, :] = nodes
        push!(elements, SmallStrainContinuum(nodes, elnodes, prop, ngp))
    end
end

Edge_Traction_Data = Array{Int64}[]
for i = 1:m 
  elem = elements[i]
  for k = 1:4
    if elem.coords[k,2]<0.001 && elem.coords[k+1>4 ? 1 : k+1,2]<0.001
      push!(Edge_Traction_Data, [i, k, 1])
    end
  end
end

for i = 1:n
  elem = elements[(i-1)*m+1]
  for k = 1:4
    if elem.coords[k,1]<0.001 && elem.coords[k+1>4 ? 1 : k+1,1]<0.001
      push!(Edge_Traction_Data, [(i-1)*m+1, k, 0])
    end
  end
end

Edge_Traction_Data = hcat(Edge_Traction_Data...)'|>Array

# fixed on the bottom, push on the right
EBC = zeros(Int64, (m+1)*(n+1), 2)
FBC = zeros(Int64, (m+1)*(n+1), 2)
g = zeros((m+1)*(n+1), 2)
f = zeros((m+1)*(n+1), 2)
for j = 1:n+1
    idx = (j-1)*(m+1) + m+1
    EBC[idx,:] .= -2 # time-dependent boundary, right
end
for i = 1:m+1
    idx = n*(m+1) + i 
    EBC[idx,:] .= -1 # fixed boundary, bottoms
end

dimension = 2
domain = Domain(coords, elements, dimension, EBC, g, FBC, f, Edge_Traction_Data)

xy = domain.nodes[domain.dof_to_eq]
n_ = div(length(xy),2)
x = xy[1:n_,1]
y = xy[n_+1:end,1]
# Set initial condition 
Dstate = zeros(domain.neqs) # d at last step 
state = [(@. (1-y^2)*(x^2+y^2)); (@. (1-y^2)*(x^2-y^2))] * 0.1 
velo = -[(@. (1-y^2)*(x^2+y^2)); (@. (1-y^2)*(x^2-y^2))] * 0.1 
acce = [(@. (1-y^2)*(x^2+y^2)); (@. (1-y^2)*(x^2-y^2))] * 0.1 
gt = nothing
ft = nothing

EBC_DOF = findall(EBC[:,1] .== -2)
x_EBC = domain.nodes[EBC_DOF,1]
y_EBC = domain.nodes[EBC_DOF,2]
function EBC_func(t)
  out = [(@. 0.1*(1-y_EBC^2)*(x_EBC^2+y_EBC^2)*exp(-t));
    (@. 0.1*(1-y_EBC^2)*(x_EBC^2-y_EBC^2)*exp(-t))]
  
  out, -out, out
end

function Body_func_linear_elasticity(x, y, t)
    f1 = @. 0.987654320987654*x*y*exp(-t) + 0.592592592592593*y^2*exp(-t) + (0.1 - 0.1*y^2)*(x^2 + y^2)*exp(-t) - (0.148148148148148 - 0.148148148148148*y^2)*exp(-t) - (0.641975308641975 - 0.641975308641975*y^2)*exp(-t) - (-0.148148148148148*x^2 - 0.148148148148148*y^2)*exp(-t)
    f2 = @. 0.987654320987654*x*y*exp(-t) - 2.5679012345679*y^2*exp(-t) + (0.1 - 0.1*y^2)*(x^2 - y^2)*exp(-t) - (0.148148148148148 - 0.148148148148148*y^2)*exp(-t) - (-0.641975308641975*x^2 + 0.641975308641975*y^2)*exp(-t) - (0.641975308641975*y^2 - 0.641975308641975)*exp(-t)
    [f1 f2] 
end

FBC_func = nothing 
Edge_func = nothing
globaldata = GlobalData(state, Dstate, velo, acce, domain.neqs, EBC_func, FBC_func,Body_func_linear_elasticity, Edge_func)

x = domain.nodes[:,1]
y = domain.nodes[:,2]
d0 = [(@. (1-y^2)*(x^2+y^2)); (@. (1-y^2)*(x^2-y^2))] * 0.1 
v0 = -[(@. (1-y^2)*(x^2+y^2)); (@. (1-y^2)*(x^2-y^2))] * 0.1 
a0 = [(@. (1-y^2)*(x^2+y^2)); (@. (1-y^2)*(x^2-y^2))] * 0.1 
assembleMassMatrix!(globaldata, domain)

# linear elasticity matrix at each Gauss point
Hs = zeros(domain.neles*length(domain.elements[1].weights), 3, 3)
for i = 1:size(Hs,1)
    Hs[i,:,:] = elements[1].mat[1].H
end

# Construct Edge_func
function Edge_func_linear_elasticity(x, y, t, idx)
  if idx==0
      f1 = @. -6.41975308641975*x*(0.1 - 0.1*y^2)*exp(-t) + 3.45679012345679*y*(0.1 - 0.1*y^2)*exp(-t) + 0.345679012345679*y*(x^2 - y^2)*exp(-t)
      f2 = @. -1.48148148148148*x*(0.1 - 0.1*y^2)*exp(-t) - 1.48148148148148*y*(0.1 - 0.1*y^2)*exp(-t) + 0.148148148148148*y*(x^2 + y^2)*exp(-t)
    elseif idx==1
      f1 = @. -1.48148148148148*x*(0.1 - 0.1*y^2)*exp(-t) - 1.48148148148148*y*(0.1 - 0.1*y^2)*exp(-t) + 0.148148148148148*y*(x^2 + y^2)*exp(-t)
      f2 = @. -3.45679012345679*x*(0.1 - 0.1*y^2)*exp(-t) + 6.41975308641975*y*(0.1 - 0.1*y^2)*exp(-t) + 0.641975308641975*y*(x^2 - y^2)*exp(-t)
    end
    return [f1 f2] 
end
globaldata.Edge_func = Edge_func_linear_elasticity
  
ts = ExplicitSolverTime(Δt, NT)
ubd, abd = compute_boundary_info(domain, globaldata, ts)
Fext = compute_external_force(domain, globaldata, ts) 
d, v, a= ExplicitSolver(globaldata, domain, d0, v0, a0, Δt, NT, Hs, Fext, ubd, abd)

# # NOTE: You can also use the implicit alpha solvers
# ts = GeneralizedAlphaSolverTime(Δt, NT)
# ubd, abd = compute_boundary_info(domain, globaldata, ts)
# Fext = compute_external_force(domain, globaldata, ts) 
# d, v, a= GeneralizedAlphaSolver(globaldata, domain, d0, v0, a0, Δt, NT, Hs, Fext, ubd, abd)

sess = Session(); init(sess)
d_, v_, a_ = run(sess, [d,v,a])


using Random; Random.seed!(233)
for k = 1:5
    i = rand(1:m+1)
    j = rand(1:n+1)
    if k==1
        plot(d_[:,(j-1)*(m+1)+i], color = "C$k", label="Computed")
    else
        plot(d_[:,(j-1)*(m+1)+i], color = "C$k")
    end
    x0 = (i-1)*h 
    y0 = (j-1)*h

    if k==1
        plot((@. (1-y0^2)*(x0^2+y0^2)*exp(-ts))*0.1 ,"--", color="C$k", label="Reference")
    else
        plot((@. (1-y0^2)*(x0^2+y0^2)*exp(-ts))*0.1 ,"--", color="C$k")
    end
end
legend()
