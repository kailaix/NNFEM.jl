using Revise
using NNFEM 
using PyPlot
using LinearAlgebra
using ADCME
using ADCMEKit
using MAT 


NT = 100
Δt = 1/NT 

node, elem = load_mesh("holes")

elements = SmallStrainContinuum[]
prop = Dict("name"=> "PlaneStress", "rho"=> 8000,  "E"=>200e9, "nu"=>0.3)
for i = 1:size(elem,1)
    nodes = node[elem[i,:], :]
    elnodes = elem[i,:]
    ngp = 3
    push!(elements, SmallStrainContinuum(nodes, elnodes, prop, ngp))
end

Edge_Traction_Data = Array{Int64}[]
for i = 1:length(elements)
    elem = elements[i]
    for k = 1:4
        if elem.coords[k,1]>99.5 && elem.coords[k+1>4 ? 1 : k+1,1]>99.5
            push!(Edge_Traction_Data, [i, k, 1])
        end
    end
end
Edge_Traction_Data = hcat(Edge_Traction_Data...)'|>Array


EBC = zeros(Int64, size(node, 1), 2)
FBC = zeros(Int64, size(node, 1), 2)
g = zeros(size(node, 1), 2)
f = zeros(size(node, 1), 2)
for i = 1:size(node, 1)
    if node[i,1]<0.05
        EBC[i,:] .= -1
    end
end

domain = Domain(node, elements, 2, EBC, g, FBC, f, Edge_Traction_Data)


Dstate = zeros(domain.neqs) 
state = zeros(domain.neqs)
velo = zeros(domain.neqs)
acce = zeros(domain.neqs)
EBC_func = nothing 
FBC_func = nothing 
Body_func = nothing 
# Construct Edge_func
function Edge_func(x, y, t, idx)
  return [1e6*ones(length(x)) zeros(length(x))]
end
globaldata = GlobalData(state, Dstate, velo, acce, domain.neqs, EBC_func, FBC_func,Body_func, Edge_func)

d0 = zeros(2domain.nnodes)
Fext = getExternalForce!(domain, globaldata)

H = constant(domain.elements[1].mat[1].H)

sess = Session(); init(sess)

# incremental load method 
# d = ImplicitStaticSolver(globaldata, domain, d0, Δt, NT, H, Fext, method="incremental")
# d_ = run(sess, d)
# visualize_displacement(d_*25199.1, domain)


# # newton's method 
# d = ImplicitStaticSolver(globaldata, domain, d0, Δt, NT, H, Fext, method="newton")
# sess = Session(); init(sess)
# d_ = run(sess, d)
# visualize_displacement(reshape(d_, 1, :)*25199.1, domain)

# newton's method (also work for nonlinear equations)

H = domain.elements[1].mat[1].H
function nn(ε, θ)
    H = reshape(θ, (3,3))
    Hs = repeat(reshape(H, (-1,)), getNGauss(domain))
    Hs = reshape(Hs, (getNGauss(domain), 3, 3) )
    ε*H, Hs
end
ADCME.options.newton_raphson.verbose = true 
ADCME.options.newton_raphson.rtol = 1e-6
ADCME.options.newton_raphson.tol = 1e-6
d = ImplicitStaticSolver(globaldata, domain, d0,  nn, H[:], Fext)
d_ = run(sess, d)
visualize_displacement(reshape(d_, 1, :)*25199.1, domain)

# println("Press any key to learn the linear elastic matrix...")
# readline()



# H = spd(Variable(rand(3,3))) * 1e10
# d = ImplicitStaticSolver(globaldata, domain, d0, Δt, NT, H, Fext)
# loss = sum((d - d_)^2)
# init(sess)
# BFGS!(sess, loss)

# println("Estimated Stiffness Matrix:\n$(run(sess, H))")
# println("Exact Stiffness Matrix:\n$(domain.elements[1].mat[1].H)")