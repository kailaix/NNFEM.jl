using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra




testtype = "PlaneStress"
np = pyimport("numpy")
nx, ny =  20, 10
nnodes, neles = (nx + 1)*(ny + 1), nx*ny
x = np.linspace(0.0, 0.5, nx + 1)
y = np.linspace(0.0, 0.5, ny + 1)
X, Y = np.meshgrid(x, y)
nodes = zeros(nnodes,2)
nodes[:,1], nodes[:,2] = X'[:], Y'[:]
ndofs = 2

EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)

EBC[collect(1:nx+1), :] .= -1
EBC[collect((nx+1)*ny + 1:(nx+1)*ny + nx+1), 1] .= -1
#pull in the y direction
EBC[collect((nx+1)*ny + 1:(nx+1)*ny + nx+1), 2] .= -2



# function ggt(t)
#     v = 0.01
#     if t<1.0
#         t*v*ones(sum(EBC.==-2))
#     elseif t<3.0
#         (0.02 - t*v)*ones(sum(EBC.==-2))
#     end
# end

function ggt(t)
    v = 0.01
    f = 0.5
    k = sum(EBC.==-2)
    u  = Array{Float64}(1:k)*2
    state = (2π*f*t)*v*u
    acc = -(2π*f)^2*sin(2π*f*t)*v*u
    acc = zeros(size(acc))
    # @show state
    state, acc
end
gt = ggt


NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)



prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45,
"sigmaY"=>0.3e+9, "K"=>1/9*200e+9)

elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx+1)*(j-1) + i
        elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop,2))
    end
end


domain = Domain(nodes, elements, ndofs, EBC, g, NBC, f)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs),
                    zeros(domain.neqs),∂u, domain.neqs, gt)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)



T = 0.2
NT = 5
Δt = T/NT
for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-3, 10)
    
end



nntype = "linear"
H_ = Variable(diagm(0=>ones(3)))
H = H_'*H_

E = prop["E"]; ν = prop["nu"]; ρ = prop["rho"]
H0 = zeros(3,3)

H0[1,1] = E/(1. -ν*ν)
H0[1,2] = H0[1,1]*ν
H0[2,1] = H0[1,2]
H0[2,2] = H0[1,1]
H0[3,3] = E/(2.0*(1.0+ν))

H0 /= 1e11

# H = Variable(H0.+1)
# H = Variable(H0+rand(3,3))
E = Variable(1.0)
ν = Variable(0.0)
H = [
    E/(1-ν*ν) ν*E/(1-ν*ν) constant(0.0);
    ν*E/(1-ν*ν) E/(1-ν*ν) constant(0.0);
    constant(0.0) constant(0.0)  E/(2.0*(1.0+ν))
]
H = tensor(H)


function nn(ε, ε0, σ0)
    local y
    if nntype=="linear"
        y = ε*H*1e11
        # op1 = tf.print("* ", ε,summarize=-1)
        # y = bind(y, op1)
        # op2 = tf.print("& ", y, summarize=-1)
        # y = bind(y, op2)
    elseif nntype=="nn"
        x = [ε ε0 σ0]
        y = ae(x, [20,20,20,20,3], "nn")
    end
    y
end


F = zeros(domain.neqs, NT+1)
Ftot, E_all = preprocessing(domain, globdat, F, Δt)

# @info "Fext ", Fext
loss = DynamicMatLawLoss(domain, E_all, Ftot, nn)/1e8
sess = Session(); init(sess)
@show run(sess, loss)
BFGS!((sess, loss, 50)
println("Real H = ", H0)
run(sess, H)
run(sess, gradients(loss, H))