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
nx, ny =  1,2
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



function ggt(t)
    v = 0.01
    if t<1.0
        t*v*ones(sum(EBC.==-2))
    elseif t<3.0
        (0.02 - t*v)*ones(sum(EBC.==-2))
    end
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
# updateStates!(domain, globdat)



T = 0.8
NT = 5
Δt = T/NT
for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, -1.0, 0.0, 1e-6, 10)
    
end



nntype = "linear"
H_ = Variable(diagm(0=>ones(3)))
H = H_'*H_
# H = Variable(rand(3,3))
H0 = [250783699059.561126708984375 112852664576.802505493164063 0.000000000000000; 112852664576.802505493164063 250783699059.561126708984375 0.000000000000000; 0.000000000000000 0.000000000000000 68965517241.379318237304688]
H = constant(H0/1e11)

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
Fext, E_all = preprocessing(domain, globdat, F, Δt)
@info "Fext ", Fext
loss = DynamicMatLawLoss(domain, E_all, Fext, nn)
sess = Session(); init(sess)
@show run(sess, loss)
# BFGS(sess, loss)
# println("Real H = ", H0/1e11)
# run(sess, H)