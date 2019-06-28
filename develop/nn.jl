using PyPlot
using LinearAlgebra
using PyCall
using ADCME
using DelimitedFiles
reset_default_graph()

# * viscoplasticity or elasticity
type = :elasticity 

# * parameters for Newmark algorithm 
# Newmark Algorithm: http://solidmechanics.org/text/Chapter8_2/Chapter8_2.htm
β1 = 0.5
β2 = 0.5
NT = 100
Δt = 0.4/NT
n = 11
tol = 1e-3


function remove_bd!(K)
    if length(size(K))==1
        K[Dir] .= 0.0
    else
        T = UniformScaling(1.0)+zeros(2length(dirichlet),2length(dirichlet))
        K[Dir,Dir] = T
        K[Dir,.! Dir] .= 0.0
        K[.! Dir,Dir] .= 0.0
    end
end


# Domain: [0,1]^2
# Dirichlet boundary (u=0) : {0}x[0,1]
# Neumann boundary : {1}x[0,1] σ⋅n = (t1=0, t2), (up to a scalar).
#                    others: 0


# Data Structure
# nodes : nv x 2, coordinates per row
# elem : ne x 3, vertex index per row
nv = n^2; ne = 2(n-1)^2
nodes = zeros(nv,2)
elem = zeros(Int64, ne, 3)
for i = 1:n 
    for j = 1:n 
        nodes[i+(j-1)*n,:] = [(i-1)/(n-1) (j-1)/(n-1)]
    end
end

k = 1
for i = 1:n-1
    for j = 1:n-1
        global k
        ii = i + (j-1)*n 
        elem[k,:] = [ii;ii+1;n+ii+1]
        elem[k+1,:] = [ii;n+ii+1;n+ii]
        k += 2
    end
end

# store information for later use
Areas = Float64[]
Ns = Array{Float64}[]
Vs = Array{Int64}[]
for i = 1:size(elem,1)
    x1, y1 = nodes[elem[i,1],:]
    x2, y2 = nodes[elem[i,2],:]
    x3, y3 = nodes[elem[i,3],:]
    A = inv([x1 y1 1;x2 y2 1;x3 y3 1])
    Area = det([x1 y1 1;x2 y2 1;x3 y3 1])/2
    N = [A[1,1] 0 A[1,2] 0 A[1,3] 0;0 A[2,1] 0 A[2,2] 0 A[2,3];A[2,1] A[1,1] A[2,2] A[1,2] A[2,3] A[1,3]]
    ind = [elem[i,1];elem[i,1]+n^2;elem[i,2];elem[i,2]+n^2;elem[i,3];elem[i,3]+n^2] # index, (u1,u2,u3,v1,v2,v3)
    push!(Areas, Area)
    push!(Ns, N)
    push!(Vs, ind)
end

# Dirichlet and Neumann
dirichlet = 1:n:n*(n-1)+1|>Array
neumann = [2:n-1|>Array 3:n|>Array]
Dir = zeros(Bool, 2n^2)
Dir[dirichlet] .= true
Dir[dirichlet .+ n^2] .= true

# Constitutive law: σ=Cε
ν,E = 0.3, 2000
E = E/(1-ν^2)*[1 ν 0.0;ν 1 0.0;0.0 0.0 (1-ν)/2]

# * computing stiffness and mass matrix for linear elasticity 
K = zeros(2nv,2nv) # stiffness matrix
M = zeros(2nv,2nv) # mass matrix
for i = 1:ne
    Q = [1/3 0 1/3 0 1/3 0;0 1/3 0 1/3 0 1/3]
    B = Ns[i]'*E*Ns[i]*Areas[i]    # stiffness matrix
    D = Q'*Q*Areas[i]
    ind = Vs[i]
    K[ind,ind] += B 
    M[ind,ind] += D
end
remove_bd!(K)
remove_bd!(M)

tfE = Variable(diagm(0=>ones(3)))
function constitutive_law(Δε, ε)
    local σ
    Δε = tf.reshape(Δε,(ne,3)); ε = tf.reshape(ε,(ne,3))
    # * neural network constitutive law
    if type==:elasticity
        σ = (ε+Δε)*tfE 
        # σ = vec(σ')
    else
        σ = ae([Δε ε],[20,20,20,3],"nn")
        # σ = vec(σ')
    end
    return σ
end

tfNs_ = zeros(ne, 3, 6)
tfVs_ = zeros(Int32, ne, 6)
for i = 1:ne 
    tfNs_[i,:,:] = Ns[i]
    tfVs_[i,:] = Vs[i]
end
const tfNs = constant(tfNs_); const tfVs = constant(tfVs_); const tfAreas = constant(Areas)
function residual(∂∂u, F, ∂u, ∂∂uk, Δε, ε) 
    r = M*∂∂u - F
    σB = constitutive_law(Δε, ε)
    for i = 1:ne
        #@show i
        B = Ns[i]; ind = Vs[i]
        r = scatter_add(r, ind, B'*σB[i] * Areas[i])
    end
    sum(r[findall(.!Dir)]^2)
    # function cond0(i, ta)
    #     return i<=ne 
    # end
    # function body(i, ta)
    #     B = tfNs[i]; ind = tfVs[i]       
    #     r = read(ta, i)
    #     r = scatter_add(r, ind, B'*σB[i] * tfAreas[i])
    #     ta = write(ta, i+1, r)
    #     i+1, ta
    # end
    # i = constant(1,dtype=Int32)
    # ta = TensorArray(ne+1)
    # ta = write(ta, 1, r)
    # _, out = while_loop(cond0, body, [i, ta], parallel_iterations=50)
    # out = stack(out)
    # sum(out[ne+1][findall(.!Dir)]^2)
end

# imposing neumann boundary condition
F = zeros(2n^2)
# for i = 1:size(neumann,1)
#     ind = [neumann[i,1]; neumann[i,2]]
#     F[ind] += ones(2)*t
# end
# F[Dir] .= 0.0

# NOTE Newmark Algorithm 
# load data
using Random; Random.seed!(233)
# ∂U = rand(NT+1,2nv)
# ∂∂U = rand(NT+1,2nv)

∂U = readdlm("Data/∂U$type.txt"); @assert size(∂U)==(NT+1,2nv)
∂∂U = readdlm("Data/∂∂U$type.txt"); @assert size(∂∂U)==(NT+1,2nv)

Δε = zeros(NT+1, 3ne)
ε = zeros(NT+1,3ne)

# * precompute Δε and ε
for k=2:NT+1
    Δu = Δt * ∂U[k-1,:] + (1-β2)/2*Δt^2*∂∂U[k-1,:] + β2/2*Δt^2*∂∂U[k,:]
    for j = 1:ne 
        B = Ns[j]
        Δε[k, 3(j-1)+1:3j] = B*Δu[Vs[j]]
    end 
    ε[k,:] = ε[k-1,:] + Δε[k,:]
end
Δε = constant(Δε); ε = constant(ε)
∂U = constant(∂U); ∂∂U = constant(∂∂U)

function cond0(i, ta)
    i <= NT+1
end
function body(k, ta)
    l = residual(∂∂U[k], F, ∂U[k-1], ∂∂U[k-1], Δε[k], ε[k-1]) # ε[k-1] stress at last step
    # op = tf.print(k)
    # l = bind(l, op)
    ta = write(ta, k, l)
    k+1, ta
end

i = constant(2, dtype=Int32)
ta = TensorArray(NT+1)
ta = write(ta,1,constant(0.0))
_, out = while_loop(cond0, body, [i, ta], parallel_iterations=50)
loss = sum(stack(out))

# lr = placeholder(0.1, shape=())
# opt = AdamOptimizer(lr).minimize(loss)
sess = Session(); init(sess)
l0 = run(sess, loss)
println("Initial loss = $l0")
# error()

# for i = 1:10000
#     _, l_ = run(sess, [opt, loss], feed_dict=Dict(lr=>0.05))
#     #@show i, l_
# end
# error()

__cnt = 0
function print_loss(l)
    global __cnt
    if mod(__cnt,1)==0
        println("iter $__cnt, current loss=",l)
    end
    __cnt += 1
end
__iter = 0
function step_callback(rk)
    global __iter
    if mod(__iter,1)==0
        println("================ ITER $__iter ===============")
    end
    __iter += 1
end
opt = ScipyOptimizerInterface(loss, method="L-BFGS-B",options=Dict("maxiter"=> 30000, "ftol"=>1e-12, "gtol"=>1e-12))
#@show "Optimization starts..."
for j = 1:5
    ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=step_callback, fetches=[loss])
    save(sess, "$type$j.txt")
end

#=
# Set up formatting for the movie files
animation = pyimport("matplotlib.animation")
Writer = animation.writers.avail["html"]
writer = Writer(fps=15, bitrate=1800)

close("all")
fig = figure()
# visualization
scat0 = scatter(nodes[:,1], nodes[:,2], color="k")
grid(true)
ims = Any[(scat0,)]

for k = 1:NT+1
    u1 = U[1:n^2,k] + nodes[:,1]
    u2 = U[n^2+1:end,k] + nodes[:,2]

    scat = scatter(u1, u2, color="orange")
    grid("on")
    tt = gca().text(.5, 1.05,"t = $(round((k-1)*Δt,digits=3))")
    push!(ims, (scat,scat0,tt))
end

im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                   blit=true)
im_ani.save("im$type.html", writer=writer)
=#