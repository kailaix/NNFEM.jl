using PyPlot
using LinearAlgebra

t = 0.0
β1 = 0.5
β2 = 0.5
# Newmark Algorithm: http://solidmechanics.org/text/Chapter8_2/Chapter8_2.htm

# Domain: [0,1]^2
# Dirichlet boundary (u=0) : {0}x[0,1]
# Neumann boundary : {1}x[0,1] σ⋅n = (t1=0, t2), (up to a scalar).
#                    others: 0


# Data Structure
# nodes : nv x 2, coordinates per row
# elem : ne x 3, vertex index per row
n = 10
nodes = zeros(n^2,2)
elem = zeros(Int64, 2(n-1)^2, 3)
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

# Dirichlet and Neumann
dirichlet = 1:n:n*(n-1)+1|>Array
neumann = [2:n-1|>Array 3:n|>Array]

# Constitutive law: σ=Cε
ν,E = 0.3, 2000
C = E/(1-ν^2)*[1 ν 0.0;ν 1 0.0;0.0 0.0 (1-ν)/2]

# Assembling: Dense matrix
K = zeros(2n^2,2n^2) # stiffness matrix
M = zeros(2n^2,2n^2) # mass matrix
for i = 1:size(elem,1)
    x1, y1 = nodes[elem[i,1],:]
    x2, y2 = nodes[elem[i,2],:]
    x3, y3 = nodes[elem[i,3],:]
    A = inv([x1 y1 1;x2 y2 1;x3 y3 1])
    Area = det([x1 y1 1;x2 y2 1;x3 y3 1])/2
    N = [A[1,1] 0 A[1,2] 0 A[1,3] 0;0 A[2,1] 0 A[2,2] 0 A[2,3];A[2,1] A[1,1] A[2,2] A[1,2] A[2,3] A[1,3]]
    Q = [1/3 0 1/3 0 1/3 0;0 1/3 0 1/3 0 1/3]
    B = N'*C*N*Area    # stiffness matrix
    D = Q'*Q*Area
    ind = [elem[i,1];elem[i,1]+n^2;elem[i,2];elem[i,2]+n^2;elem[i,3];elem[i,3]+n^2] # index, (u1,u2,u3,v1,v2,v3)
    K[ind,ind] += B 
    M[ind,ind] += D
end


D = zeros(Bool, 2n^2)
D[dirichlet] .= true
D[dirichlet .+ n^2] .= true
T = UniformScaling(1.0)+zeros(2length(dirichlet),2length(dirichlet))
K[D,D] = T
K[D,.! D] .= 0.0
K[.! D,D] .= 0.0
M[D,D] = T
M[D,.! D] .= 0.0
M[.! D,D] .= 0.0

# imposing neumann boundary condition
f = zeros(2n^2)
for i = 1:size(neumann,1)
    ind = [neumann[i,1]; neumann[i,2]]
    f[ind] += ones(2)*t
end
f[D] .= 0.0


# time stepping, assume u(0) = u'(0) = 0
NT = 100
Δt = 1/NT
U = zeros(2n^2,NT+1)
∂U = zeros(2n^2,NT+1)
∂∂U = zeros(2n^2,NT+1)
∂∂U[:,1] = M\(f-K*U[:,1])
∂U[n^2 .+ (1:n^2),1] .= 5.0; ∂U[D,1] .= 0.0
L = M + β2*Δt^2*K 
for k = 2:NT+1
    g = -K*(U[:,k-1]+Δt*∂U[:,k-1]+Δt^2/2*(1-β2)*∂∂U[:,k-1])+f
    ∂∂U[:,k] = L\g 
    U[:,k] = U[:,k-1] + Δt*∂U[:,k-1] + Δt^2/2*((1-β2)*∂∂U[:,k-1]+β2*∂∂U[:,k])
    ∂U[:,k] = ∂U[:,k-1] + Δt*((1-β1)*∂∂U[:,k-1]+β1*∂∂U[:,k])
end


# visualization
scatter(nodes[:,1], nodes[:,2])
scat = scatter(u1, u2)
grid("on")
ims = []

for k = 1:NT+1
    u1 = U[1:n^2,k] + nodes[:,1]
    u2 = U[n^2+1:end,k] + nodes[:,2]
    scat.set_offsets([u1 u2])
    title("Δt = $(round((k-1)*Δt, digits=2))")
    pause(0.1)
    push!(ims, gcf())
end
