using PyPlot
using LinearAlgebra

t = 1000


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
neumann = [n:n:n^2-n|>Array 2n:n:n^2|>Array]

# Constitutive law: σ=Cε
ν,E = 0.3, 2000
C = E/(1-ν^2)*[1 ν 0.0;ν 1 0.0;0.0 0.0 (1-ν)/2]

# Assembling: Dense matrix
M = zeros(2n^2,2n^2)
for i = 1:size(elem,1)
    x1, y1 = nodes[elem[i,1],:]
    x2, y2 = nodes[elem[i,2],:]
    x3, y3 = nodes[elem[i,3],:]
    A = inv([x1 y1 1;x2 y2 1;x3 y3 1])
    N = [A[1,1] 0 A[1,2] 0 A[1,3] 0;0 A[2,1] 0 A[2,2] 0 A[2,3];A[2,1] A[1,1] A[2,2] A[1,2] A[2,3] A[1,3]]
    B = N'*C*N    # stiffness matrix
    ind = [elem[i,1];elem[i,1]+n^2;elem[i,2];elem[i,2]+n^2;elem[i,3];elem[i,3]+n^2] # index, (u1,u2,u3,v1,v2,v3)
    M[ind,ind] += B 
end

# imposing neumann boundary condition
f = zeros(2n^2)
for i = 1:size(neumann,1)
    ind = [neumann[i,1]; neumann[i,2]]
    f[ind .+ n^2] += ones(2)*t
end

# imposing dirichlet boundary condition
D = zeros(Bool, 2n^2)
D[dirichlet] .= true
D[dirichlet .+ n^2] .= true
T = UniformScaling(1.0)+zeros(2length(dirichlet),2length(dirichlet))
M[D,D] = T
M[D,.! D] .= 0.0
M[.! D,D] .= 0.0
f[D] .= 0.0
u = M\f

u1 = u[1:n^2] + nodes[:,1]
u2 = u[n^2+1:end] + nodes[:,2]

# visualization
scatter(nodes[:,1], nodes[:,2])
scatter(u1, u2)
grid("on")