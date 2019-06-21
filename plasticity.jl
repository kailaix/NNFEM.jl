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

fα = 0.0
fY = 10
fK = 1.0
function f(σ)
    return sqrt(σ[1]^2-σ[1]*σ[2]+σ[2]^2+3*σ[3]^2)-fY-fK*fα
end

function fσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    v = sqrt(σ1^2-σ1*σ2+σ2^2+3*σ3^2)
    z = [(2*σ1-σ2)/2/v;
        (2*σ2-σ1)/2/v;
        3*σ3/v]
end

function fσσ(σ)
    σ1, σ2, σ3 = σ[1], σ[2], σ[3]
    s1 = [(-σ1 + σ2/2)*(σ1 - σ2/2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) + 1/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)
    (-σ1 + σ2/2)*(-σ1/2 + σ2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) - 1/(2*sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2))
                                                       3*σ3*(-σ1 + σ2/2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2)]
    s2 = [(σ1/2 - σ2)*(σ1 - σ2/2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) - 1/(2*sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2))
    (-σ1/2 + σ2)*(σ1/2 - σ2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) + 1/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)
                                                   3*σ3*(σ1/2 - σ2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2)]
    s3 = [-3*σ3*(σ1 - σ2/2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2)
        -3*σ3*(-σ1/2 + σ2)/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2)
    -9*σ3^2/(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)^(3/2) + 3/sqrt(σ1^2 - σ1*σ2 + σ2^2 + 3*σ3^2)]
    out = [reshape(s1, 1, 3);reshape(s2, 1, 3);reshape(s3, 1, 3)]
end

function DσDε(σ, Δγ)
    rhs = [C;zeros(1,3)]
    A = [UniformScaling(1.0)+Δγ*E*fσσ(σ) -reshape(E*fσ(σ),3,1);
        reshape(fσ(σ),1,3) 0]
    out = A\rhs
    return out[1:3,:]
end

function Kt(σ)
    Kt = zeros(2n^2,2n^2) 
    for i = 1:size(elem,1)
        x1, y1 = nodes[elem[i,1],:]
        x2, y2 = nodes[elem[i,2],:]
        x3, y3 = nodes[elem[i,3],:]
        A = inv([x1 y1 1;x2 y2 1;x3 y3 1])
        Area = det([x1 y1 1;x2 y2 1;x3 y3 1])/2
        N = [A[1,1] 0 A[1,2] 0 A[1,3] 0;0 A[2,1] 0 A[2,2] 0 A[2,3];A[2,1] A[1,1] A[2,2] A[1,2] A[2,3] A[1,3]]
        ind = [elem[i,1];elem[i,1]+n^2;elem[i,2];elem[i,2]+n^2;elem[i,3];elem[i,3]+n^2] # index, (u1,u2,u3,v1,v2,v3)
        Q = DσDε(σ[3(i-1)+1:3i])
        B = β2*Δt^2*N'*Q*N*Area    # stiffness matrix
        Kt[ind,ind] += B 
    end
    remove_bd!(Kt)
    Kt
end

# NOTE Return Mapping Algorithm
function _rtm(Δσ, Δγ, σA, Δε)
    q1 = zeros(3ne); q2 = zeros(ne)
    J = zeros(4ne,4ne)
    Iz = zeros(Bool, 4ne)
    for i = 1:ne 
        B = Ns[i]
        q1[3(i-1)+1:3i] = Δσ[3(i-1)+1:3i] - C*Δε[3(i-1)+1:3i] + Δγ[i]*C*fσ(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i])
        q2[i] = f(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i])

        if q2[i]<0
            q1[3(i-1)+1:3i] = Δσ[3(i-1)+1:3i] - C*Δε[3(i-1)+1:3i]
            q2[i] = 0.0
            J[3(i-1)+1:3i,3(i-1)+1:3i] += UniformScaling(1.0)
            Iz[3ne+i] = true
        else
            J[3(i-1)+1:3i,3(i-1)+1:3i] += UniformScaling(1.0) + Δγ[i]*C*fσσ(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i])
            J[3(i-1)+1:3i,3ne+i] += C*fσ(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i])
            J[3ne+i,3(i-1)+1:3i] += fσ(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i])
        end
    end
    T = zeros(sum(Iz), sum(Iz)); T += UniformScaling(1.0)
    J[Iz, Iz] = T; J[.!Iz, Iz] .= 0.0; J[Iz, .!Iz] .= 0.0;
    -[q1;q2],J
end

function __rtm(Δσ,Δγ,σA,Δε)
    dσdε = Array{Float64}[]
    for i = 1:ne 
        temp = f(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i])
        if temp<0
            push!(dσdε, C)
        else
            A = [UniformScaling(1.0) + Δγ[i]*C*fσσ(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i]) C*fσ(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i]);
            fσ(Δσ[3(i-1)+1:3i]+σA[3(i-1)+1:3i]) 0]
            rhs = [C;zeros(1,3)]
            out = A\rhs
            push!(dσdε,out[1:3,:])
        end
    end
    dσdε
end

# rtm estimate dΔσ/dΔε
function rtm(Δε, σA)
    global fα
    tol = 1e-8
    Δσ = zeros(3ne);Δγ = zeros(ne)
    # initialization 
    for i = 1:ne 
        Δσ[3(i-1)+1:3i] = C*Δε[3(i-1)+1:3i]
    end
    for iter = 1:10000
        q, J = _rtm(Δσ, Δγ, σA, Δε)
        if norm(q)<tol
            break
        end
        δ = J\q 
        Δσ += q[1:3ne]; Δγ += q[3ne+1:end]
        if iter==10000
            error("Newton iteration fails")
        end
    end
    # update other things needed
    fα += Δγ
    dσdε = __rtm(Δσ,Δγ,σA,Δε)
    σA+Δσ, dσdε
end
        
# NOTE Large Newton loop
function _lnl(∂∂u, F, Δε, σA)
    r = M*∂∂u - F
    σB, dσdε = rtm(Δε, σA)
    Kt = zeros(nv, nv)
    for i = 1:ne
        B = Ns[i]; ind = Vs[i]
        loc = B'*dσdε*B*β2*Δt^2 * Areas[i]
        Kt[ind, ind] += loc
        r[ind] += B'*σB[3(i-1)+1:3i]*Areas[i]
    end
    remove_bd!(Kt)
    remove_bd!(r)
    -r, Kt, σB
end

function lnl(∂u, ∂∂uk, F, σA)
    local σB
    ∂∂u = copy(∂∂uk)
    for i = 1:10000
        Δu = Δt * ∂u + (1-2β2)/2*Δt^2*∂∂uk + β2*∂∂u
        Δε = zeros(3ne)
        for j = 1:ne 
            B = Ns[j]
            Δε[3(j-1)+1:3j] = B*Δu[Vs[j]]
        end
        q, J, σB = _lnl(∂∂u, F, Δε, σA)
        if norm(q)<1e-8
            return 
        end
        if i==10000
            error("Newton iteration fails")
        end
        δ = J\q 
        ∂∂u += δ
    end
    ∂∂u,σB
end

remove_bd!(K)
remove_bd!(M)


# imposing neumann boundary condition
F = zeros(2n^2)
# for i = 1:size(neumann,1)
#     ind = [neumann[i,1]; neumann[i,2]]
#     F[ind] += ones(2)*t
# end
# F[Dir] .= 0.0

# NOTE Newmark Algorithm 
NT = 100
Δt = 1/NT
U = zeros(2n^2,NT+1)
∂U = zeros(2n^2,NT+1)
∂∂U = zeros(2n^2,NT+1)
∂∂U[:,1] = M\(F-K*U[:,1])
Σ = zeros(3ne, NT+1)
∂U[n^2 .+ (1:n^2),1] .= 5.0; ∂U[Dir,1] .= 0.0
for k = 2:NT+1
    ∂∂U[:,k],Σ[:,k] = lnl(∂U[:,k-1], ∂∂U[:,k-1], F, Σ[:,k-1])
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
