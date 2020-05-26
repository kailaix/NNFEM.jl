using Revise
using NNFEM
using PyPlot
using MAT
using ADCME


NT = 100
T = 0.5
Δt = T/NT

m, n =  15, 15
h = 1/m

ν = 0.35
E = 2.0
H = zeros(3,3)
H[1,1] = E*(1. -ν)/((1+ν)*(1. -2. *ν));
H[1,2] = H[1,1]*ν/(1-ν);
H[2,1] = H[1,2];
H[2,2] = H[1,1];
H[3,3] = H[1,1]*0.5*(1. -2. *ν)/(1. -ν);


# Create a very simple mesh
elements = []
prop = Dict("name"=> "PlaneStrain", "rho"=> 1.0, "E"=> 2.0, "nu"=> 0.35)
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

# fixed on the bottom, push on the right
EBC = zeros(Int64, (m+1)*(n+1), 2)
FBC = zeros(Int64, (m+1)*(n+1), 2)
g = zeros((m+1)*(n+1), 2)
f = zeros((m+1)*(n+1), 2)

for i = 2:m
    idx = n*(m+1)+i
    EBC[idx,:] .= -2
    EBC[i,:] .= -2
end

for j = 1:n+1 
    idx = (j-1)*(m+1)+1
    EBC[idx,:] .= -2
    idx = (j-1)*(m+1)+m+1
    EBC[idx,:] .= -2
end




ndims = 2
domain = Domain(coords, elements, ndims, EBC, g, FBC, f)


EBC_idx = findall(EBC[:,1] .== -2)
EBC_X = domain.nodes[EBC_idx,:]
function gt(t)
    x, y = EBC_X[:,1], EBC_X[:,2]
    u1 = @. (x^2+y^2)*exp(-t) * 0.1
    u2 = @. (x^2-y^2)*exp(-t) * 0.1
    return [u1;u2], -[u1;u2], [u1;u2]
end

FBC_idx = findall(FBC[:,1] .== -2)
FBC_X = domain.nodes[FBC_idx,:]
# function ft(t)
#     x, y = FBC_X[:,1], FBC_X[:,2]
#     T1 = @. -1.48148148148148*x*exp(-t) - 1.48148148148148*y*exp(-t)
#     T2 = @. -3.45679012345679*x*exp(-t) - 6.41975308641975*y*exp(-t)
#     return [T1;T2] * h * 0.1
# end

function bt(x, y, t)
    f1 = @. 0.1*(x^2 + y^2)*exp(-t) - 0.790123456790123*exp(-t)
    f2 = @. 0.1*(x^2 - y^2)*exp(-t) + 0.493827160493827*exp(-t)
    [f1 f2] 
end

c = domain.nodes[domain.dof_to_eq]
nn = div(length(c), 2)
x = c[1:nn]
y = c[nn+1:end]

Dstate =  @. [x^2+y^2;x^2-y^2]* 0.1
state = @. [x^2+y^2;x^2-y^2]* 0.1
velo =  @. -[x^2+y^2;x^2-y^2]* 0.1
acce =  @. [x^2+y^2;x^2-y^2]* 0.1
globdat = GlobalData(state, Dstate, velo, acce, domain.neqs, gt, nothing, bt)

assembleMassMatrix!(globdat, domain)
# updateTimeDependentEssentialBoundaryCondition!(domain,globdat)
# updateStates!(domain, globdat)

# for i = 1:NT
#     @info i 
#     # global globdat, domain = GeneralizedAlphaSolverStep(globdat, domain, Δt)
#     global globdat, domain = ExplicitSolverStep(globdat, domain, Δt)
# end


# d_ = hcat(domain.history["state"]...)'
# i = 3
# j = 4
# plot(d_[:,(j-1)*(m+1)+i])
# x0 = (i-1)*h 
# y0 = (j-1)*h
# ts = LinRange(0, 1, NT+1)
# plot((@. (x0^2+y0^2)*exp(-ts))*0.1,"--")


# error()
Hs = zeros(domain.neles*length(domain.elements[1].weights), 3, 3)
for i = 1:size(Hs,1)
    Hs[i,:,:] = H 
end

# ts = GeneralizedAlphaSolverTime(Δt, NT)
ts = ExplicitSolverTime(Δt, NT)

ubd = zeros(NT,sum(domain.EBC[:].!=0))
abd = zeros(NT,sum(domain.EBC[:].!=0))
Fext = zeros(NT,domain.neqs)

idx = findall(domain.EBC[:,1].!=0)
x = domain.nodes[idx,1]
y = domain.nodes[idx,2]
for i = 1:NT
    t = ts[i]
    ubd[i,:] = [
        @. (x^2+y^2)*0.1*exp(-t)
        @. (x^2-y^2)*0.1*exp(-t)
    ]
    abd[i,:] = [
        @. (x^2+y^2)*0.1*exp(-t)
        @. (x^2-y^2)*0.1*exp(-t)
    ]
    globdat.time = t
    Fext[i,:] = getBodyForce(domain, globdat)
end

x = domain.nodes[:,1]
y = domain.nodes[:,2]
a0 = @. [x^2+y^2;x^2-y^2]* 0.1
v0 =  @. -[x^2+y^2;x^2-y^2]* 0.1
d0 =  @. [x^2+y^2;x^2-y^2]* 0.1


H = constant(Hs)


function linear_law(ε)
    @info H, ε
    batch_matmul(H, ε)
end

function nn_law(ε)
    ae(ε, [20,20,20,3])
end
# d, v, a= GeneralizedAlphaSolver(globdat, domain, d0, v0, a0, Δt, NT, H, Fext, ubd, abd)
# d, v, a= ExplicitSolver(globdat, domain, d0, v0, a0, Δt, NT, nn_law, Fext, ubd, abd)
d, v, a= ExplicitSolver(globdat, domain, d0, v0, a0, Δt, NT, nn_law, Fext, ubd, abd; strain_type="finite")



# # ...1... use `linear_law` for 8th argument in the last line to generate data
# sess = Session(); init(sess)
# d_, v_, a_ = run(sess, [d,v,a])
# matwrite("data.mat", Dict("d"=>d_, "v"=>v_, "a"=>a_))
# using Random; Random.seed!(233)
# k = 0
# for i = 1:5
#     i = rand(2:m)
#     j = rand(2:n)
#     x0 = (i-1)*h 
#     y0 = (j-1)*h
#     global k += 1
#     plot(d_[:,(j-1)*(m+1)+i], "o", markersize=2,color = "C$k")
#     plot((@. (x0^2+y0^2)*exp(-ts))*0.1,"-", color="C$k")

#     global k += 1
#     plot(d_[:,(j-1)*(m+1)+i+(m+1)*(n+1)], "o", markersize=2,color = "C$i")
#     plot((@. (x0^2-y0^2)*exp(-ts))*0.1,"-", color="C$i")
# end
# error()


# ...2...comment the last block and replace `linear_law` with `nn_law` to train a neural network
dobs = matread("data.mat")["d"]
loss = sum((d - dobs)^2)
BFGS!(sess, loss)
