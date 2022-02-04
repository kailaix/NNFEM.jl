# SmallStrainContinuum

prop = Dict("name"=> "PlaneStrain", "rho"=> 1.0, "E"=> 2.0, "nu"=> 0.35)
ngp = 2
elnodes = [1;2;3;4]
nodes = [
    0.0 0.0
    1.0 0.0
    1.0 1.0
    0.0 1.0
] 
element =  SmallStrainContinuum(nodes, elnodes, prop, ngp)

ν = 0.35
E = 2.0

H = zeros(3,3)
H[1,1] = E*(1. -ν)/((1+ν)*(1. -2. *ν));
H[1,2] = H[1,1]*ν/(1-ν);
H[2,1] = H[1,2];
H[2,2] = H[1,1];
H[3,3] = H[1,1]*0.5*(1. -2. *ν)/(1. -ν);


@testset "small strain gauss" begin
    ggp = getGaussPoints(e)
    c = e.coords
    scatter(c[:,1], c[:,2])
    scatter(ggp[:,1], ggp[:,2])
end

@testset "body force" begin
    ggp = getGaussPoints(element)
    f = @. ggp[:,1]^2 + ggp[:,2]^2
    fbody = getBodyForce(element, [f f])
    idx = [1;2;4;3]
    idx = [idx; idx .+ 4]
    @test fbody[idx] ≈ compute_fem_source_term(f, f, 1, 1, 1.) 
end


m, n =  2, 2
h = 1/m

# Create a very simple mesh
elements = SmallStrainContinuum[]
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

for i = 1:m+1
    for j = 1:n+1 
        if i==1 || i==m+1 || j == 1|| j==n+1
            idx = (j-1)*(m+1) + i 
            FBC[idx,:] .= -1
        end
    end
end

ndims = 2
domain = Domain(coords, elements, ndims, EBC, g, FBC, f)

function bt(x, y, t)
    f1 = @. x^2+y^2 
    f2 = @. x^2+y^2
    [f1 f2]
end

Dstate = zeros(domain.neqs)
state = rand(domain.neqs)
velo =  zeros(domain.neqs)
acce =  zeros(domain.neqs)
globdat = GlobalData(state, Dstate, velo, acce, domain.neqs, gt, ft, bt)

assembleMassMatrix!(globdat, domain)
updateTimeDependentEssentialBoundaryCondition!(domain,globdat)
updateStates!(domain, globdat)


@testset "small strain body force full" begin 
    fbody = getBodyForce(domain, globdat)
    f = eval_f_on_gauss_pts((x,y)->x^2+y^2, m, n, h)
    @test norm(fbody - compute_fem_source_term(f, f, m, n, h)) < 1e-10
end


@testset "small strain assembleMassMatrix!"
    @test maximum(abs.(compute_fem_mass_matrix(m, n, h) - globdat.M))<1e-10
end

@testset "small strain assembleStiffAndForce" begin 
    fint, stiff = assembleStiffAndForce( globdat, domain)
    @test maximum(abs.(compute_fem_stiffness_matrix(H, m, n, h)-stiff))<1e-10
    @test norm(fint - stiff * state)<1e-10
end
