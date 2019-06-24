@testset "workflow" begin
using Revise
using Test 
using NNFEM
using PyCall
np = pyimport("numpy")
    nx, ny =  2, 2
    nnodes, neles = (nx + 1)*(ny + 1), nx*ny
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X[:], Y[:]
    ndofs = 2
    Δt = 0.01

    EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1


    prop = Dict("name"=> "PlaneStrain", "rho"=> 1.0, "E"=> 1000.0, "nu"=> 0.4)

    elements = []
    for j = 1:ny
        for i = 1:nx 
            n = (nx+1)*(j-1) + i
            elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
            coords = nodes[elnodes,:]
            push!(elements,FiniteStrainContinuum(coords,elnodes, prop))
        end
    end


    domain = Domain(nodes, elements, ndofs, EBC, g)
    globdat = GlobalData(zeros(domain.neqs),zeros(domain.neqs),
                        zeros(domain.neqs),zeros(domain.neqs), domain.neqs)
    assembleMassMatrix!(globdat, domain)

    F = assembleInternalForce(globdat, domain)
    solver = ExplicitSolver(Δt, globdat, domain )
    #solver.run( props , globdat )
end