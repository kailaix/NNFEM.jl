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
    @info "X", X, "Y", Y
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X'[:], Y'[:]
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
    state = [1.0;2.0;3.0;4.0;5.0;6.0;7.0;8.0;9.0;10.0;11.0;12.0]
    globdat = GlobalData(state,zeros(domain.neqs),
                        zeros(domain.neqs),zeros(domain.neqs), domain.neqs)
    assembleMassMatrix!(globdat, domain)

    updateStates(domain, globdat.state, globdat.Dstate, globdat.time)

    F = assembleInternalForce(globdat, domain)
    @info "F" F
    solver = ExplicitSolver(Δt, globdat, domain )
    #solver.run( props , globdat )
end