@testset "workflow" begin
    using Revise
    using Test 
    using NNFEM
    using PyCall

    testtype = "Plasticity"
    np = pyimport("numpy")
    nx, ny =  2, 2
    nnodes, neles = (nx + 1)*(ny + 1), nx*ny
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X'[:], Y'[:]
    ndofs = 2
    Δt = 0.01

    EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1

    NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    NBC[nx+1, :] .= -1
    f[nx+1, 1],  f[nx+1, 2] = -10, -20


    prop = Dict("name"=> testtype, "rho"=> 1.0, "E"=> 1000.0, "nu"=> 0.4,
                "sigmaY"=>1000, "K"=>100)

    elements = []
    for j = 1:ny
        for i = 1:nx 
            n = (nx+1)*(j-1) + i
            elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
            coords = nodes[elnodes,:]
            push!(elements,SmallStrainContinuum(coords,elnodes, prop))
        end
    end


    domain = Domain(nodes, elements, ndofs, EBC, g, NBC, f)
    state = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs),
                        zeros(domain.neqs),zeros(domain.neqs), domain.neqs)
    assembleMassMatrix!(globdat, domain)

    updateStates(domain, globdat)

    F1,K = assembleStiffAndForce(globdat, domain)

    F = assembleInternalForce(globdat, domain)

    @info "F - F1", F - F1
    @info "F", F
    @info "K", K
    @info "M", globdat.M


    # solver = ExplicitSolver(Δt, globdat, domain )
    #solver = NewmarkSolver(Δt, globdat, domain )
    solver = StaticSolver(globdat, domain )
    #solver.run( props , globdat )
    # visualize(domain)
end