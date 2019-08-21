# @testset "workflow" begin
    using Revise
    using Test 
    using NNFEM
    using PyCall

    testtype = "PlaneStressPlasticity"
    np = pyimport("numpy")
    nx, ny =  50, 50
    nnodes, neles = (nx + 1)*(ny + 1), nx*ny
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X'[:], Y'[:]
    ndofs = 2
    Δt = 0.1

    EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1

    NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    NBC[nx+1, :] .= -1
    f[nx+1, 1],  f[nx+1, 2] = -10.0/20, -10.0/20


    prop = Dict("name"=> testtype, "rho"=> 1.0, "E"=> 1000.0, "nu"=> 0.4,
                "sigmaY"=>1000, "K"=>1000)

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
    ∂u = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs),
                        zeros(domain.neqs),∂u, domain.neqs)
    assembleMassMatrix!(globdat, domain)

    updateStates!(domain, globdat)

    F1,K = assembleStiffAndForce(globdat, domain)

    F = assembleInternalForce(globdat, domain)

    # #@show "F - F1", F - F1
    # #@show "F", F
    # #@show "K", K
    # #@show "M", globdat.M


    # solver = ExplicitSolver(Δt, globdat, domain )
    for i = 1:10
        solver = NewmarkSolver(Δt, globdat, domain, 0.5, 0.5, 1e-8, 1e-8, 500) # ok
    end
    visdynamic(domain,"dym")
    # solver = StaticSolver(globdat, domain )
    #solver.run( props , globdat )
    # visualize(domain)
# end