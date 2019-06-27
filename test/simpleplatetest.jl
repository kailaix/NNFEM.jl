# @testset "workflow" begin
    using Revise
    using Test 
    using NNFEM
    using PyPlot
    using PyCall

    testtype = "PlaneStressPlasticity"
    np = pyimport("numpy")
    nx, ny =  20, 10
    nnodes, neles = (nx + 1)*(ny + 1), nx*ny
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(x, y)
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X'[:], Y'[:]
    ndofs = 2

    EBC, g = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)


    # EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1
    # EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 2] .= -1
    # EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 1] .= -1
    # g[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 1] .= 0.04
    # gt = t->0.0

    EBC[collect(1:nx+1:(nx+1)*(ny+1)), :] .= -1
    EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 2] .= -1
    EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), 1] .= -2
    gt = t -> t*0.02*ones(sum(EBC.==-2))

    # EBC[collect(1:nx+1:(nx+1)*(ny+1)), 1] .= -2
    # EBC[collect(1:nx+1:(nx+1)*(ny+1)), 2] .= -1
    # EBC[collect(nx+1:nx+1:(nx+1)*(ny+1) + nx), :] .= -1
    # gt = t -> -t*0.04*ones(sum(EBC.==-2))

    NBC, f = zeros(Int64, nnodes, ndofs), zeros(nnodes, ndofs)
    

    prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 1000.0, "nu"=> 0.4,
                "sigmaY"=>50, "K"=>1000)

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
                        zeros(domain.neqs),∂u, domain.neqs, gt)
    assembleMassMatrix!(globdat, domain)

    updateStates!(domain, globdat)

    # F1,K = assembleStiffAndForce(globdat, domain)

    # F = assembleInternalForce(globdat, domain)

    # @info "F - F1", F - F1
    # @info "F", F
    # @info "K", K
    # @info "M", globdat.M


    # solver = ExplicitSolver(Δt, globdat, domain )
    NT = 10
    Δt = 1/NT
    for i = 1:NT
        solver = NewmarkSolver(Δt, globdat, domain, 0.5, 0.5, 1e-6, 500)
        # break
        # close("all");visstatic(domain)
        # pause(0.5)
    end
    # solver = StaticSolver(globdat, domain )
    visstatic(domain)
    
    #solver.run( props , globdat )
    # visualize(domain)
# end