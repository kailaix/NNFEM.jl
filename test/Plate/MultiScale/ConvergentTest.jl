# tid = parse(Int64, ARGS[1])
tid = 203
printstyled("tid=$tid\n", color=:green)

include("CommonFuncs.jl")
np = pyimport("numpy")

prop = Dict("name"=> "PlaneStress", "rho"=> 3.2, "E"=>4e6, "nu"=>0.35)

T = 0.05
NT = 100
force_scale = 50
REFINE_SCALES = [1]#[1,2,4,8]
porder = 2
nxc, nyc = 10,5

for refine_scale in REFINE_SCALES

    nx, ny =  nxc*refine_scale, nyc*refine_scale



nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, porder)
elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx*porder+1)*(j-1)*porder + (i-1)porder+1
        #element (i,j)
        if porder == 1
            #   4 ---- 3
            #
            #   1 ---- 2

            elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        elseif porder == 2
            #   4 --7-- 3
            #   8   9   6 
            #   1 --5-- 2
            elnodes = [n, n + 2, n + 2 + 2*(2*nx+1),  n + 2*(2*nx+1), n+1, n + 2 + (2*nx+1), n + 1 + 2*(2*nx+1), n + (2*nx+1), n+1+(2*nx+1)]
        else
            error("polynomial order error, porder= ", porder)
        end

        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end


ndofs = 2
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


Δt = T/NT
stress_scale = 1
strain_scale = 1

ρ_oo = 0.0
αm = (2*ρ_oo - 1)/(ρ_oo + 1)
αf = ρ_oo/(ρ_oo + 1)

for i = 1:NT
    @info i, "/" , NT
    solver = NewmarkSolver(Δt, globdat, domain, αm, αf, 1e-4, 1e-6, 10)
end

@save "Data/ConvergentTest_O$porder/domain_$refine_scale.jld2" domain
end


close("all")
for refine_scale in REFINE_SCALES
    nx, ny =  nxc*refine_scale, nyc*refine_scale
    @load "Data/ConvergentTest_O$porder/domain_$refine_scale.jld2" domain 
    uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["state"])]
    ay = [reshape(convertState(domain, domain.history["acc"][i],"Expand")[(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["acc"])]
    plot(ay)

end
savefig("Debug/ConvergentTest_uy_O$porder.png")
close("all")
for refine_scale in REFINE_SCALES
    @show refine_scale
    nx, ny =  nxc*refine_scale, nyc*refine_scale
    @load "Data/ConvergentTest_O$porder/domain_$refine_scale.jld2" domain
    ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["state"])]
    ax = [reshape(convertState(domain, domain.history["acc"][i],"Expand")[1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["acc"])]
    plot(ax)

end
savefig("Debug/ConvergentTest_ux_O$porder.png")