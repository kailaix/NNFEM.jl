# tid = parse(Int64, ARGS[1])
force_scale = 50.0
tid = 203  
# if Sys.MACHINE=="x86_64-pc-linux-gnu"
#    global tid = parse(Int64, ARGS[1])
#    global force_scale = parse(Float64, ARGS[2])
# end
printstyled("tid=$tid\n", color=:green)

include("CommonFuncs.jl")

testtype = "NeuralNetwork2D"
nntype = "linear"
include("nnutil.jl")

H0 = [4.5584e6   1.59544e6  0.0      
1.59544e6  4.5584e6   0.0      
0.0        0.0        1.48148e6]

# density 4.5*(1 - 0.25) + 3.2*0.25
#fiber_fraction = 0.25
#todo
fiber_fraction = 1.0
prop = Dict("name"=> testtype, "rho"=> 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction, "nn"=>post_nn)

T = 0.05
NT = 100


nx_f, ny_f = 10*2, 5*2
# nx_f, ny_f = 12, 4
# homogenized computaional domain
# number of elements in each directions
nx, ny = 20, 10

porder = 2

nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny,porder)

ndofs=2
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



domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs)
, zeros(domain.neqs),∂u, domain.neqs, gt, ft)

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
    # close("all")
    # visσ(domain,-1.5e9, 4.5e9)
    # savefig("Debug/$i.png")
    # error()
    if i==75
        close("all")
        visσ(domain)
        # visσ(domain,-1.5e9, 4.5e9)
        savefig("Debug/test$(tid)i=75.png")
    end
end

# plot
close("all")
visσ(domain)
savefig("Debug/test$tid.png")

close("all")
visσ(domain)
axis("equal")
savefig("Debug/order$porder/test_stress$(tid)_$force_scale.png")

close("all")
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["state"])]
plot(ux)
savefig("Debug/order$porder/test_ux$(tid)_$force_scale.png")

close("all")
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["state"])]
plot(uy)
savefig("Debug/order$porder/test_uy$(tid)_$force_scale.png")

