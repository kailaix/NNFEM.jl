force_scale = 5.0 #50
tid = 200
fiber_size = 1
porder = 2
if length(ARGS) == 4
   global tid = parse(Int64, ARGS[1])
   global force_scale = parse(Float64, ARGS[2])
   global fiber_size = parse(Int64, ARGS[3])
   global porder = parse(Int64, ARGS[4])
end
printstyled("force_scale=$force_scale, tid=$tid, fiber_size=$fiber_size, porder=$porder\n", color=:green)
include("CommonFuncs.jl")
if Sys.MACHINE=="x86_64-apple-darwin18.6.0"
    matplotlib.use("macosx")
end
np = pyimport("numpy")

testtype = "PlaneStressIncompressibleRivlinSaunders"
#https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/bmk/default.htm?startat=ch01s01ach08.html
prop = Dict("name"=> testtype, "rho"=> 800.0, "C1"=>0.1863e6, "C2"=>-0.00979e6)
#prop = Dict("name"=> testtype, "rho"=> 0.800, "C1"=>0.1863e3, "C2"=>-0.00979e3)
# testtype = "PlaneStress"
# prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e+9, "nu"=> 0.45)

T = 0.2  #s
NT = 200



nxc, nyc = 10,5
nx, ny =  nxc*fiber_size, nyc*fiber_size
Lx, Ly = 0.1, 0.05 #m
#Lx, Ly = 1.0, 0.5 #m
nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid, nx, ny, porder, Lx, Ly; force_scale=force_scale)
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
setGeometryPoints!(domain, npoints, node_to_point)

state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


Δt = T/NT


ρ_oo = 0.0
αm = (2*ρ_oo - 1)/(ρ_oo + 1)
αf = ρ_oo/(ρ_oo + 1)

for i = 1:NT
    @info i, "/" , NT

    
    solver = NewmarkSolver(Δt, globdat, domain, αm, αf, 1e-4, 1e-6, 10)
end

# error()
# todo write data
if !isdir("$(@__DIR__)/Data/order$porder")
    mkdir("$(@__DIR__)/Data/order$porder")
end
if !isdir("$(@__DIR__)/Debug/order$porder")
    mkdir("$(@__DIR__)/Debug/order$porder")
end

write_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat", domain)
@save "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain

close("all")
visσ(domain)
axis("equal")
savefig("Debug/order$porder/terminal$(tid)_$(force_scale)_$(fiber_size).png")

close("all")
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[end,end] for i = 1:length(domain.history["state"])]
plot(ux)
savefig("Debug/order$porder/ux$(tid)_$(force_scale)_$(fiber_size).png")

close("all")
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[end,end] for i = 1:length(domain.history["state"])]
plot(uy)
savefig("Debug/order$porder/uy$(tid)_$(force_scale)_$(fiber_size).png")

