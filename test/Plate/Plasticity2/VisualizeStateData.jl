stress_scale = 1.0e5
strain_scale = 1

include("nnutil.jl")

# H0 = constant(H1/stress_scale)
testtype = "NeuralNetwork2D"
# force_scale = 5.0
force_scales = [5.0]
nntype = "piecewise"

# ! define H0
# Trained with nx, ny = 10, 5
H0 = [1.04167e6  2.08333e5  0.0      
      2.08333e5  1.04167e6  0.0      
      0.0        0.0        4.16667e5]/stress_scale


n_data = [100, 200, 201, 202, 203]

# density 4.5*(1 - 0.25) + 3.2*0.25
#fiber_fraction = 0.25
#todo
porder = 2
prop = Dict("name"=> testtype, "rho"=> 4.5, "nn"=>nn)


T = 0.05
NT = 100

# DNS computaional domain
fiber_size = 2
# nx_f, ny_f = 40*fiber_size, 20*fiber_size
nx_f, ny_f = 10*fiber_size, 5*fiber_size

# nx_f, ny_f = 12, 4

# homogenized computaional domain
# number of elements in each directions
nx, ny = 10, 5
# number of subelements in one element in each directions
sx_f, sy_f = div(nx_f,nx), div(ny_f,ny)

ndofs = 2
fine_to_coarse = zeros(Int64, ndofs*(nx*porder+1)*(ny*porder+1))
for idof = 1:ndofs
    for iy = 1:ny*porder+1
        for ix = 1:nx*porder+1
            fine_to_coarse[ix + (iy - 1)*(nx*porder+1) + (idof-1)*(nx*porder+1)*(ny*porder+1)] = 
            1 + (ix - 1) * sx_f + (iy - 1) * (nx_f*porder + 1) * sy_f + (nx_f*porder + 1)*(ny_f*porder + 1)*(idof - 1)
        end
    end
end


Δt = T/NT
nodes, _, _, _, _, _, _ = BoundaryCondition(n_data[1], nx, ny, porder)
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

force_scales = 5.0
tid = 200
nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, porder, force_scale )
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
assembleMassMatrix!(globdat, domain)
# full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_2.dat")

#update state history and fext_history on the homogenized domain
state_history = [x[fine_to_coarse] for x in full_state_history]

visstate(domain, state_history[end])
savefig("test.png")