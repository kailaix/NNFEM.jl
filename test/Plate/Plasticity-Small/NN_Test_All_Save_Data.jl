stress_scale = 1.0e+5
strain_scale = 1

# tid = parse(Int64, ARGS[1])
force_scale = 5.0
fiber_size = 2



testtype = "NeuralNetwork2D"
nntype = "piecewise"
include("nnutil.jl")



H0 = [1.0406424793819175e6 209077.08366547766         0.0
      209077.08366547766   1.0411467691352057e6       0.0
      0.0                  0.0                   419057.32049008965]/stress_scale


T = 200.0
NT = 200


nx, ny = 10, 5
porder = 2
Lx, Ly = 1.0, 0.5


# write  to a file named: testnn$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt
# the displacement "disp"
# in the file, the first line has
# Time    ux_tr    uy_tr    ux_tb    uy_tb
# Each tid, we save ux uy at right-top corner and right-bottom corner
# and the stress history "stress"

function TestNN(tid::Int64, nx::Int64, ny::Int64, prop::Dict{String, Any}, restart_id::Int64)
    printstyled("tid=$tid\n", color=:green)

nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny,porder, Lx, Ly; force_scale=force_scale)

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
globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)

assembleMassMatrix!(globdat, domain)
updateStates!(domain, globdat)


ts = LinRange(0, T, NT+1)
adaptive_solver_args = Dict("Newmark_rho"=> 0.0, 
                          "Newton_maxiter"=>10, 
                          "Newton_Abs_Err"=>1e-4, 
                          "Newton_Rel_Err"=>1e-6, 
                          "damped_Newton_eta" => 1.0)

globdat, domain, ts = AdaptiveSolver("NewmarkSolver", globdat, domain, T, NT, adaptive_solver_args)


Disp = zeros(Float64, length(ts), 5)
Disp[:,1] = ts
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[end,end] for i = 1:length(domain.history["state"])]
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[end,end] for i = 1:length(domain.history["state"])]
Disp[:,2], Disp[:,3] = ux, uy
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["state"])]
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[1,end] for i = 1:length(domain.history["state"])]
Disp[:,4], Disp[:,5]  = ux, uy

file = matopen("Plot/test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt", "w")
write(file, "Disp", Disp)
write(file, "stress", domain.history["stress"])
close(file)
end


function Reference(tid::Int64, nx::Int64, ny::Int64)
    
  
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    strain, stress = read_strain_stress("Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    N = length(full_state_history)
    ts = LinRange(0, T, N)
    Disp = zeros(Float64, N, 5)
    Disp[:,1] = ts
    ux = [reshape(full_state_history[i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[end,end] for i = 1:N]
    uy = [reshape(full_state_history[i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[end,end] for i = 1:N]
    Disp[:,2], Disp[:,3] = ux, uy
    ux = [reshape(full_state_history[i][1:(nx*porder+1)*(ny*porder+1)], ny*porder+1, nx*porder+1)[1,end] for i = 1:N]
    uy = [reshape(full_state_history[i][(nx*porder+1)*(ny*porder+1)+1:end], ny*porder+1, nx*porder+1)[1,end] for i = 1:N]
    Disp[:,4], Disp[:,5]  = ux, uy
    
    file = matopen("Plot/reference$(tid).txt", "w")
    write(file, "Disp", Disp)
    write(file, "stress", stress)
    close(file)
end


function Plot(tid::Int64, restart_id::Int64)
    close("all")

    vars = matread("Plot/reference$(tid).txt")
    #use fint, strain, stress, for debugging purpose
    disp_ref = vars["Disp"]

    plot(disp_ref[:, 1], disp_ref[:, 2], "--", label="u (ref)")
    plot(disp_ref[:, 1], disp_ref[:, 3], "--", label="v (ref)")
    plot(disp_ref[:, 1], disp_ref[:, 4], "--", label="u (ref)")
    plot(disp_ref[:, 1], disp_ref[:, 5], "--", label="v (ref)")
   

    vars = matread("Plot/test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt")
    #use fint, strain, stress, for debugging purpose
    disp_test = vars["Disp"]
    plot(disp_test[:, 1], disp_test[:, 2], label="u (pred)")
    plot(disp_test[:, 1], disp_test[:, 3], label="v (pred)")
    plot(disp_test[:, 1], disp_test[:, 4], label="u (pred)")
    plot(disp_test[:, 1], disp_test[:, 5], label="v (pred)")

    savefig("Plot/test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).png")

end


function PlotStress(tid::Int64, nx::Int64, ny::Int64, fiber_size::Int64, porder::Int64, 
    Lx::Float64, Ly::Float64, force_scale::Float64, restart_id::Int64)
    nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny,porder, Lx, Ly)
    prop = Dict("name"=> "PlaneStress","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2)  #dummy
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

    close("all")
    vars = matread("Plot/reference$(tid).txt")
    ts = vars["Disp"][:,1]
    strain, stress = read_strain_stress("Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    # stress size is NT
    # full_state_history size is NT+1

    frame = 100
    visσ(domain, nx, ny,  stress[frame], full_state_history[frame+1]; scaling = 1.0)

    savefig("test.png")

end

PlotStress(300, 20, 10, 2, 2, 
    1.0, 0.5, 5.0, 3)


#=
for restart_id in [2,3,4,5]

train_id = 50
#s = ae_to_code("Data/$(nntype)/NNLearn_$(idx)_$(H_function)_ite50.mat", nntype)
#s = ae_to_code("Data/$(nntype)/NNPreLSfit_$(idx)_$(H_function)_2.mat", nntype)
s = ae_to_code("Data/$(nntype)/NN_Train_$(idx)_from_$(restart_id)_$(H_function)_ite$(train_id).mat", nntype)
eval(Meta.parse(s))
prop = Dict("name"=> testtype, "rho"=> 4.5, "nn"=>post_nn)

for tid in [105, 106,205, 206, 300]
    
    nx,ny = 10, 5  
    TestNN(tid, nx, ny, prop, restart_id)
    nx, ny =10*fiber_size,5*fiber_size
    Reference(tid, nx, ny, restart_id)
    Plot(tid)
end
=#