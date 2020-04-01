stress_scale = 1.0e+5
strain_scale = 1

# tid = parse(Int64, ARGS[1])
force_scale = 1.0
fiber_size = 2

using PyCall
using Interpolations
mpl = pyimport("tikzplotlib")


testtype = "NeuralNetwork2D"
nntype = "stiffmat"
include("nnutil.jl")



H0 = [1.0406424793819175e6 209077.08366547766         0.0
      209077.08366547766   1.0411467691352057e6       0.0
      0.0                  0.0                   419057.32049008965]/stress_scale


T = 0.200
NT = 200


nx, ny = 10, 5
porder = 2
nxf, nyf =10*fiber_size,5*fiber_size
Lx, Ly = 0.1, 0.05
prop_dummy = Dict("name"=> "PlaneStress","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2)  #dummy

#scale  length uses cm, stress uses     MPa,     time uses  s, 
#       current is 100cm    and      10-9GPa               s  
scales = [100.0, 1.0e-6, 1.0] 



function BuildDomain(nx::Int64, ny::Int64, Lx::Float64, Ly::Float64, tid::Int64, porder::Int64, force_scale::Float64, prop::Dict{String, Any})
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
        push!(elements,FiniteStrainContinuum(coords,elnodes, prop, 3))
    end
end



domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext), ft, gt
end
# write  to a file named: testnn$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt
# the displacement "disp"
# in the file, the first line has
# Time    ux_tr    uy_tr    ux_tb    uy_tb
# Each tid, we save ux uy at right-top corner and right-bottom corner
# and the stress history "stress"

function TestNN(tid::Int64, nx::Int64, ny::Int64, prop::Dict{String, Any}, file_name::String, restart_id::Int64)
    printstyled("tid=$tid\n", color=:green)

domain, ft, gt = BuildDomain(nx, ny, Lx, Ly, tid, porder, force_scale, prop)
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
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], nx*porder+1, ny*porder+1)[end,end] for i = 1:length(domain.history["state"])]
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], nx*porder+1, ny*porder+1)[end,end] for i = 1:length(domain.history["state"])]
Disp[:,2], Disp[:,3] = ux, uy
ux = [reshape(domain.history["state"][i][1:(nx*porder+1)*(ny*porder+1)], nx*porder+1, ny*porder+1)[div((nx*porder+1),2),end] for i = 1:length(domain.history["state"])]
uy = [reshape(domain.history["state"][i][(nx*porder+1)*(ny*porder+1)+1:end], nx*porder+1, ny*porder+1)[div((nx*porder+1),2),end] for i = 1:length(domain.history["state"])]
Disp[:,4], Disp[:,5]  = ux, uy

#file = matopen("Plot/test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt", "w")
file = matopen(file_name, "w")
write(file, "state", Disp)
write(file, "full_stress_history", domain.history["stress"])
write(file, "full_state_history", domain.history["state"])
close(file)
end


function Reference(tid::Int64, nx::Int64, ny::Int64)
    full_state_history, full_fext_history = read_data("$(@__DIR__)/Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    strain, stress = read_strain_stress("Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    N = length(full_state_history)
    ts = LinRange(0, T, N)
    Disp = zeros(Float64, N, 5)
    Disp[:,1] = ts
    ux = [reshape(full_state_history[i][1:(nx*porder+1)*(ny*porder+1)], nx*porder+1, ny*porder+1)[end,end] for i = 1:N]
    uy = [reshape(full_state_history[i][(nx*porder+1)*(ny*porder+1)+1:end], nx*porder+1, ny*porder+1)[end,end] for i = 1:N]
    Disp[:,2], Disp[:,3] = ux, uy
    ux = [reshape(full_state_history[i][1:(nx*porder+1)*(ny*porder+1)], nx*porder+1, ny*porder+1)[div((nx*porder+1),2),end] for i = 1:N]
    uy = [reshape(full_state_history[i][(nx*porder+1)*(ny*porder+1)+1:end], nx*porder+1, ny*porder+1)[div((nx*porder+1),2),end] for i = 1:N]
    Disp[:,4], Disp[:,5]  = ux, uy
    
    file = matopen("Plot/reference$(tid).mat", "w")
    write(file, "state", Disp)
    write(file, "full_stress_history", stress)
    write(file, "full_state_history", full_state_history)
    close(file)
end


function Plot(tid::Int64, restart_id::Int64)
    close("all")
    L_scale, t_scale = scales[1], scales[3]
    markevery = 5
    # Reference
    vars = matread("Plot/reference$(tid).mat")
    disp_ref = vars["state"] 
    plot(disp_ref[1:markevery:end, 1]*t_scale, disp_ref[1:markevery:end, 2]*L_scale, "or", fillstyle="none", label="Reference")
    plot(disp_ref[1:markevery:end, 1]*t_scale, disp_ref[1:markevery:end, 3]*L_scale, "oy", fillstyle="none")
    plot(disp_ref[1:markevery:end, 1]*t_scale, disp_ref[1:markevery:end, 4]*L_scale, "ob", fillstyle="none")
    plot(disp_ref[1:markevery:end, 1]*t_scale, disp_ref[1:markevery:end, 5]*L_scale, "og", fillstyle="none")
   
    # NNTrain
    vars = matread("Plot/test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt")
    disp_test = vars["state"]
    plot(disp_test[:, 1]*t_scale, disp_test[:, 2]*L_scale, "r",  label="Indirect")
    plot(disp_test[:, 1]*t_scale, disp_test[:, 3]*L_scale, "y")
    plot(disp_test[:, 1]*t_scale, disp_test[:, 4]*L_scale, "b")
    plot(disp_test[:, 1]*t_scale, disp_test[:, 5]*L_scale, "g")


    # NNLearn
    vars = matread("Plot/test_nnlearn$(idx)_$(nntype)_test$(tid).txt")
    #use fint, strain, stress, for debugging purpose
    disp_test = vars["state"]
    plot(disp_test[:, 1]*t_scale, disp_test[:, 2]*L_scale, "--r",  label="Direct")
    plot(disp_test[:, 1]*t_scale, disp_test[:, 3]*L_scale, "--y")
    plot(disp_test[:, 1]*t_scale, disp_test[:, 4]*L_scale, "--b")
    plot(disp_test[:, 1]*t_scale, disp_test[:, 5]*L_scale, "--g")

    xlabel("Time (s)")
    ylabel("Displacement (cm)")

    PyPlot.tight_layout()
	
    savefig("Plot/plate_hyperelasticity_disp_nn$(idx)_$(nntype)_from$(restart_id)_test$(tid).pdf")
    mpl.save("Plot/plate_hyperelasticity_disp_nn$(idx)_$(nntype)_from$(restart_id)_test$(tid).tex")



    ########################################################### Plot diff
    close("all")
    L_scale, t_scale = scales[1], scales[3]
    markevery = 5
    # Reference
    vars = matread("Plot/reference$(tid).mat")
    disp_ref = vars["state"] 

    # NNTrain
    vars = matread("Plot/test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt")
    disp_test = vars["state"]
    ref = Array{Array{Float64}}(undef,4)
    for i = 1:4
        ref[i] = LinearInterpolation(disp_ref[1:markevery:end, 1]*t_scale, disp_ref[1:markevery:end, i+1]*L_scale)(disp_test[:, 1]*t_scale)
    end
    semilogy(disp_test[:, 1]*t_scale, abs.(ref[1]-disp_test[:, 2]*L_scale), "r",  label="Indirect data training")
    semilogy(disp_test[:, 1]*t_scale, abs.(ref[2]-disp_test[:, 3]*L_scale), "y")
    semilogy(disp_test[:, 1]*t_scale, abs.(ref[3]-disp_test[:, 4]*L_scale), "b")
    semilogy(disp_test[:, 1]*t_scale, abs.(ref[4]-disp_test[:, 5]*L_scale), "g")


    # NNLearn
    vars = matread("Plot/test_nnlearn$(idx)_$(nntype)_test$(tid).txt")
    #use fint, strain, stress, for debugging purpose
    disp_test = vars["state"]
    ref = Array{Array{Float64}}(undef,4)
    for i = 1:4
        ref[i] = LinearInterpolation(disp_ref[1:markevery:end, 1]*t_scale, disp_ref[1:markevery:end, i+1]*L_scale)(disp_test[:, 1]*t_scale)
    end
    semilogy(disp_test[:, 1]*t_scale, abs.(ref[1]-disp_test[:, 2]*L_scale), "r", linestyle="dotted",  label="Direct data fitting")
    semilogy(disp_test[:, 1]*t_scale, abs.(ref[2]-disp_test[:, 3]*L_scale), "y", linestyle="dotted")
    semilogy(disp_test[:, 1]*t_scale, abs.(ref[3]-disp_test[:, 4]*L_scale), "b", linestyle="dotted")
    semilogy(disp_test[:, 1]*t_scale, abs.(ref[4]-disp_test[:, 5]*L_scale), "g", linestyle="dotted")

    xlabel("Time (s)")
    ylabel("Displacement (cm)")
    legend()

    PyPlot.tight_layout()
	
	
    savefig("Plot/plate_hyperelasticity_disp_nn$(idx)_$(nntype)_from$(restart_id)_test$(tid)_error.pdf")
    mpl.save("Plot/plate_hyperelasticity_disp_nn$(idx)_$(nntype)_from$(restart_id)_test$(tid)_error.tex")
    #savefig("Plot/plate_hyperelasticity_disp_nn$(idx)_$(nntype)_from$(restart_id)_test$(tid).pdf")
    #mpl.save("Plot/diff$(tid).tex")

end


function PlotStress(tid::Int64, nx::Int64, ny::Int64, nxf::Int64, nyf::Int64, fiber_size::Int64, porder::Int64, 
    Lx::Float64, Ly::Float64, force_scale::Float64, restart_id::Int64)
    
    #visualize exact solution 
    close("all")
    domain, _, _ = BuildDomain(nxf, nyf, Lx, Ly, tid, porder, force_scale, prop_dummy)
    vars = matread("Plot/reference$(tid).mat")
    ts, stress, full_state_history = vars["state"][:,1], vars["full_stress_history"], vars["full_state_history"]
    # stress size is NT
    # full_state_history size is NT+1
    frame = Int64(NT/2)
    vmin, vmax = visσ(domain, nxf, nyf,  stress[frame], full_state_history[frame+1]; scaling = scales)
    xlabel("X (cm)")
    ylabel("Y (cm)")
    PyPlot.tight_layout()
    savefig("Plot/plate_hyperelasticity_stress_reference$(tid).pdf")


    #visualize nntrain solution
    close("all")
    domain,_,_ = BuildDomain(nx, ny, Lx, Ly, tid, porder, force_scale, prop_dummy)
    vars = matread("Plot/test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt")
    ts, stress, full_state_history = vars["state"][:,1], vars["full_stress_history"], vars["full_state_history"]
    # stress size is NT
    # full_state_history size is NT+1
    frame = 0
    for iframe = 1 : length(ts)
        if ts[iframe] > T/2.0
            frame = ((ts[iframe] - T/2.0) > (T/2.0 - ts[iframe - 1]) ? iframe-1  : iframe) 
            break
        end
    end
    @show "frame is ", frame, " ,time is ", ts[frame]

    visσ(domain, nx, ny,  stress[frame-1], full_state_history[frame], vmin, vmax; scaling = scales)
    xlabel("X (cm)")
    ylabel("Y (cm)")
    PyPlot.tight_layout()
    savefig("Plot/plate_hyperelasticity_stress_test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).pdf")



    #visualize nnlearn solution
    close("all")
    domain,_,_ = BuildDomain(nx, ny, Lx, Ly, tid, porder, force_scale, prop_dummy)
    vars = matread("Plot/test_nnlearn$(idx)_$(nntype)_test$(tid).txt")
    ts, stress, full_state_history = vars["state"][:,1], vars["full_stress_history"], vars["full_state_history"]
    # stress size is NT
    # full_state_history size is NT+1
    frame = 0
    for iframe = 1 : length(ts)
        if ts[iframe] > T/2.0
            frame = ((ts[iframe] - T/2.0) > (T/2.0 - ts[iframe - 1]) ? iframe-1  : iframe) 
            break
        end
    end
    @show "frame is ", frame, " ,time is ", ts[frame]

    visσ(domain, nx, ny,  stress[frame-1], full_state_history[frame], vmin, vmax; scaling = scales)
    xlabel("X (cm)")
    ylabel("Y (cm)")
    PyPlot.tight_layout()
    savefig("Plot/plate_hyperelasticity_stress_test_nnlearn$(idx)_$(nntype)_test$(tid).pdf")



end







#####################################################
GENERATE_DATA = false
PLOT = true
restart_id_list = [3]
tid_list = [106, 206, 300]
train_id = 50


if GENERATE_DATA
    if !isdir("Plot"); mkdir("Plot"); end
for restart_id in restart_id_list
    # Reference
    for tid in tid_list
        Reference(tid, nxf, nyf)
    end
    

    # NNTrain
    s = ae_to_code("Data/$(nntype)/NN_Train_$(idx)_from_$(restart_id)_ite$(train_id).mat", nntype)
    eval(Meta.parse(s))
    prop = Dict("name"=> testtype, "rho"=> 0.8, "nn"=>post_nn)
    for tid in tid_list
        file_name = "Plot/test_nntrain$(idx)_$(nntype)_from$(restart_id)_test$(tid).txt"
        TestNN(tid, nx, ny, prop, file_name, restart_id)
    end

    # NNLearn 
    s = ae_to_code("Data/$(nntype)/NNLearn_$(idx)_$(H_function)_ite$(train_id).mat", nntype)
    eval(Meta.parse(s))
    prop = Dict("name"=> testtype, "rho"=> 0.8, "nn"=>post_nn)
    for tid in tid_list
        file_name = "Plot/test_nnlearn$(idx)_$(nntype)_test$(tid).txt"
        TestNN(tid, nx, ny, prop, file_name, restart_id)
    end


end
end

if PLOT
for tid in tid_list
    for restart_id in restart_id_list
        Plot(tid, restart_id)
        PlotStress(tid, nx, ny, nxf, nyf, fiber_size, porder, Lx, Ly, force_scale, restart_id)
    end
end
end



