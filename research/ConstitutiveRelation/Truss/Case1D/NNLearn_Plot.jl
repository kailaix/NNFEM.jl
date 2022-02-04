using Revise
using ADCME
using NNFEM
using JLD2
using PyPlot
using PyCall
using JLD2
using MAT
using Statistics


rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
font0 = Dict(
        "font.size" => 16,
        "axes.labelsize" => 16,
        "xtick.labelsize" => 16,
        "ytick.labelsize" => 16,
        "legend.fontsize" => 16,
)
merge!(rcParams, font0)
np = pyimport("numpy")
mpl = pyimport("tikzplotlib")



reset_default_graph()
include("nnutil.jl")
t_scale, s_scale = 1000.0, 1000.0 #ms->s, MPa->GPa
markevery = 5
Nite = 10
max_idx = 4
max_nn_init_id = 10
cols = ["r", "y", "b", "g"]
nnname=nntype
if nntype == "piecewise"
    nnname = "CholNN"
elseif nntype == "ae_scaled"
    nnname = "NN1"
elseif nntype == "ae_scaled2"
    nnname = "NN2"
end

function plot_loss(nntype::String)
    close("all")
    Method = "Min"  #"Median" #"Min"
    min_nn_init_ids = zeros(Int64, max_idx)
    
    for idx = 1:max_idx
        loss_array = zeros(Float64, max_nn_init_id)
        # minLoss = 1e100
        for nn_init_id = 1:max_nn_init_id
            file="Data/$(nntype)/learned_nn$(idx)_loss_$(nn_init_id).txt"
            vars = matread(file)
            train_loss = vars["loss"]

            # if train_loss[end] < minLoss
            #     minLoss = train_loss[end]
            #     min_nn_init_ids[idx] = nn_init_id
            # end
            loss_array[nn_init_id] = train_loss[end]

            N = length(train_loss)
            NN = 1:div(N,100):N
            if nn_init_id == 1
                semilogy(NN, train_loss[NN],"--$(cols[idx])", label="$(idx)-layer $(nnname)") 
            else
                semilogy(NN, train_loss[NN],"--$(cols[idx])") 
            end
        end

        if Method == "Min"
            min_nn_init_ids[idx] = argmin(loss_array)
        elseif Method == "Median"
            min_nn_init_ids[idx] = argmin(abs.(loss_array .- median(loss_array)))
        else
            error("Method is ", Method)
        end



    end

    legend()
    xlabel("Iteration")
    ylabel("Loss")
    
    PyPlot.tight_layout()
    savefig("nnlearn_$(nntype)_loss.pdf")
    mpl.save("nnlearn_$(nntype)_loss.tex")
    @show min_nn_init_ids
    return min_nn_init_ids
end

function plot_nnlearn_p2p_strain_stress(tid::Int64, nntype::String, min_nn_init_ids::Array{Int64,1})
    close("all")
    sid = 8
    strain, stress = read_strain_stress("Data/$(tid).dat")
    X, Y = prepare_strain_stress_data1D(strain, stress )
    NT = Int64(length(Y)/sid)

    plot(X[NT*(sid-1)+1:markevery:NT*sid, 1], Y[NT*(sid-1)+1:markevery:NT*sid]/s_scale, "o",fillstyle="none", label="Reference")

    
    for idx = 1:max_idx

        reset_default_graph()
        sess = Session(); init(sess); 

        # reset config in nnutil.jl
        
        if idx == 1
            config = [20,1]
        elseif idx == 2
            config = [20,20,1]
        elseif idx == 3
            config = [20,20,20,1]
        elseif idx == 4
            config = [20,20,20,20,1]
        elseif idx == 5
            config = [20,20,20,20,20,1]
        else 
            error(idx, "idx <= 5")
        end

        function nn(ε, ε0, σ0)
            # make sure all inputs are 2d matrix
            @show ε, ε0, σ0
        
            local y
            if nntype=="linear"
                y = E0 * ε
            elseif nntype=="ae_scaled"
                x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
                y = ae(x, config, "ae_scaled")*stress_scale
            elseif nntype=="ae_scaled2"
                x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
                y = ae(x, config, "ae_scaled2")*stress_scale + σ0
            elseif nntype=="piecewise"
                x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
                H = ae(x, config, "piecewise")^2
                s = σ0^2
                i = sigmoid((s - 0.01e6)/1.0e3)  
         
                y = ( H.* i + E0 * (1-i) ) .* (ε-ε0)*stress_scale/strain_scale + σ0
        
            elseif nntype=="piecewise2"
                x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
                H = ae(x, config, "piecewise2")
                s = σ0^2
                i = sigmoid((s - 0.01e6))  
         
                y = ( H.* i + E0 * (1-i) ) .* (ε-ε0)*stress_scale/strain_scale + σ0
            else
                error("nntype must be specified.")
            end
        
            return y
        
            
        end



        y = squeeze(nn(constant(X[:,1:1]), constant(X[:,2:2]), constant(X[:,3:3])))

        ADCME.load(sess, "Data/$(nntype)/learned_nn$(idx)_ite$(Nite)_$(min_nn_init_ids[idx]).mat")
        
        out = run(sess, y)
        plot(X[NT*(sid-1)+1:NT*sid,1], out[NT*(sid-1)+1:NT*sid]/s_scale, "--$(cols[idx])", label="$(idx)-layer $(nnname)")
        
    end
    ylim((-0.1,0.6))
    xlim((-0.00125, 0.015))

    xlabel("Strain")
    ylabel("Stress (GPa)")
    legend()
    PyPlot.tight_layout()

    savefig("nnlearn_$(nntype)_p2p_strain_stress_tid$(tid).pdf")
    mpl.save("nnlearn_$(nntype)_p2p_strain_stress_tid$(tid).tex")
end


function run_nnlearn_test(tid::Int64, nntype::String, min_nn_init_ids::Array{Int64,1})

    #Data/$(nntype)/learned_nn$(idx)_ite$(i)_$(nn_init_id).mat

    for idx = 1:max_idx
        testcommand = `julia Test_NNTrussPull_Plasticity.jl $(nntype) $idx $(min_nn_init_ids[idx])` 
        @show testcommand
        run(testcommand)
    end
end


function plot_nnlearn_fem_strain_stress(tid::Int64, nntype::String)
    close("all")

    @load "Data/domain$(tid).jld2" domain 
    sid = 8
    strain = hcat(domain.history["strain"]...)
    stress = hcat(domain.history["stress"]...)
    plot(strain[sid,1:markevery:end], stress[sid,1:markevery:end]/s_scale, "o",fillstyle="none", label="Reference")


    for idx = 1:max_idx
        @load "Data/$(nntype)/nn$(idx)_domain_te$(tid).jld2" domain_te 
        
        strain = hcat(domain_te.history["strain"]...)
        stress = hcat(domain_te.history["stress"]...)
        plot(strain[sid,1:markevery:end], stress[sid,1:markevery:end]/s_scale, "--$(cols[idx])", label="$(idx)-layer $(nnname)")
    end

    xlabel("Strain")
    ylabel("Stress (GPa)")
    legend()
    ylim((-0.1,0.6))
    xlim((-0.00125, 0.015))
    PyPlot.tight_layout()

    savefig("nnlearn_$(nntype)_fem_strain_stress$tid.pdf")
    mpl.save("nnlearn_$(nntype)_fem_strain_stress$tid.tex")
end


function plot_nnlearn_fem_disp(tid::Int64, nntype::String)
    close("all")
    @load "Data/domain$(tid).jld2" domain 
    
    
    u1 = hcat(domain.history["state"]...)
    t1 = vcat([0.0], domain.history["time"]...)
    
    plot(t1[1:markevery:end]/t_scale, u1[5,1:markevery:end], "o",fillstyle="none", label="Reference")



    for idx = 1:max_idx
        @load "Data/$(nntype)/nn$(idx)_domain_te$(tid).jld2" domain_te 
        u2 = hcat(domain_te.history["state"]...)
        t2 = vcat([0.0], domain_te.history["time"]...)

        @show idx, u2[5,:]
        plot(t2[1:markevery:end]/t_scale, u2[5,1:markevery:end], "--$(cols[idx])", label="$(idx)-layer $(nnname)")
    end
    xlabel("Time (s)")
    ylabel("Displacement (m)")
    legend()
    ylim((-0.005,0.06))
    PyPlot.tight_layout()
    savefig("nnlearn_$(nntype)_fem_disp$tid.pdf")
    mpl.save("nnlearn_$(nntype)_fem_disp$tid.tex")

end

tid = 3


min_nn_init_ids = plot_loss(nntype)

plot_nnlearn_p2p_strain_stress(tid, nntype, min_nn_init_ids)
run_nnlearn_test(tid, nntype, min_nn_init_ids)
plot_nnlearn_fem_strain_stress(tid, nntype)
plot_nnlearn_fem_disp(tid, nntype)