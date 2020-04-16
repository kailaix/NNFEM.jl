using Revise
using Test 
using NNFEM
using PyCall
using PyPlot
using JLD2
using ADCME
using LinearAlgebra
reset_default_graph()



include("nnutil.jl")

testtype = "NeuralNetwork1D"
include("NNTrussPull_Domain.jl")


#use only rho
prop = Dict("name"=> testtype, "rho"=> 8000.0, "E"=> 200e3, "nu"=> 0.45,
           "sigmaY"=>0.3e3, "K"=>1/9*200e3, "B"=> 0.0, "A0"=> 0.005, "nn"=>post_nn)


n_data = [1,2, 4,5]

function approximate_stress(tid::Int64, method::String)
    nodes, EBC, g, gt, FBC, fext, ft, npoints, node_to_point = BoundaryCondition(tid)
    #domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)

    @load "Data/domain$(tid).jld2" domain 
    setGeometryPoints!(domain, npoints, node_to_point)
    state = zeros(domain.neqs)
    ∂u = zeros(domain.neqs)
    globdat = GlobalData(state,zeros(domain.neqs), zeros(domain.neqs),∂u, domain.neqs, gt, ft)
    assembleMassMatrix!(globdat, domain)

    state_history, full_fext_history = read_data("$(@__DIR__)/Data/$(tid).dat")

    #update state history and fext_history on the homogenized domain
    fext_history = []
    setConstantNodalForces!(domain, FBC, fext)
    for i = 1:NT
        globdat.time = Δt*i
        updateDomainStateBoundary!(domain, globdat)
        fext = getExternalForce!(domain, globaldat)
        push!(fext_history, fext)
    end

    E_all, S_all = LSfittingStress(domain, globdat, hcat(state_history...), hcat(fext_history...), Δt, method)

end



nodes, _, _, _, _, _, _, _, _ = BoundaryCondition(tid)
elements = []
for i = 1:nx 
    elnodes = [i, i+1]; coords = nodes[elnodes,:];
    push!(elements, FiniteStrainTruss(coords,elnodes, prop, ngp))
end





loss = constant(0.0)

method = "Constant"
for tid in n_data
    E_all, S_all = approximate_stress(tid, method)
    X = zeros(Float64, (NT-1)*ngp, 3)
    Y = zeros(Float64, (NT-1)*ngp)
    k = 1
    for it = 2:NT
        for igp = 1:ngp
            X[k,:] = [E_all[it,igp]; E_all[it-1,igp]; S_all[it-1,igp]]#ε, ε0, σ0
            Y[k] = S_all[it,igp]
            k = k + 1
        end
    end



    y = squeeze(nn(constant(X[:,1:1]), constant(X[:,2:2]), constant(X[:,3:3])))
    
    global loss
    loss += sum((y - Y)^2) #/stress_scale^2
end

sess = Session(); init(sess)
@show run(sess, loss)
# ADCME.load(sess, "Data/$(nntype)/NNPreLSfit.mat")
if !isdir("Data/$(nntype)")
    mkdir("Data/$(nntype)")
end

for i = 1:5
    BFGS!(sess, loss, 1000)
    ADCME.save(sess, "Data/$(nntype)/NNPreLSfit_nn$(idx)_ite$(i).mat")
end


tid = 3
strain, stress = read_strain_stress("Data/$(tid).dat")
X, Y = prepare_strain_stress_data1D(strain, stress )
y = squeeze(nn(constant(X[:,1:1]), constant(X[:,2:2]), constant(X[:,3:3])))
out = run(sess, y)
plot(X[:,1], out,"+", label="NN")
plot(X[:,1], Y, ".", label="Exact")
#legend()
savefig("nnpreLSfit_$(nntype)_truss1d_stress$tid.png")
