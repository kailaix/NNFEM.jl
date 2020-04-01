using Revise
using ADCME
using NNFEM
using JLD2
using PyCall
using LinearAlgebra
reset_default_graph()

stress_scale = 1e5
strain_scale = 1.0
force_scale = 1.0
fiber_size = 2
porder = 2

nntype = "stiffmat"
#nntype = "ae_scaled"

include("nnutil.jl")

H0 = [1.04167e6  2.08333e5  0.0      
      2.08333e5  1.04167e6  0.0      
      0.0        0.0        4.16667e5]/stress_scale

n_data = [100, 101, 102, 103, 104, 200, 201, 202, 203, 204]

loss = constant(0.0)
for tid in n_data
    global loss
    strain, stress = read_strain_stress("Data/order$porder/$(tid)_$(force_scale)_$(fiber_size).dat")
    X, Y = prepare_strain_stress_data2D(strain, stress)
    x = constant(X)
    y = nn(x[:,1:3], x[:,4:6], x[:,7:9])
    @show "tid is ", tid
    loss += mean((y-Y)^2) #/stress_scale^2
end


sess = Session(); init(sess)
@show run(sess, loss)
# ADCME.load(sess, "Data/NNLearn.mat")
for i = 1:50
    BFGS!(sess, loss, 1000)
    ADCME.save(sess, "Data/$nntype/NNLearn_$(idx)_ite$(i).mat")
end
