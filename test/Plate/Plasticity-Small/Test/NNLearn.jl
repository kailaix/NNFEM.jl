using Revise
using ADCME
using NNFEM
using JLD2
using PyCall
using LinearAlgebra
reset_default_graph()

stress_scale = 1e5
strain_scale = 1.0
force_scale = 5.0
fiber_size = 2
porder = 2

nntype = "piecewise"

include("nnutil.jl")

#H0 = [1.04167e6  2.08333e5  0.0      
#      2.08333e5  1.04167e6  0.0      
#      0.0        0.0        4.16667e5]/stress_scale
      
H0 = [1.0406424793819175e6 209077.08366547766         0.0
      209077.08366547766   1.0411467691352057e6       0.0
      0.0                  0.0                   419057.32049008965]/stress_scale

n_data = [100, 101, 102, 103, 104, 200, 201, 202, 203, 204]

loss = constant(0.0)
for tid in n_data
    global loss
    @show "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).dat"
    # @load "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain
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
    ADCME.save(sess, "Data/$nntype/NNLearn_$(idx)_$(H_function)_ite$(i).mat")
end
