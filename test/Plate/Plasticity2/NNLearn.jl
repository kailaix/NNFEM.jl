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
fiber_size = 1
porder = 1

include("nnutil.jl")

nntype = "piecewise"
H0 = [1.04167e6  2.08333e5  0.0      
      2.08333e5  1.04167e6  0.0      
      0.0        0.0        4.16667e5]/stress_scale

n_data = [100, 200, 201, 202, 203]

loss = constant(0.0)
for tid in n_data
    global loss
    @load "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain
    X, Y = prepare_strain_stress_data2D(domain)
    x = constant(X)
    y = nn(X[:,1:3], X[:,4:6], X[:,7:9])

    loss += mean((y-Y)^2)/stress_scale^2
end


sess = Session(); init(sess)
@show run(sess, loss)
# ADCME.load(sess, "Data/order$porder/learned_nn.mat")
for i = 1:8
BFGS!(sess, loss, 15000)
ADCME.save(sess, "Data/order$porder/learned_nn_$(force_scale)_$(fiber_size).mat")
end

# error()
close("all")
tid = n_data[end]
@load "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain
X, Y = prepare_strain_stress_data2D(domain)
x = constant(X)
y = nn(X[:,1:3], X[:,4:6], X[:,7:9])

init(sess)
ADCME.load(sess, "Data/order$porder/learned_nn_$(force_scale)_$(fiber_size).mat")
O = run(sess, y)

using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 200, 200)
# ADCME.save(sess, "Data/learned_nn.mat")

error("Learning stop!")

ADCME.load(sess, "Data/order$porder/learned_nn_$(force_scale)_$(fiber_size).mat")
@show run(sess, loss)
close("all")
O = run(sess, y)
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 20)



VisualizeStrainStressSurface(X, Y)
VisualizeStrainStressSurface(X, O)
