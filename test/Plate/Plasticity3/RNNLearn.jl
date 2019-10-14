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

include("nnutil.jl")



######################################################################
function compute_sequence_loss(strain_seq, stress_seq, nn)
    loss = constant(0.0)
    nt = size(strain_seq, 1)
    ngp = size(strain_seq, 2)

    y = stress_seq[1,:,:]
    for it = 2:nt
        xs, xs_old, ys = strain_seq[it,:,:], strain_seq[it-1,:,:], stress_seq[it,:,:]
        
        y = nn(constant(xs), constant(xs_old), constant(y))
        
        #todo change the loss function, mean or sum
        loss += mean((ys - y)^2)
    end
    return loss
end
#####################################################################################


nntype = "piecewise"
H0 = [1.04167e6  2.08333e5  0.0      
      2.08333e5  1.04167e6  0.0      
      0.0        0.0        4.16667e5]/stress_scale

n_data = [100, 200, 201, 202, 203]

@load "Data/order$porder/domain$(n_data[1])_$(force_scale)_$(fiber_size).jld2" domain
strain_seq, stress_seq = prepare_sequence_strain_stress_data2D(domain)
nt = size(strain_seq, 1)
ngp = size(strain_seq, 2)

strain_seq_tot, stress_seq_tot = zeros(nt, ngp*length(n_data), 3), zeros(nt, ngp*length(n_data), 3)

for (i, tid) in enumerate(n_data)
    @load "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain
    @show size(strain_seq_tot), i
    strain_seq_tot[:,(i-1)*ngp+1:i*ngp,:], stress_seq_tot[:,(i-1)*ngp+1:i*ngp,:] = prepare_sequence_strain_stress_data2D(domain)
end

loss = compute_sequence_loss(strain_seq_tot, stress_seq_tot, nn) #/stress_scale^2


sess = Session(); init(sess)
@show run(sess, loss)
# ADCME.load(sess, "Data/order$porder/learned_nn.mat")
for i = 1:10
    BFGS!(sess, loss, 1000)
    ADCME.save(sess, "Data/order$porder/learned_rnn_$(force_scale)_$(fiber_size).mat")
end

error()
close("all")
tid = n_data[end]
@load "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain
X, Y = prepare_strain_stress_data2D(domain)
x = constant(X)
y = nn(X[:,1:3], X[:,4:6], X[:,7:9])

init(sess)
ADCME.load(sess, "Data/order$porder/learned_nn_$(force_scale)_$(fiber_size).mat")
ADCME.load(sess, "Data/nn_train0.mat")
O = run(sess, y)
using Random; Random.seed!(233)
VisualizeStress2D(Y, O, 200, 200)

error("Learning stop!")

ADCME.load(sess, "Data/order$porder/learned_nn_$(force_scale)_$(fiber_size).mat")

@show run(sess, loss)
close("all")
O = run(sess, y)
using Random; Random.seed!(233)
close("all")
VisualizeStress2D(Y, O, 20)
savefig("test.png")



VisualizeStrainStressSurface(X, Y)
VisualizeStrainStressSurface(X, O)
