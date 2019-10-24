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


nntype = "piecewise"
H0 = [1.04167e6  2.08333e5  0.0      
      2.08333e5  1.04167e6  0.0      
      0.0        0.0        4.16667e5]/stress_scale

n_data = [100, 200, 201, 202, 203]


# load true data
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

n_tot = nt*ngp*length(n_data)
σ_tot = zeros(n_tot, 3)
ε_tot = zeros(n_tot, 3)
for i = 1:nt
    for j = 1:ngp*length(n_data)
            idx = (i-1)*ngp*length(n_data) + j
            σ_tot[idx, :] = stress_seq_tot[i, j, :]
            ε_tot[idx, :] = strain_seq_tot[i, j, :]
    end
end
σ_tot = constant(σ_tot)
ε_tot = constant(ε_tot)

σ = Variable(zeros(n_tot,3))
output = nn(ε_tot[ngp*length(n_data)+1:end,:], ε_tot[1:n_tot-ngp*length(n_data),:], σ[1:n_tot-ngp*length(n_data),:])
loss1 = mean((σ[ngp*length(n_data)+1:end,:]-σ_tot[ngp*length(n_data)+1:end,:])^2)
loss2 = mean((output-σ[ngp*length(n_data)+1:end,:])^2)  
loss =  loss1  + 100*loss2

sess = Session(); init(sess)
@show run(sess, [loss1,100*loss2])
BFGS!(sess, loss, gradients(loss, σ), σ)
# opt = AdamOptimizer(10.0).minimize(loss)
# sess = Session(); init(sess)

# for i = 1:500
#     _, l = run(sess, [opt, loss])
#     @show i, l
# end


# 
# tfAssembleInternalForce(domain::Domain, nn::Function, E_all::PyObject, DE_all::PyObject, w∂E∂u_all::PyObject, σ0_all::PyObject)

