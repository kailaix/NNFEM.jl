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


H0 = [1.04167e6  2.08333e5  0.0      
      2.08333e5  1.04167e6  0.0      
      0.0        0.0        4.16667e5]/stress_scale

# function nn(ε, ε0, σ0) # ε, ε0, σ0 450x3
#     global H0
    
#     threshold = 1e7 # σY ≈ 1e8
#     config = [20, 20, 20, 4]

#     x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
#     x = constant(x)
#     ε = constant(ε)
#     ε0 = constant(ε0)
#     σ0 = constant(σ0)
    
#     y = ae(x, config, nntype)
    
#     z = spd_Chol_Orth(y)

#     σnn = squeeze(tf.matmul(z, tf.reshape((ε-ε0)/strain_scale, (-1,3,1)))) 
#     σH = (ε-ε0)/strain_scale * H0
#     s = σ0[:,1]^2-σ0[:,1]*σ0[:,2]+σ0[:,2]^2+3*σ0[:,3]^2 
    
#     i = sigmoid(1000*(s-threshold)/1e9)        
#     i = [i i i]
#     out = σnn .* i + σH .* (1-i)  + σ0/stress_scale
#     out*stress_scale
# end


function nn(ε, ε0, σ0) # ε, ε0, σ0 450x3
    global H0
    
    threshold = 1e7 # σY ≈ 1e8
    config = [20, 20, 20, 4]

    x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
    x = constant(x)
    ε = constant(ε)
    ε0 = constant(ε0)
    σ0 = constant(σ0)
    
    y = ae(x, config, nntype)
    
    z = spd_Chol_Orth(y)

    σnn = squeeze(tf.matmul(z, tf.reshape((ε-ε0)/strain_scale, (-1,3,1)))) 
    
    out = σnn  + σ0/stress_scale
    out*stress_scale
end


tid = 200
# @load "Data/order$porder/domain$(tid)_$(force_scale)_$(fiber_size).jld2" domain
# X, _ = prepare_strain_stress_data2D(domain)
X = rand(10,9)
x = constant(X)
y = nn(x[:,1:3], x[:,4:6], x[:,7:9])


sess = Session(); init(sess)
ADCME.save(sess, "Data/NNLearn.mat")
out1 = run(sess, y)



config = [9, 20, 20, 20, 4]
theta = convert_mat("nn2array", config,  "Data/NNLearn.mat")
out2, _, _ =  constitutive_law(X, theta, nothing, false, false, strain_scale=strain_scale, stress_scale=stress_scale)

@show norm(out1 - out2)