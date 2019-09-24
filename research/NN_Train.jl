
# Choose model and dimension of x array and y array, and nn output 
model_type, kx, ky, ky_nn = "Plasticity", 1, 2, 1
#model_type, kx, ky, ky_nn = "Plasticity", 1, 2, 2
#model_type, kx, ky, ky_nn  = "PlasticityLawBased", 1, 1, 1


nn_type = "piecewise2"

#nn_type = "ae"

include("CommonFuncs.jl")
# m set of data, each has n time steps(including initial points)
m, n = 4, 201


xs_set, ys_set = generate_data(model_type, m, n)
sess = Session()
Random.seed!(2333)  
loss = compute_sequence_loss(xs_set, ys_set, nn)
init(sess)
BFGS!(sess, loss)



# point2point test
ys_pred_set = point2point_test(xs_set, ys_set, sess)

colors = ["blue", "green" , "red", "cyan", "magenta", "yellow", "black"]
close("all")
for i = 1:m
    plot(xs_set[i], ys_set[i][:,1], color=colors[i])
    plot(xs_set[i], ys_pred_set[i][:,1], color=colors[i], ".-")
end
savefig("S_Train_P2P_Test_NN$(nn_type)_Prob$(model_type)_ky_nn$(ky_nn).png")



# sequence test
ys_pred_set = sequence_test(xs_set, sess)

colors = ["blue", "green" , "red", "cyan", "magenta", "yellow", "black"]
close("all")
for i = 1:m
    plot(xs_set[i], ys_set[i][:,1], color=colors[i])
    plot(xs_set[i], ys_pred_set[i][:,1], color=colors[i], ".-")
end
savefig("S_Train_S_Test_NN$(nn_type)_Prob$(model_type)_ky_nn$(ky_nn).png")

