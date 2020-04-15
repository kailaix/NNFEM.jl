using DelimitedFiles
using Statistics

D = Dict()
S = Dict()
for activation in ["tanh", "relu", "leaky_relu", "selu", "elu"]
    D[activation] = zeros(4,3)
    for (k1, nn_depth) in enumerate([3,5,10,20])
        for (k2, nn_width) in enumerate([5,20,40])
            FILEID2 = "nn_width$(nn_width)nn_depth$(nn_depth)sigmaY10000.0dsigma1.0activation$(activation)"
            try 
                V = readdlm(FILEID2*".txt")
                D[k1, k2] = mean(V)
                S[k1, k2] = std(V)
                # println(val)
            catch
                println("srun -n 1 -N 1 ./Benchmark_.sh $nn_width $nn_depth 10000.0 1.0 $activation &")
            end
        end
    end
end





using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")

sigmaY = [100.0, 0.001e6, 5000.0, 0.01e6, 50000.0, 0.1e6, 1e6]
dsigma = [0.1, 1.0, 10.0, 100.0]
Mean = zeros(length(sigmaY),length(dsigma))
Std = zeros(length(sigmaY),length(dsigma))
for activation in ["tanh"]#, "relu", "leaky_relu", "selu", "elu"]
    for (k1,σY) in enumerate(sigmaY)
        for (k2,dσ) in enumerate(dsigma)
            FILEID2 = "nn_width20nn_depth3sigmaY$(σY)dsigma$(dσ)activation$(activation).txt"
            # @show isfile(FILEID2)
            try 
                V = readdlm(FILEID2)[:]
                Mean[k1,k2] = mean(V)
                Std[k1,k2] = std(V)
            catch
                @show FILEID2
                println("srun -n 1 -N 1 ./Benchmark_.sh 20 3 $(σY) $(dσ) $activation &")
            end
        end
    end
end
close("all")
for i = 1:4
    loglog(sqrt.(sigmaY), Mean[:,i], label="\$d=$(dsigma[i])\$", linewidth=4)
end
xlabel("\$\\sigma_Y\$")
ylabel("Error")
legend()
grid("on")
gca().invert_xaxis()
mpl.save("compare_d.tex")