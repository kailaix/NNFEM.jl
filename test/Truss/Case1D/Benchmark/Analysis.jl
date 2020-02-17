using DelimitedFiles
using Statistics

for activation in ["tanh", "relu", "leaky_relu", "selu", "elu"]
    for nn_width in [3,5,10,20]
        for nn_depth in [5,20,40]
            FILEID2 = "nn_width$(nn_width)nn_depth$(nn_depth)sigmaY10000.0dsigma1.0activation$(activation)"
            try 
                V = readdlm("Data/"*FILEID2*".txt")
                val = median(V)
                println(val)
            catch
                @warn FILEID2
            end
        end
    end
end
