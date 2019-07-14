
using ForwardDiff
using DelimitedFiles
W1 = rand(9,3)
b1 = rand(1,3)
W2 = rand(3,3)
b2 = rand(1,3)
W3 = rand(3,3)
b3 = rand(1,3)

function nn_helper(ε, ε0, σ0)
    x = [ε;ε0;σ0]'
    # x = ε
    # y = ae(x, [20,3], "nn")*1e11
    # @show ε,ε0,σ0
    y1 = x*W1+b1
    y2 = tanh.(y1)
    y2 = y2*W2+b2
    y3 = tanh.(y2)
    y3 = @. 1.0 / (1.0 + exp(-(y3*W3+b3)))
    y3 = y3[1:3]
    return y3
end

function post_nn(ε, ε0, σ0, Δt)
    f = x -> nn_helper(x, ε0, σ0)
    df = ForwardDiff.jacobian(f, ε)
    return f(ε), df
end
