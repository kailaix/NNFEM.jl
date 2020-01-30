
using ForwardDiff
using DelimitedFiles
using MAT


function nn(ε, ε0, σ0)
    local y, y1, y2, y3
    if nntype=="linear"
        y = ε*H0
        y
    elseif nntype=="ae_scaled"
        x = [ε ε0 σ0/1e10]
        y = ae(x, [20,20,20,20,1], "ae_scaled")*1e10
    end
end


function sigmoid_(z)
    return 1.0 / (1.0 + exp(-z))
end


function post_nn(ε::Float64, ε0::Float64, σ0::Float64, Δt::Float64)
    # @show "Post NN"
    f = x -> nnae_scaled([x ε0 σ0/1e10])*1e10
    df = ForwardDiff.derivative(f, ε)
    return f(ε)[1], df[1]
end
