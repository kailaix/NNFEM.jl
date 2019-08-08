
using ForwardDiff
using DelimitedFiles
using MAT


function nn(ε, ε0, σ0)
    local y, y1, y2, y3

    if nntype=="ae_scaled"
        x = [ε ε0 σ0/stress_scale]
        y = ae(x, [20,20,20,20,1], "ae_scaled")*stress_scale
    else
        error("nntype must be specified.")
    end

    
end


function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

function post_nn(ε::Float64, ε0::Float64, σ0::Float64, Δt::Float64)
    # @show "Post NN"
    f = x -> nnae_scaled(reshape([x;ε0;σ0/stress_scale],1,3))[1,1]*stress_scale
    df = ForwardDiff.derivative(f, ε)
    return f(ε), df
end
