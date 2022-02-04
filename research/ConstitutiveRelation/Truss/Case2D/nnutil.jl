
using ForwardDiff
using DelimitedFiles
using MAT


function nn(ε, ε0, σ0)
    local y, y1, y2, y3
    if nntype=="linear"
        y = 200 * ε
    elseif nntype=="ae_scaled"
        x = [ε ε0 σ0/stress_scale]
        y = ae(x, [20,20,20,20,1], "ae_scaled")*stress_scale
    elseif nntype=="piecewise"
        x = [ε ε0 σ0/stress_scale]
        H = ae(x, [20,20,20,20,1], "piecewise")^2*stress_scale
        s = σ0^2
        i = sigmoid((s-0.01)/0.001)  
        # @show H, i
        out = ( H .* i + 200 * (1-i) ) .* (ε-ε0) + σ0
    else
        error("nntype must be specified.")
    end

    
end


function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

function post_nn(ε::Float64, ε0::Float64, σ0::Float64, Δt::Float64)
    # @show "Post NN"
    local f, df
    if nntype=="piecewise"
        # error()
        f = x -> begin
            H = nnpiecewise(reshape([x;ε0;σ0/stress_scale],1,3))[1,1]^2*stress_scale
            s = σ0^2
            i = sigmoid_((s-0.01)/0.001) 
            ( H * i + 200 * (1-i) ) * (ε-ε0) + σ0
        end
        df = ForwardDiff.derivative(f, ε)
    elseif nntype=="ae_scaled"
        f = x -> nnae_scaled(reshape([x;ε0;σ0/stress_scale],1,3))[1,1]*stress_scale
        df = ForwardDiff.derivative(f, ε)
    end
    return f(ε), df
end
