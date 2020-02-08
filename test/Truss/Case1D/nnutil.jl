
using ForwardDiff
using DelimitedFiles
using MAT


nntype = length(ARGS)>=1 ? ARGS[1] : "linear"

stress_scale = 1e3
strain_scale = 1e-2
config = [20,20,1]
E0 = 200.0e3
#H0 = 20.0e3

function nn(ε, ε0, σ0)
    local y
    if nntype=="linear"
        y = E0 * ε
    elseif nntype=="ae_scaled"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        y = ae(x, config, "ae_scaled")*stress_scale
    elseif nntype=="piecewise"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        H = ae(x, config, "piecewise")*stress_scale
        s = σ0^2
        i = sigmoid((s - 0.01e6))  

        op = tf.print(tf.reduce_sum(i))
        i = bind(i, op)
 
        y = ( H.* i + E0 * (1-i) ) .* (ε-ε0) + σ0

        #y = H  .* (ε-ε0) + σ0
    else
        error("nntype must be specified.")
    end

    return y

    
end


function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

function post_nn(ε::Float64, ε0::Float64, σ0::Float64, Δt::Float64)
    # @show "Post NN"
    local f, df
    if nntype=="linear"
        f = x -> E0 * ε
        df = ForwardDiff.derivative(f, ε)
    elseif nntype=="piecewise"
        f = x -> begin
            H = nnpiecewise(reshape([x;ε0;σ0/stress_scale],1,3))[1,1]^2*stress_scale
            s = σ0^2
            i = sigmoid_((s - 0.01e6)) 
            ( H * i + E0 * (1-i) ) * (x-ε0) + σ0
        end
        df = ForwardDiff.derivative(f, ε)
    elseif nntype=="ae_scaled"
        f = x -> nnae_scaled(reshape([x;ε0;σ0/stress_scale],1,3))[1,1]*stress_scale
        df = ForwardDiff.derivative(f, ε)
    end
    return f(ε), df
end
