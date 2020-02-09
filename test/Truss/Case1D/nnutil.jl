
using ForwardDiff
using DelimitedFiles
using MAT


nntype = length(ARGS)>=1 ? ARGS[1] : "piecewise"
idx = length(ARGS)>=2 ? parse(Int,ARGS[2]) : 3
config = [20,20,20,1]

if idx == 1
    config = [20,1]
elseif idx == 2
    config = [20,20,1]
elseif idx == 3
    config = [20,20,20,1]
else 
    error(idx == 3, idx, "idx <= 3")
end


stress_scale = 1e2
strain_scale = 1e-3

E0 = 200.0e3/(stress_scale/strain_scale)


function nn(ε, ε0, σ0)
    # make sure all inputs are 2d matrix
    @show ε, ε0, σ0

    local y
    if nntype=="linear"
        y = E0 * ε
    elseif nntype=="ae_scaled"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        y = ae(x, config, "ae_scaled")*stress_scale/strain_scale
    elseif nntype=="piecewise"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        H = ae(x, config, "piecewise")^2
        s = σ0^2
        i = sigmoid((s - 0.01e6))  
 
        y = ( H.* i + E0 * (1-i) ) .* (ε-ε0)*stress_scale/strain_scale + σ0

    elseif nntype=="piecewise2"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        H = ae(x, config, "piecewise2")
        s = σ0^2
        i = sigmoid((s - 0.01e6))  
 
        y = ( H.* i + E0 * (1-i) ) .* (ε-ε0)*stress_scale/strain_scale + σ0
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
        f = x -> E0*x
        df = ForwardDiff.derivative(f, ε)
    elseif nntype=="piecewise"
        f = x -> begin
            H = nnpiecewise(reshape([x/strain_scale;ε0/strain_scale;σ0/stress_scale],1,3))[1,1]^2

            @show H
            s = σ0^2
            i = sigmoid_((s - 0.01e6))

            (H * i + E0 * (1.0-i) )*stress_scale/strain_scale * (x-ε0) + σ0

            
        end
        df = ForwardDiff.derivative(f, ε)
    elseif nntype=="piecewise2"
        f = x -> begin
            H = nnpiecewise2(reshape([x/strain_scale;ε0/strain_scale;σ0/stress_scale],1,3))[1,1]

            @show H

            s = σ0^2
            i = sigmoid_((s - 0.01e6))
            (H * i + E0 * (1.0-i))*stress_scale/strain_scale * (x-ε0) + σ0
        end
        df = ForwardDiff.derivative(f, ε)
    elseif nntype=="ae_scaled"
        f = x -> nnae_scaled(reshape([x/strain_scale;ε0/strain_scale;σ0/stress_scale],1,3))[1,1]*stress_scale/strain_scale
        df = ForwardDiff.derivative(f, ε)
    else
        error("$(nntype) must be specified.")
    end
    return f(ε), df
end


# function nn_helper(ε, ε0, σ0)
#     if nntype=="linear"
#         E0 * ε
#     elseif nntype=="piecewise"
#         H = nnpiecewise(reshape([ε/strain_scale;ε0/strain_scale;σ0/stress_scale],1,3))[1,1]^2*stress_scale
#         s = σ0^2
#         i = sigmoid_((s - 0.01e6)) 
#         ( H * i + E0 * (1.0-i) ) * (ε-ε0) + σ0
#     elseif nntype=="ae_scaled"
#         nnae_scaled(reshape([ε/strain_scale;ε0/strain_scale;σ0/stress_scale],1,3))[1,1]*stress_scale
#     else
#         error("$nntype does not exist")
#     end
# end


# function post_nn(ε, ε0, σ0, Δt)
#     f = x -> nn_helper(x, ε0, σ0)
#     df = ForwardDiff.jacobian(f, ε)
#     return f(ε), df
# end
