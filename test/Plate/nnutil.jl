
using ForwardDiff
using DelimitedFiles


function nn(ε, ε0, σ0) # ε, ε0, σ0 are all length 3 vector
    local y

    if nntype=="linear"
        y = ε*H0
        y
    elseif nntype=="ae"
        x = [ε ε0 σ0]
        y = ae(x, [20,20,20,20,3], "ae")
    elseif nntype=="ae_scaled"
        x = [ε ε0 σ0/stress_scale]
        if isa(x, Array)
            x = constant(x)
        end
        y = ae(x, [20,20,20,20,3], "ae_scaled")*stress_scale
    end

end



function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

function nn_helper(ε, ε0, σ0)
    if nntype=="ae_scaled"
        x = reshape([ε;ε0;σ0/stress_scale],1, 9)
        reshape(nnae_scaled(x)*stress_scale,3,1)
    elseif nntype=="linear"
        x = reshape(reshape(ε,1,3)*H0,3,1)
    end
end

function post_nn(ε, ε0, σ0, Δt)
    # @show "Post NN"
    f = x -> nn_helper(x, ε0, σ0)
    df = ForwardDiff.jacobian(f, ε)
    return f(ε), df
end
