
using ForwardDiff
using DelimitedFiles


function nn(ε, ε0, σ0)
    local y, y1, y2, y3
    if nntype=="linear"
        y = ε*H*1e11
        # op1 = tf.print("* ", ε,summarize=-1)
        # y = bind(y, op1)
        # op2 = tf.print("& ", y, summarize=-1)
        # y = bind(y, op2)
        y
    elseif nntype=="nn"
        x = [ε ε0 σ0]
    
        y1 = x*W1+b1
        y2 = tanh(y1)
        y2 = y2*W2+b2
        y3 = tanh(y2)
        y3 = sigmoid(y3*W3+b3)
        # i = cast(squeeze(y3)>0.5, Float64)
        i = squeeze(y3)
        i = [i i i]

        y1 = x*_W1+_b1
        y2 = tanh(y1)
        y2 = y2*_W2+_b2
        y3 = tanh(y2)
        y3 = y3*_W3+_b3
        i .* (σ0 + (ε-ε0)*H0*1e11) + (1-i) .* (y1+y2+y3)
        
    end
    # op = tf.print(σ0)
    # y = bind(y, op)
    
end


function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

function nn_helper(ε, ε0, σ0)
    x = [ε;ε0;σ0]'
    
    y1 = x*W1 .+ b1'
    y2 = tanh.(y1)
    y2 = y2*W2 .+ b2'
    y3 = tanh.(y2)
    y3 = sigmoid_.(y3*W3 .+ b3')
    
    i = reshape(y3, length(y3))
    i = [i i i]
    # @show i
    y1 = x*_W1 .+ _b1'
    y2 = tanh.(y1)
    y2 = y2*_W2 .+ _b2'
    y3 = tanh.(y2)
    y3 = y3*_W3 .+ _b3'
    # @show i
    # println(size(i), size(y1), size(y2), size(y3), size(σ0),
    #                 size(ε-ε0), size(H0))
    # @show size(σ0' + (ε-ε0)'*H0*1e11)
    out = i .* (σ0' + (ε-ε0)'*H0*1e11) + (1 .- i) .* (y1+y2+y3) 
    out = reshape(out, 3)
    # println(out)
    # error()
end

function post_nn(ε, ε0, σ0, Δt)
    # @show "Post NN"
    f = x -> nn_helper(x, ε0, σ0)
    df = ForwardDiff.jacobian(f, ε)
    return f(ε), df
end
