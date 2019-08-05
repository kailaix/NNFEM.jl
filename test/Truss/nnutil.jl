
using ForwardDiff
using DelimitedFiles



function nn(ε, ε0, σ0)
    local y, y1, y2, y3

    if nntype=="linear"
        y = ε*H0
        y
    elseif nntype=="nn"
        x = [ε ε0 σ0]
        # op1 = tf.print("* ", ε,summarize=-1)
        # y = bind(y, op1)
        # op2 = tf.print("& ", x, summarize=-1)
        # y = bind(y, op2)
        y1 = x*W1+b1
        y2 = tanh(y1)
        y2 = y2*W2+b2
        y3 = tanh(y2)
        y3 = sigmoid(y3*W3+b3)
        # i = cast(squeeze(y3)>0.5, Float64)
        # i = squeeze(y3)
        i = y3
        @show size(y3)


        y1 = x*_W1+_b1
        y2 = tanh(y1); y1_ = y1*_W1o; @show y1_
        y2 = y2*_W2+_b2
        y3 = tanh(y2); y2_ = y2*_W2o; @show y2_
        y3 = y3*_W3+_b3
        y3 = tanh(y3); y3_ = y3*_W3o; @show y3_
        @show size(y3)

        @show i
        @show i .* (σ0 + (ε-ε0)*H0)
        @show (1-i) .* (y1_+y2_+y3_)
        out = i .* (σ0 + (ε-ε0)*H0) + (1-i) .* (y1_+y2_+y3_)
        @show out
    elseif nntype=="ae"
        x = [ε0 ε σ0]
        y = ae(x, [20,20,20,20,1], "nn")
    elseif nntype=="ae_scaled"
        x = [ε0 ε σ0/100.0]
        y = ae(x, [20,20,20,20,1], "ae_scaled")*100.0
    end

    
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
    
    i = squeeze(y3)

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
    out = i * (σ0' + (ε-ε0)'*H0) + (1 - i) * (y1+y2+y3) 
    out = reshape(out, 1)
    # println(out)
    # error()
end

function post_nn(ε, ε0, σ0, Δt)
    # @show "Post NN"
    f = x -> nn_helper(x, ε0, σ0)
    df = ForwardDiff.jacobian(f, ε)
    return f(ε), df
end
