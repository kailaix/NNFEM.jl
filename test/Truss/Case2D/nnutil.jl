
using ForwardDiff
using DelimitedFiles
using MAT


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
        x = [ε ε0 σ0]
        y = ae(x, [20,20,20,20,1], "ae")
    elseif nntype=="ae_scaled"
        x = [ε ε0 σ0/100.0]
        y = ae(x, [20,20,20,20,1], "ae_scaled")*100.0
    end

    
end


function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

aedictae_scaled = matread("Data/trained_nn_fem.mat"); # using MAT
Wkey = "ae_scaledbackslashfully_connectedbackslashweightscolon0"
Wkey = "ae_scaledbackslashfully_connected_1backslashweightscolon0"
Wkey = "ae_scaledbackslashfully_connected_2backslashweightscolon0"
Wkey = "ae_scaledbackslashfully_connected_3backslashweightscolon0"
Wkey = "ae_scaledbackslashfully_connected_4backslashweightscolon0"
function nnae_scaled(net)
        net = reshape(net, 1, 3)
        W0 = aedictae_scaled["ae_scaledbackslashfully_connectedbackslashweightscolon0"]; b0 = aedictae_scaled["ae_scaledbackslashfully_connectedbackslashbiasescolon0"];
        isa(net, Array) ? (net = net * W0 .+ b0') : (net = net *W0 + b0)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W1 = aedictae_scaled["ae_scaledbackslashfully_connected_1backslashweightscolon0"]; b1 = aedictae_scaled["ae_scaledbackslashfully_connected_1backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W1 .+ b1') : (net = net *W1 + b1)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W2 = aedictae_scaled["ae_scaledbackslashfully_connected_2backslashweightscolon0"]; b2 = aedictae_scaled["ae_scaledbackslashfully_connected_2backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W2 .+ b2') : (net = net *W2 + b2)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W3 = aedictae_scaled["ae_scaledbackslashfully_connected_3backslashweightscolon0"]; b3 = aedictae_scaled["ae_scaledbackslashfully_connected_3backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W3 .+ b3') : (net = net *W3 + b3)
        isa(net, Array) ? (net = tanh.(net)) : (net=tanh(net))
        W4 = aedictae_scaled["ae_scaledbackslashfully_connected_4backslashweightscolon0"]; b4 = aedictae_scaled["ae_scaledbackslashfully_connected_4backslashbiasescolon0"];
        isa(net, Array) ? (net = net * W4 .+ b4') : (net = net *W4 + b4)
        return net[1,1]
end 

function post_nn(ε::Float64, ε0::Float64, σ0::Float64, Δt::Float64)
    # @show "Post NN"
    f = x -> nnae_scaled([x;ε0;σ0/100.0])*100.0
    df = ForwardDiff.derivative(f, ε)
    return f(ε), df
end
