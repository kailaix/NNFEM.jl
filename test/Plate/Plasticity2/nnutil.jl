
include("CommonFuncs.jl")
threshold = 1e-5
wgt_func = x->1. + 100x^3



if length(ARGS)==1
    global idx = parse(Int64, ARGS[1])
else
    global idx = 0
end

if idx == 0
    global config=[20,20,20,20,4]
elseif idx == 1
    global config=[20,20,20,20,20,4] 
elseif idx == 2
    global config=[20,20,20,20,20,20,4] 
elseif idx == 3
    global config=[20,20,20,20,20,20,20,4] 
end
printstyled("idx = $idx, config=$config", color=:green)


function nn(ε, ε0, σ0) # ε, ε0, σ0 are all length 3 vector
    local y
    global H0
    if nntype=="linear"
        y = ε*H0*stress_scale
        y
    elseif nntype=="ae_scaled"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        if isa(x, Array)
            x = constant(x)
        end
        # y = ae(x, [50,50,50,50,50,50,50,3], nntype)*stress_scale
        # y = ae(x, [50,50,50,50,50,50,50,3], nntype)*stress_scale
        y = ae(x, config, nntype)*stress_scale
    elseif nntype=="piecewise"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        x = constant(x)
        ε = constant(ε)
        ε0 = constant(ε0)
        σ0 = constant(σ0)
        
        y = ae(x, config, nntype)
        # z = orthotropic_H(y)
        @show y
        # y = NNFEM.orthotropic_op(y)
        y = constant(rand(size(y,1), 9))
    @show y
        z = tf.reshape(y, (-1,3,3)) 

        σnn = squeeze(tf.matmul(z, tf.reshape((ε-ε0)/strain_scale, (-1,3,1)))) 
        σH = (ε-ε0)/strain_scale * H0
        z = sum(ε^2,dims=2)
        # i = sigmoid(1e9*(z-(threshold)^2))
        i = sigmoid(1e6*(z-threshold))        
        i = [i i i]
        #out = σnn .* i + σH .* (1-i)  + σ0/stress_scale
        out = σnn .* i + σH + σ0/stress_scale
        out*stress_scale
    else
        error("$nntype does not exist")
    end
end



function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

function nn_helper(ε, ε0, σ0)
    if nntype=="linear"
        x = reshape(reshape(ε,1,3)*H0,3,1)
    elseif nntype=="ae_scaled"
        x = reshape([ε;ε0;σ0/stress_scale],1, 9)
        reshape(nnae_scaled(x)*stress_scale,3,1)
    elseif nntype=="piecewise"
        ε = ε/strain_scale
        ε0 = ε0/strain_scale
        σ0 = σ0/stress_scale
        x = reshape([ε;ε0;σ0],1, 9)
        y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*orthotropic_H(nnpiecewise(x))
        y1 = reshape(y1, 3, 1)
        y2 = reshape((reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H0, 3,1)
        # y2 = reshape(reshape(ε,1,3)*H0,3,1)
        i = sigmoid_(1e6*(norm(ε)^2-threshold))
        # @show y1 * i
        #out = y1 * i + y2 * (1-i)  + reshape(σ0, 3, 1)
        out = y1 * i + y2 + reshape(σ0, 3, 1)
        out*stress_scale
    else
        error("$nntype does not exist")
    end
end

function post_nn(ε, ε0, σ0, Δt)
    f = x -> nn_helper(x, ε0, σ0)
    df = ForwardDiff.jacobian(f, ε)
    return f(ε), df
end
