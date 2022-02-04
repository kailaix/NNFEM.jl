
include("CommonFuncs.jl")
threshold = 1e7 # σY ≈ 1e8


if nntype=="ae_scaled"
    global nout = 3
end


if idx == 1
    global config=[20,nout]
elseif idx == 2
    global config=[20, 20, nout] 
elseif idx == 3
    global config=[20, 20, 20, nout] 
elseif idx == 4
    global config=[20, 20, 20, 20, nout]
elseif idx == 5
    global config=[nout]
end
printstyled("idx = $idx, config=$config, H_function=$H_function\n", color=:green)


function nn(ε, ε0, σ0) # ε, ε0, σ0 450x3
    local y, z
    global H0
    if nntype=="linear"
        y = ε*H0*stress_scale
        y
    elseif nntype=="ae_scaled"
        x = ε/strain_scale
        if isa(x, Array)
            x = constant(x)
        end
        y = ae(x, config, nntype)*stress_scale
    elseif nntype=="stiffmat"
        x = ε/strain_scale
        x = constant(x)
        
        σ0 = constant(σ0)
        
        y = ae(x, config, nntype)
        
        if H_function==spd_H
            z = spd_H(y, H0)
        else
            z = H_function(y)
        end
        
	    

        σnn = squeeze(tf.matmul(z, tf.reshape((ε-ε0)/strain_scale, (-1,3,1)))) 

        ####################################
        # σH = (ε-ε0)/strain_scale * H0
        # s = σ0[:,1]^2-σ0[:,1]*σ0[:,2]+σ0[:,2]^2+3*σ0[:,3]^2 

        # i = sigmoid(1000*(s-threshold)/1e9)        
        # i = [i i i]
        # out = σnn .* i + σH .* (1-i)  + σ0/stress_scale
        ###############################################
        
        out = σnn + σ0/stress_scale


        out*stress_scale    
    else
        error("$nntype does not exist")
    end
end



function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

function nn_helper(ε, ε0, σ0)
    local y1
    if nntype=="linear"
        x = reshape(reshape(ε,1,3)*H0,3,1)
    elseif nntype=="ae_scaled"
        x = reshape(ε/strain_scale,1, 3)
        reshape(nnae_scaled(x)*stress_scale,3,1)
    elseif nntype=="stiffmat"
        ε = ε/strain_scale
        ε0 = ε0/strain_scale
        σ0 = σ0/stress_scale
        x = reshape(ε, 1, 3)
        # y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*sym_H(nnpiecewise(x))
        if H_function==spd_H
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*spd_H(nnstiffmat(x), H0)
        else
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnstiffmat(x))
        end
        y1 = reshape(y1, 3, 1)
        ################################################
        # y2 = reshape((reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H0, 3,1)
        # s = σ0[1]^2-σ0[1]*σ0[2]+σ0[2]^2+3*σ0[3]^2 
        # i = sigmoid_(1000*(s*stress_scale^2 - threshold)/1e9) 
        # out = y1 * i + y2 * (1-i)  + reshape(σ0, 3, 1)
        #####################################################
        out = y1  + reshape(σ0, 3, 1)
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
