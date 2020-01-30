
include("CommonFuncs.jl")
threshold = 1e7 # σY ≈ 1e8

idx = 2
H_function = spd_Chol_Orth

@show length(ARGS)
if length(ARGS)==1
    global idx = parse(Int64, ARGS[1])
else
    @warn("No ARGS provided")
end

@show H_function
if H_function==spd_zero_to_H || H_function==spd_Chol_Orth
    global nout = 4
end


if idx == 0
    global config=[20,20,20,nout] 
elseif idx == 1
    global config=[20,20,20,20,nout]
elseif idx == 2
    global config=[20,20,20,20,20,nout]
elseif idx == 3
    global config=[20,20,20,20,20,20,nout]
elseif idx == 5
    global config=[nout]
end
printstyled("idx = $idx, config=$config, H_function=$H_function\n", color=:green)


function nn(ε, ε0, σ0) # ε, ε0, σ0 450x3
    local y, z
    global H0
    if nntype=="linear"
        y = ε*H0
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

        if H_function==spd_H 
            z = H_function(y, H0)
        elseif H_function==spd_zero_to_H
            z = H_function(y, H0inv)
        else
            z = H_function(y)
        end
        # z = bind(z, op)
        # z = sym_H(y)

        σnn = squeeze(tf.matmul(z, tf.reshape((ε-ε0)/strain_scale, (-1,3,1)))) 
        σH = (ε-ε0)/strain_scale * H0
        s = σ0[:,1]^2-σ0[:,1]*σ0[:,2]+σ0[:,2]^2+3*σ0[:,3]^2 

        i = sigmoid(1000*(s-threshold)/1e9)        
        i = [i i i]
        out = σnn .* i + σH .* (1-i)  + σ0/stress_scale
        out*stress_scale
    elseif nntype=="orthpiecewise"
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        #x_orth = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
        mask = ones(Float64, size(x))
        mask[:, [3,6,9]] *= -1.0
        x_orth = x .* mask

        x = constant(x)
        x_orth = constant(x_orth)

        ε = constant(ε)
        ε0 = constant(ε0)
        σ0 = constant(σ0)
        
        y = ae(x, config, nntype) + ae(x_orth, config, nntype)

        if H_function==spd_H 
            z = H_function(y, H0)
        elseif H_function==spd_zero_to_H
            z = H_function(y, H0inv)
        else
            z = H_function(y)
        end
        # z = bind(z, op)
        # z = sym_H(y)

        σnn = squeeze(tf.matmul(z, tf.reshape((ε-ε0)/strain_scale, (-1,3,1)))) 
        σH = (ε-ε0)/strain_scale * H0
        s = σ0[:,1]^2-σ0[:,1]*σ0[:,2]+σ0[:,2]^2+3*σ0[:,3]^2 

        i = sigmoid(1000*(s-threshold)/1e9)        
        i = [i i i]
        out = σnn .* i + σH .* (1-i)  + σ0/stress_scale
        out*stress_scale

    elseif nntype=="isopiecewise"
        x = [(ε[:,1]+ε[:,2])/strain_scale (ε[:,1].*ε[:,2]-0.25*ε[:,3].*ε[:,3])/(strain_scale^2) (ε0[:,1]+ε0[:,2])/strain_scale (ε0[:,1].*ε0[:,2]-0.25*ε0[:,3].*ε0[:,3])/(strain_scale^2) (σ0[:,1]+σ0[:,2])/stress_scale (σ0[:,1].*σ0[:,2]-σ0[:,3].*σ0[:,3])/(stress_scale^2)]
        x = constant(x)
        ε = constant(ε)
        ε0 = constant(ε0)
        σ0 = constant(σ0)
        
        y = ae(x, config, nntype) 
        if H_function==spd_H 
            z = H_function(y, H0)
        elseif H_function==spd_zero_to_H
            z = H_function(y, H0inv)
        else
            z = H_function(y)
        end
        # z = bind(z, op)
        # z = sym_H(y)

        σnn = squeeze(tf.matmul(z, tf.reshape((ε-ε0)/strain_scale, (-1,3,1)))) 
        σH = (ε-ε0)/strain_scale * H0
        s = σ0[:,1]^2-σ0[:,1]*σ0[:,2]+σ0[:,2]^2+3*σ0[:,3]^2 

        i = sigmoid(1000*(s-threshold)/1e9)        
        i = [i i i]
        out = σnn .* i + σH .* (1-i)  + σ0/stress_scale
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
        x = reshape([ε;ε0;σ0/stress_scale],1, 9)
        reshape(nnae_scaled(x)*stress_scale,3,1)
    elseif nntype=="piecewise"
        ε = ε/strain_scale
        ε0 = ε0/strain_scale
        σ0 = σ0/stress_scale
        x = reshape([ε;ε0;σ0],1, 9)
        # y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*sym_H(nnpiecewise(x))
        if H_function==spd_H
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnpiecewise(x), H0)

        elseif H_function==spd_zero_to_H
            
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnpiecewise(x), H0inv)
        else
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnpiecewise(x))
        end
        y1 = reshape(y1, 3, 1)
        y2 = reshape((reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H0, 3,1)
        s = σ0[1]^2-σ0[1]*σ0[2]+σ0[2]^2+3*σ0[3]^2 
        i = sigmoid_(1000*(s*stress_scale^2 - threshold)/1e9) 
        out = y1 * i + y2 * (1-i)  + reshape(σ0, 3, 1)
        out*stress_scale
    elseif nntype=="orthpiecewise"
        ε = ε/strain_scale
        ε0 = ε0/strain_scale
        σ0 = σ0/stress_scale
        x = reshape([ε;ε0;σ0],1, 9)
        x_orth = reshape([ε;ε0;σ0],1, 9)
        x_orth[1, [3,6,9]] *= -1.0

        if H_function==spd_H
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnorthpiecewise(x)+nnorthpiecewise(x_orth), H0)

        elseif H_function==spd_zero_to_H
            
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnorthpiecewise(x)+nnorthpiecewise(x_orth), H0inv)
        else
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnorthpiecewise(x)+nnorthpiecewise(x_orth))
        end
        y1 = reshape(y1, 3, 1)
        y2 = reshape((reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H0, 3,1)
        s = σ0[1]^2-σ0[1]*σ0[2]+σ0[2]^2+3*σ0[3]^2 
        i = sigmoid_(1000*(s*stress_scale^2 - threshold)/1e9) 
        out = y1 * i + y2 * (1-i)  + reshape(σ0, 3, 1)
        out*stress_scale
    elseif nntype=="isopiecewise"
        ε = ε/strain_scale
        ε0 = ε0/strain_scale
        σ0 = σ0/stress_scale
        x = [(ε[1]+ε[2]) (ε[1]*ε[2]-0.25*ε[3]*ε[3]) (ε0[1]+ε0[2]) (ε0[1]*ε0[2]-0.25*ε0[3]*ε0[3]) (σ0[1]+σ0[2]) (σ0[1]*σ0[2]-σ0[3]*σ0[3])]
        # y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*sym_H(nnpiecewise(x))
        if H_function==spd_H
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnisopiecewise(x), H0)

        elseif H_function==spd_zero_to_H
            
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnisopiecewise(x), H0inv)
        else
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(nnisopiecewise(x))
        end
        y1 = reshape(y1, 3, 1)
        y2 = reshape((reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H0, 3,1)
        s = σ0[1]^2-σ0[1]*σ0[2]+σ0[2]^2+3*σ0[3]^2 
        i = sigmoid_(1000*(s*stress_scale^2 - threshold)/1e9) 
        out = y1 * i + y2 * (1-i)  + reshape(σ0, 3, 1)
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
