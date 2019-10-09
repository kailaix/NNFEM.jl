
include("CommonFuncs.jl")
threshold = 1e7 # σY ≈ 1e8

if length(ARGS)==1
    global idx = parse(Int64, ARGS[1])
elseif length(ARGS)==2
    global idx = parse(Int64, ARGS[1])
    global tid = parse(Int64, ARGS[2])
else
    global idx = 0
end

H_function = spd_H
n_internal = 128
ny = 3 + n_internal

if idx == 0
    global config=[20,ny]
elseif idx == 1
    global config=[100,ny] 
elseif idx == 2
    global config=[20,20,20,ny] 
elseif idx == 3
    global config=[20,20,20,20,20,20,ny]
elseif idx == 5
    global config=[ny]
end
printstyled("idx = $idx, config=$config, H_function=$H_function\n", color=:green)


function nn(ε, ε0, σ0, α) # ε, ε0, σ0 450x3
    local y, z
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
        α = repeat(α', size(ε,1), 1)
        x = [ε/strain_scale ε0/strain_scale σ0/stress_scale α]
        x = constant(x)
        ε = constant(ε)
        ε0 = constant(ε0)
        σ0 = constant(σ0)
        
        y = ae(x, config, nntype)
        α = y[ny+1:end]
        y = y[1:ny]
        # op = tf.print("call spd_H")
        if H_function==spd_H
            z = spd_H(y, H0)
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
        out*stress_scale, α
    else
        error("$nntype does not exist")
    end
end



function sigmoid_(z)

    return 1.0 / (1.0 + exp(-z))
  
end

function nn_helper(ε, ε0, σ0, α)
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
        x = reshape([ε;ε0;σ0;α],1, 9)
        # y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*sym_H(nnpiecewise(x))
        out = nnpiecewise(x)
        α = out[1:3]
        out = out[4:end]
        if H_function==spd_H
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*spd_H(out, H0)
        else
            y1 = (reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H_function(out)
        end
        y1 = reshape(y1, 3, 1)
        y2 = reshape((reshape(ε, 1, 3) - reshape(ε0, 1, 3))*H0, 3,1)
        s = σ0[1]^2-σ0[1]*σ0[2]+σ0[2]^2+3*σ0[3]^2 
        i = sigmoid_(1000*(s*stress_scale^2 - threshold)/1e9) 
        out = y1 * i + y2 * (1-i)  + reshape(σ0, 3, 1)
        out*stress_scale, α
    else
        error("$nntype does not exist")
    end
end

function post_nn(ε, ε0, σ0, α, Δt)
    f = x -> nn_helper(x, ε0, σ0, α)[1]
    df = ForwardDiff.jacobian(f, ε)
    val = nn_helper(ε, ε0, σ0, α)
    return val[1:ny], df, val[ny+1:end]
end
