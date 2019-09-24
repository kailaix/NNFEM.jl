using ADCME
using PyPlot
using Random
using PyCall
np = pyimport("numpy")

# ! need kx, ky and nn_type, and ky_nn
# ! E is always 1
# ! loss is based only on y[:,1]
function hidden_function(model_type, x, x_, y_)
    if model_type == "Plasticity"
        # x  = [ε_n+1]
        # x_ = [ε_n]
        # y_ = [σ_n, α_n] or [σ_n, α_n, q_n] 
        # The yield condition is
        #      f = |sigma - q| - (σY + alpha * K)
        # here K is the plastic modulus, , σY is the flow stress
        # D_eps_p = gamma df/dsigma        
        # D_alpha = |D_eps_p|
        # D_q     = B * D_eps_p
        # f  = 0    or f  < 0
        #trial stress
        
        σY,  E,  K, B =  0.006, 1.0, 0.1, 0.01
        if length(y_) == 3
            σ0, α0, q0 = y_[1], y_[2], y_[3]  
        elseif length(y_) == 2
            σ0, α0, q0 = y_[1], y_[2], 0.0  
            B = 0.0
        end
        ε, ε0 = x[1], x_[1]
        Δγ = 0.0
        σ = σ0 + E*(ε - ε0) 
        α = α0 + abs(Δγ)
        q = q0 + B*Δγ
        ξ = σ - q

        r2 = abs(ξ) - (σY + K*α)
        if r2 <= 0
            σ = σ0 + E*(ε - ε0)
            dΔσdΔε = E

        else
        
            Δγ = r2/(B + E + K)
            q += B * Δγ * sign(ξ)
            α += Δγ
            σ -= Δγ * E * sign(ξ)
            dΔσdΔε = E*(B + K)/(B + E + K)
        end
        if length(y_) == 3 
            y = [σ; α + Δγ; q]
        elseif length(y_) == 2
            y = [σ; α + Δγ]
        end
            
    elseif model_type == "PlasticityLawBased"
        # x  = [ε_n+1]
        # x_ = [ε_n]
        # y_ = [σ_n] 
        # The yield condition is
        #      f = σ_vm - (α + β * ε_vm^pow)
        # D_eps_p = gamma df/dsigma        
        # f  = 0    or f  < 0
        

        E, α, β, pow =  1.0, 0.006, 0.018, 0.4
        σ0 = y_[1]
        ε, ε0 = x[1], x_[1]
        Δγ = 0.0

        #trial stress
        σ = σ0 + E*(ε - ε0) 
        ε_vm = α + β*2.0/3.0 * abs(ε)
        r2 = abs(σ) - ε_vm
        if r2 <= 0
            σ = σ0 + E*(ε - ε0)
            dΔσdΔε = E

        else
        
            Δγ = r2/E
            σ -= Δγ * E * sign(σ)
        end
         
        y = [σ]
        
    else
        error("model_type ", model_type, " have not implemented yet ")
    end
    
    return y
end

function generate_sequence(model_type, xs, y0)
    n = size(xs,1)
    ys = zeros(n,ky)
    ys[1,:] = y0
    for i = 2:n 
        ys[i,:] = hidden_function(model_type, xs[i,:], xs[i-1,:], ys[i-1,:])
        @show "generate data ", i, ys[i,:]
    end
    ys 
end







@doc """
    generate data 
    m set of data, each is a serial sequency of n time
"""->
function generate_data(model_type, m = 2, n = 100)
    xs_set, ys_set = [], []

    # generate xs_set
    if model_type == "Plasticity" || model_type == "PlasticityLawBased"
        T = 0.1
        #
        t = np.linspace(0.0, T, n)
        A = 0.02
        xs = A * reshape(sin.(π*t/(3.0*T)), :, kx)
        push!(xs_set, xs)

        #
        if m >= 2
            t = np.linspace(0.0, T, n)
            A = 0.02
            xs = A * reshape(sin.(π*t/(2.5*T)), :, kx)
            push!(xs_set, xs)
        end

        if m >= 3
            t = np.linspace(0.0, T, n)
            A = 0.01
            xs = A * reshape(sin.(π*t/(2.5*T)), :, kx)
            push!(xs_set, xs)
        end

        if m >= 4
            t = np.linspace(0.0, T, n)
            A = 0.01
            xs = A * reshape(sin.(π*t/(3.0*T)), :, kx)
            push!(xs_set, xs)
        end

    else
        error("model_type ", model_type, " have not implemented yet ")
    end

    # generate ys_set array
    for i = 1:m
        y0 = zeros(ky) 
        ys = generate_sequence(model_type, xs_set[i], y0)
        push!(ys_set, ys)
    end

    return xs_set, ys_set 
end


function sequence_test(xs_set, sess)
    m = length(xs_set)
    ys_pred_set = []
    for i_set = 1:m
        xs = xs_set[i_set]
        n = size(xs,1)
        ys_pred = zeros((n, ky_nn))

        plx = placeholder(zeros(kx), shape=kx)
        plx_ = placeholder(zeros(kx), shape=kx)
        ply_ = placeholder(zeros(ky_nn), shape=ky_nn)
        res = nn(plx, plx_, ply_)

        for i = 2:n
            ys_pred[i,:] = run(sess,res, feed_dict = Dict(
            plx=>xs[i,:],
            plx_=>xs[i-1,:],
            ply_=>ys_pred[i-1,:]))
            @show i, ys_pred[i,:]
        end
        push!(ys_pred_set, ys_pred)
    end

    return ys_pred_set

end


function point2point_test(xs_set, ys_set, sess)
    m = length(xs_set)
    ys_pred_set = []
    for i_set = 1:m
        xs, ys = xs_set[i_set], ys_set[i_set]
        n = size(xs,1)
        ys_pred = zeros((n, ky))

        ys_pred[2:end,1:ky_nn] = run(sess, nn_all(xs[2:end,:], xs[1:end-1,:], ys[1:end-1, 1:ky_nn]))
        push!(ys_pred_set, ys_pred)
    end

    return ys_pred_set

end


######################################################################
function compute_sequence_loss(xs_set, ys_set, nn)
    loss = constant(0.0)
    m = length(xs_set)

    for i_set = 1:m
        xs, ys = xs_set[i_set], ys_set[i_set][:, 1:ky_nn]
        n = size(xs,1)
        y = constant(ys[1,:])
        for i = 2:n
            y = nn(constant(xs[i,:]), constant(xs[i-1,:]), y)
            #todo change the loss function
            #loss += (ys[i,:] - y)^2[1]

            @show i, size(y)

            loss += 100000.0* ((ys[i,:] - y)^2)[1]

            #loss += (ys[i,1] - y[1])^2
 
        end
    end
    return loss
end

######################################################################
function compute_point2point_loss(xs_set, ys_set, nn)
    loss = constant(0.0)
    m = length(xs_set)

    for i_set = 1:m
        xs, ys = xs_set[i_set], ys_set[i_set]
        

        y = nn_all(xs[2:end,:], xs[1:end-1,:], ys[1:end-1, 1:ky_nn])

    loss += sum((y - ys[2:end, 1:ky_nn])^2)
    end
    return loss
end




config=[20,ky_nn]
function nn(x, x_, y_)
    

    #ipt = reshape([x;x_;y_], 1, :)
    if nn_type == "ae"  
        ipt = reshape([x;x_;y_], :, 2*kx + ky_nn)
    
        out = ae(ipt, config)
        squeeze(out, dims=1)

    elseif nn_type=="piecewise"
    
        E = 1.0
        threshold = 1.e-5
        ipt = reshape([x;x_;y_], :, 2*kx + ky_nn)

        σnn = ae(ipt, config) 
        σH = E * (x - x_)
        z = sum(x^2)
        
        i = sigmoid(1e6*(z-threshold))        
        
        out = σnn * i + σH * (1-i)  + y_

        squeeze(out, dims=1)

    elseif nn_type=="piecewise2"
    
        @assert(ky_nn == 1)
        E = 1.0
        threshold = 1.e-5
        ipt = reshape([x;x_;y_], :, 2*kx + ky_nn)

        σnn = reshape(ae(ipt, config)* (x - x_), 1, 1)
        
        σH = E * (x - x_)
        z = sum(x^2)
        
        
        i = sigmoid(1e6*(z-threshold))        
        
        out = σnn * i + σH * (1-i)  + y_

        squeeze(out, dims=1)
    else
        error("$nn_type does not exist")
    end
end



function sigmoid_(z)

    return 1.0 ./ (1.0 .+ exp.(-z))
  
end

function nn_all(x, x_, y_)
    if nn_type == "ae"  
    
        ipt = [x x_ y_]
    
        #out = ae(ipt, [20,ky_nn])
        out = ae(ipt, config)
    elseif nn_type=="piecewise"
    
        
        E = 1.0
        threshold = 1.e-5
        ipt = [x x_ y_]

        σnn = ae(ipt, config)
        σH = E * (x - x_)
        z = x.^2
        
        i = sigmoid_(1e6*(z.-threshold))        
        
        out = σnn .* i + σH .* (1 .- i)  .+ y_

    elseif nn_type=="piecewise2"
    
        
        E = 1.0
        threshold = 1.e-5
        ipt = [x x_ y_]

        σnn = ae(ipt, config) .* (x - x_)
         
        σH = E * (x - x_)
        z = x.^2
        
        i = sigmoid_(1e6*(z.-threshold))        
        
        out = σnn .* i + σH .* (1 .- i)  .+ y_

    else
        error("$nn_type does not exist")
    end
end