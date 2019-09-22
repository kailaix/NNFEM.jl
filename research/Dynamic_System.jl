using ADCME
using PyPlot
using Random
using PyCall
np = pyimport("numpy")


function hidden_function(x, x_, y_)
    if model_type == "Plasticity"
        # x  = ε_n+1
        # x_ = ε_n
        # y_ = [σ_n, α_n]
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
            
  
    else
        error("model_type ", model_type, " have not implemented yet ")
    end
    
    return y
end

function generate_data(xs, y0)
    n = size(xs,1)
    ys = zeros(n,ky)
    ys[1,:] = y0
    for i = 2:n 
        ys[i,:] = hidden_function(xs[i,:], xs[i-1,:], ys[i-1,:])
        @show "generate data ", i, ys[i,:]
    end
    ys 
end




function compute_loss(xs_set, ys_set, nn)
    loss = constant(0.0)
    m = length(xs_set)

    for i_set = 1:m
        xs, ys = xs_set[i_set], ys_set[i_set]
        n = size(xs,1)
        y = constant(ys[1,:])
        for i = 2:n
            y = nn(constant(xs[i,:]), constant(xs[i-1,:]), y)
            #todo change the loss function
            #loss += (ys[i,:] - y)^2[1]

            loss += sum((ys[i,:] - y)^2)

            #loss += (ys[i,1] - y[1])^2
 
        end
    end
    return loss
end

function nn(x, x_, y_)
    ipt = reshape([x;x_;y_], 1, :)
    # @show ipt
    out = ae(ipt, [20,20,20,20,20,ky])
    squeeze(out)
end


@doc """
    generate data 
    m set of data, each is a serial sequency of n time
"""->
function sample(m = 2, n = 100)
    xs_set, ys_set = [], []

    # generate xs_set
    if model_type == "Plasticity"
        T = 0.1
        #
        t = np.linspace(0.0, T, n)
        A = 0.02
        xs = A * reshape(sin.(π*t/(T)), :, kx)
        push!(xs_set, xs)

        #
        if m >= 2
        t = np.linspace(0.0, T, n)
        A = 0.02
        xs = A * reshape(sin.(2.0*π*t/(T)), :, kx)
        push!(xs_set, xs)
        end

    else
        error("model_type ", model_type, " have not implemented yet ")
    end

    # generate ys_set array
    for i = 1:m
        y0 = zeros(ky) 
        ys = generate_data(xs_set[i], y0)
        push!(ys_set, ys)
    end

    return xs_set, ys_set 
end

function test(xs_set, ys_set, sess)
    m = length(xs_set)
    ys_pred_set = []
    for i_set = 1:m
        xs, ys = xs_set[i_set], ys_set[i_set]
        n = size(xs,1)
        ys_pred = zeros((n, ky))

        for i = 2:n
            ys_pred[i,:] = run(sess,nn(constant(xs[i,:]), constant(xs[i-1,:]), constant(ys_pred[i-1,:])))
            @show "in test ", i, ys_pred[i,:], " exact ", ys[i,:]
        end
        push!(ys_pred_set, ys_pred)
    end

    return ys_pred_set

end

function verify(sess)
    y_ = 0.5
    x = LinRange(-1,2,50)|>collect
    x, y = np.meshgrid(x, x)
    z = zero(x)
    p1 = placeholder(0.5)
    p2 = placeholder(0.5)
    yval = nn(p1,p2,constant(y_))
    for i = 1:50
        for j = 1:50
            z[i,j] = run(sess, yval, Dict(p1=>x[i,j], p2=>y[i,j]))
        end
    end
    mesh(x, y, z)
    mesh(x, y, hidden_function.(x,y,y_), color="orange",alpha=0.5)
end


kx = 1
ky = 2
model_type = "Plasticity"
m, n = 1, 100
xs_set, ys_set = sample(m, n)
sess = Session()
Random.seed!(2333)  
loss = compute_loss(xs_set, ys_set, nn)
init(sess)
BFGS!(sess, loss)
ys_pred_set = test(xs_set, ys_set, sess)
for i = 1:m
    plot(xs_set[i], ys_set[i])
    plot(xs_set[i], ys_pred_set[i])
end
# verify(sess)