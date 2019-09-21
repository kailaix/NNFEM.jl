using ADCME
using PyPlot
using Random
using PyCall
np = pyimport("numpy")

kx = 1
ky = 2
function hidden_function(x, x_, y_, model_type = "Plasticity")
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
        @show i, ys[i,:]
    end
    ys 
end




function compute_loss(xs, ys, nn)
    loss = constant(0.0)
    n = size(xs,1)
    y = constant(ys[1,:])
    for i = 2:n
        y = nn(constant(xs[i,:]), constant(xs[i-1,:]), y)
        loss += (ys[i,:]-y)^2[1]
    end
    return loss
end

function nn(x, x_, y_)
    ipt = reshape([x;x_;y_], 1, :)
    # @show ipt
    out = ae(ipt, [20,20,20,20,20,ky])
    squeeze(out)
    # out = [sin(sum(x)^2+sum(x_)^2+sum(y_)^2); cos(sum(x)^2+sum(x_)^2+sum(y_)^2)]
end

function train!(sess, nn)
    Random.seed!(2333)
    xs, ys = sample(200)
    loss = compute_loss(xs, ys, nn)
    init(sess)
    BFGS!(sess, loss)
    # @show run(sess, loss)
    xs, ys
end

function sample(n = 100)
    # xs = rand(n, kx)
    # y0 = rand(ky)
    T = 0.1
    t = np.linspace(0.0, T, n + 1)
    A = 0.02
    xs = A * reshape(sin.(π*t/(T)), :, kx)
    y0 = zeros(ky) 
    ys = generate_data(xs, y0)
    return xs, ys 
end

function test(sess)
    n = 100
    xs, ys = sample(n)
    ys_pred = zeros(size(ys))
    
    for i = 2:n+1
        ys_pred[i,:] = run(sess,nn(constant(xs[i,:]), constant(xs[i-1,:]), constant(ys_pred[i-1,:])))
        @show i, ys_pred[i,:]
    end
    plot(xs[:,1], ys[:,1])
    plot(xs[:,1], ys_pred[:,1])

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

sess = Session()
train!(sess, nn)
test(sess)
# verify(sess)