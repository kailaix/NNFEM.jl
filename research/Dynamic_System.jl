using ADCME
using PyPlot
using Random
using PyCall
np = pyimport("numpy")

kx = 1
ky = 2
function hidden_function(x, x_, y_)
    y = [(sum(x)+sum(x_)+sum(y_))/4; (sum(x)+sum(x_)+sum(y_))/4]
    return y
end

function generate_data(xs, y0)
    n = size(xs,1)
    ys = zeros(n,ky)
    ys[1,:] = y0
    for i = 2:n 
        ys[i,:] = hidden_function(xs[i,:], xs[i-1,:], ys[i-1,:])
    end
    ys 
end

function sample(n)
    xs = rand(n, kx)
    y0 = rand(ky)
    ys = generate_data(xs, y0)
    return xs, ys 
end


function compute_loss(xs, ys, nn)
    loss = constant(0.0)
    n = size(xs,1)
    y = constant(ys[1,:])
    for i = 2:n
        y = nn(constant(xs[i,:]), constant(xs[i-1,:]), y)
        loss += sum((ys[i,:]-y)^2)
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
# verify(sess)