using PyPlot
using PyCall
using ADCME
using DifferentialEquations


function generate_data(n, Δt)
    function lorenz(du,u,p,t)
        du[1] = sin(u[2])
        du[2] = 2*cos(u[1])
    end
    
    t = n*Δt
    u0 = [0.0;0.0]
    tspan = (0,t)
    prob = ODEProblem(lorenz,u0,tspan)
    sol = DifferentialEquations.solve(prob)
    sol = sol( (0:n-1)*Δt )
    hcat(sol.u...)[1,:], hcat(sol.u...)[2,:], collect((0:n-1)*Δt)
end

function nn1(t0, t1, x, y)
    ipt = reshape([t0;t1;x;y], 1,:)
    out = ae(ipt, [20,2], "nn1")
    out[1,1], out[1,2]
end

function nn2(t0, t1, x, y)
    ipt = reshape([t0;t1;x;y], 1,:)
    out = ae(ipt,[20,2], "nn2")
    out = reshape([x;y],1,2) + (t1-t0)*out
    out[1,1], out[1,2]
end

function loss(xset, yset, tset, nn)
    n = size(xset,1)
    m = size(xset,2)
    loss = constant(0.0)
    for i = 1:m 
        t = tset[:,i]
        x = xset[:,i]
        y = yset[:,i]
        for j = 2:n
            @show j
            x_, y_ = nn(t[j-1], t[j], x[j-1], y[j-1])
            loss += (x[j]-x_)^2+(y[j]-y_)^2
        end
    end
    loss
end

function predict(xset, yset, tset, nn)
    t = convert_to_tensor(tset)
    x = convert_to_tensor(xset)
    y = convert_to_tensor(yset)
    resx = [convert_to_tensor(x[1])]
    resy = [convert_to_tensor(y[1])]
    x_, y_ = x[1], y[1]
    for j = 2:length(tset)
        x_, y_ = nn(t[j-1], t[j], x_, y_)
        push!(resx, x_)
        push!(resy, y_)
    end
    vcat(resx...), vcat(resy...)
end
reset_default_graph()
x1, y1, t1 = generate_data(200, 0.15)
x2, y2, t2 = generate_data(200, 0.075)
y1 = y1 .* ( 1 .+ 0.01*randn(length(y1)))
y2 = y2 .* ( 1 .+ 0.01*randn(length(y2)))
# l1 = loss([x1 x2], [y1 y2], [t1 t2], nn1)
# l2 = loss([x1 x2], [y1 y2], [t1 t2], nn2)

l1 = loss(x1, y1, t1, nn1)
l2 = loss(x1, y1, t1, nn2)

v1 = predict(x1, y1, t1, nn1)
v2 = predict(x1, y1, t1, nn2)
v3 = predict(x2, y2, t2, nn1)
v4 = predict(x2, y2, t2, nn2)

sess = Session(); ADCME.init(sess)
BFGS!(sess, l1)
BFGS!(sess, l2)

close("all")
figure(figsize=(12,5))

subplot(121)
pred_x1, pred_y1 = run(sess,v1[1]), run(sess, v1[2])
pred_x2, pred_y2 = run(sess,v2[1]), run(sess, v2[2])
plot(pred_x1, pred_y1, "*-", label="Direct")
plot(pred_x2, pred_y2, "+-", label="Derivative")
plot(x1, y1,"--", label="Exact")
legend()
axis("equal")
title("Δt = $(t1[2]-t1[1])")

subplot(122)
pred_x1, pred_y1 = run(sess,v3[1]), run(sess, v3[2])
pred_x2, pred_y2 = run(sess,v4[1]), run(sess, v4[2])
plot(pred_x1, pred_y1, "*-", label="Direct")
plot(pred_x2, pred_y2, "+-", label="Derivative")
plot(x2, y2,"--", label="Exact")
legend()
axis("equal")
title("Δt = $(t2[2]-t2[1])")


