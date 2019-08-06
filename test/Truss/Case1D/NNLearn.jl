using ADCME

X, Y = prepare_strain_stress_data1D(domain)
x = constant(X)
y = squeeze(ae(x, [20,20,20,20,1], "nn"))

loss = sum((y-Y)^2)
variable_scope("nn") do
    global opt = AdamOptimizer().minimize(loss)
end

sess = Session(); init(sess)

for i = 1:50000
    l, _ = run(sess, [loss, opt])
    @show i,l
end
# BFGS!((sess, loss, 1000)

out = run(sess, y)
close("all")
plot(X[:,2], out,"+", label="NN")
plot(X[:,2], Y, ".", label="Exact")
legend()

ADCME.save(sess, "Data/learned_nn.mat")


# for α = [0.01,0.05,0.1]
# x = constant(X.*(1 .+ α*rand(size(X)...)))
# y = squeeze(ae(x, [20,20,20,20,1], "nn"))
# out = run(sess, y)
# plot(X[:,2], out,".", label="NN")
# end
# # plot(X[:,2], Y, ".", label="Exact")
# legend()

@load "Data/domain.jld2" domain
X, Y = prepare_strain_stress_data1D(domain)
x = constant(X)
y = squeeze(ae(x, [20,20,20,20,1], "nn"))
sess = Session(); init(sess)
close("all")
ADCME.load(sess, "Data/learned_nn.mat")
out = run(sess, y)
plot(X[:,1], out,"+", label="NN")
plot(X[:,1], Y, ".", label="Exact")
legend()
