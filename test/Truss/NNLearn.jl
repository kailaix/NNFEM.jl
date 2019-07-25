using ADCME

X, Y = prepare_strain_stress_data(domain)
x = constant(X)
y = squeeze(ae(x, [20,20,20,20,1], "nn"))

loss = sum((y-Y)^2)
variable_scope("nn") do
    global opt = AdamOptimizer().minimize(loss)
end

sess = Session(); init(sess)

for i = 1:10000
    l, _ = run(sess, [loss, opt])
    @show i,l
end
# BFGS(sess, loss, 1000)

out = run(sess, y)
close("all")
plot(X[:,2], out,"+", label="NN")
plot(X[:,2], Y, ".", label="Exact")
legend()

ADCME.save(sess, "learned_nn.mat")