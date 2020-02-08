using PyPlot
using PyCall
#mpl = pyimport("tikzplotlib")
using JLD2


T = 0.2
NT = 200


tid = 5
@load "../Data/domain$tid.jld2" domain 

function ft(t)
    return  sin(2pi*t/T) * 1e9 * (0.2*tid + 0.8)
end
t = LinRange(0,T,NT+1)
f = ft.(t)
plot(t, f)
xlabel("\$t\$")
ylabel("Force")
# mpl.save("truss1d_force.tex")
savefig("truss1d_force.pdf")

close("all")
for i = 1:5
    text(i-1, 0.0005, string(i))
end
scatter(domain.nodes[:, 1], domain.nodes[:,2], color="red", label="Initial Location")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(domain.nodes[:, 1] + u, domain.nodes[:,2] + v, color="blue", label="Terminal Location")
arrow(domain.nodes[end, 1] + u[end], domain.nodes[end,2] + v[end], 0.5,0.0,
 width=0.0001, head_length=0.2)


xlim(-0.1,5.0)
xlabel("x")
gca().get_yaxis().set_visible(false)
legend()
# mpl.save("truss1d_loc.tex")
savefig("truss1d_loc.pdf")


close("all")

t = LinRange(0,T,NT+1)
u = hcat(domain.history["state"]...)
for i = 1:5
    plot(t, u[i,:], label="$i")
end
xlabel("\$t\$")
ylabel("Displacement")
legend()
# mpl.save("truss1d_disp.tex")
savefig("truss1d_disp.pdf")


