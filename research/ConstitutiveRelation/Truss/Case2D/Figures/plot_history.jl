using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")
using JLD2
using NNFEM


tid = 1
@load "../Data/domain$tid.jld2" domain 

nodes = domain.nodes

plot_line = (i, j)->begin
    x = [nodes[i,1]+u[i];nodes[j,1]+u[j]]
    y = [nodes[i,2]+v[i];nodes[j,2]+v[j]]
    plot(x, y, color="grey",alpha=0.5)
end

close("all")
scatter(nodes[:, 1], nodes[:,2], color="red")
u,v = domain.state[1:domain.nnodes], domain.state[domain.nnodes+1:end]
scatter(nodes[:, 1] + u, nodes[:,2] + v, color="blue")
for i = 1:3
    plot_line(2i, 2i+2)
    plot_line(2i, 2i-1)
    plot_line(2i, 2i+1)
    plot_line(2i-1, 2i+1)
    plot_line(2i-1, 2i+2)
    plot_line(2i+1, 2i+2)
end
ylim(-0.2,1.5)
xlabel("x")
ylabel("y")
# mpl.save("truss2d_loc.tex")
savefig("truss2d_loc.pdf")


close("all")
X, Y = prepare_strain_stress_data1D(domain)
scatter(X[:,1], Y, marker=".",s=5)
xlabel("Strain")
ylabel("Stress")
grid("on")
# mpl.save("truss2d_stress.tex")
savefig("truss2d_stress.pdf")