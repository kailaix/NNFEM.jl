using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")
function sigmoid_(z)

    return 1.0 ./ (1.0 .+ exp.(-z))
  
end

H = x->sigmoid_((x-1)*100)

x = LinRange(0,2,1000)
y = H.(x)
plot(x, y, linewidth=4)
plot(ones(100), LinRange(0,1,100), "--", linewidth=4)
text(1.1, 0.1, "\$\\tilde\\sigma_Y\$", fontsize=20)

text(1.1, 0.6, "\$\\sigma_{\\mathrm{plasticity}}\$",  fontsize=20)
text(1.1, 0.5, "dominates",  fontsize=20)

text(0.1, 0.6, "\$\\sigma_{\\mathrm{elasticity}}\$", fontsize=20)
text(1.1, 0.5, "dominates",  fontsize=20)

xlabel("\$\\sigma\$")
ylabel("\$D(\\sigma, \\tilde\\sigma_Y)\$")
mpl.save("H.tex")