using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")
function sigmoid_(z)

    return 1.0 ./ (1.0 .+ exp.(-z))
  
end

H = x->sigmoid_(-(x-1)*100)

x = LinRange(0,2,1000)
y = H.(x)
plot(x, y)
plot(ones(100), LinRange(0,1,100), "--")
text(1.1, 0.1, "\$\\sigma_0\$", fontsize=20)

text(1.1, 0.6, "Plasticity",  fontsize=20)
text(0.1, 0.6, "Elasticity", fontsize=20)
xlabel("\$\\sigma\$")
ylabel("\$H(\\sigma, \\sigma_0)\$")
mpl.save("H.tex")