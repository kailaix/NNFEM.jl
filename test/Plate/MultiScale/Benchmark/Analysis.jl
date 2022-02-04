using PyPlot
using PyCall
mpl = pyimport("tikzplotlib")
using DelimitedFiles
using MAT

force_scales = [5.0, 6.0, 8.0, 10.0, 20.0]
tids = [106, 206, 300]

Multiscale = zeros(5,3)
NN = zeros(5,3)

for (k1, force_scale) in enumerate(force_scales)
    for (k2, tid) in enumerate(tids)
        Multiscale[k1, k2] = readdlm("Multiscale$(force_scale)$tid.txt")[2]
        NN[k1, k2] = readdlm("NN$(force_scale)$tid.txt")[2]
    end
end

matwrite("cpu.mat", Dict("Multiscale"=>Multiscale, "NN"=>NN))

close("all")

cols = "rgb"
for i = 1:3
    semilogy(8 ./ force_scales, Multiscale[:,i], "o$(cols[i])-", linewidth=3)
    semilogy(8 ./ force_scales, NN[:,i], "o$(cols[i])", linestyle="dotted", linewidth=3)
end

a = ones(5); fill!(a, NaN)
semilogy(8 ./ force_scales, a, "k-", label="Fiber resolved DNS", linewidth=3)
semilogy(8 ./ force_scales, a, "k", linestyle="dotted",  label="CholNN", linewidth=3)

legend()
xlabel("Load (GPa)")
ylabel("CPU Time (sec)")
grid("on")
savefig("cpu.pdf")
savefig("cpu.png")
mpl.save("cpu.tex")




