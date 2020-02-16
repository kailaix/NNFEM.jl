using Revise
using ADCME
using NNFEM
using JLD2
using PyPlot
using MAT 
using DelimitedFiles
reset_default_graph()

include("nnutil.jl")

loss = constant(0.0)
for tid = [1,2,4,5]
    strain, stress = read_strain_stress("Data/$(tid).dat")
    X, Y = prepare_strain_stress_data1D(strain, stress )
    y = squeeze(nn(constant(X[:,1:1]), constant(X[:,2:2]), constant(X[:,3:3])))
    global loss += sum((y - Y)^2)
end
sess = Session(); init(sess)
@show run(sess,loss)

loss_all = []
for i = 1:10
    println("************************** Outer Iteration = $i ************************** ")
    loss_ = BFGS!(sess, loss, 3000)
    push!(loss_all, loss_)
end
# ADCME.save(sess, "Data/$FILEID.mat")
# loss_ = vcat(loss_all...)
# writedlm("Data/$FILEID.txt", loss_)


strain, stress = read_strain_stress("Data/3.dat")
X, Y = prepare_strain_stress_data1D(strain, stress )
y = squeeze(nn(constant(X[:,1:1]), constant(X[:,2:2]), constant(X[:,3:3])))
testerror = sqrt(mean((y-Y)^2))

testerror_ = run(sess, testerror)
open("Data/$FILEID2.txt", "a") do io 
    writedlm(io, [testerror_])
end
