
ε = [100.0;200.0;1.0]
ε0 = [0.0;0.0;0.0]
σ0 = [0.0;0.0;0.0]
α0 = 0.0
H = UniformScaling(1.0)
σY = 1
K = 1



prop = Dict("name"=> "PlaneStressPlasticity", "rho"=> 80.0, "E"=> 200, "nu"=> 0.45,
"sigmaY"=>300, "K"=>1)
matlaw = PlaneStressPlasticity(prop)
Dstate = ones(3)

function f1(x)
    getStress(matlaw, x, Dstate)
end

gradtest(f1, 2ones(3))
# f1(ones(3))