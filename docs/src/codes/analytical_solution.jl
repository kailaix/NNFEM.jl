using SymPy

x, y, t = @syms x y t
u1 = (x^2+y^2)*exp(-t)*0.1
u2 = (x^2-y^2)*exp(-t)*0.1
ε = [
    diff(u1, x)
    diff(u2, y)
    diff(u2, x) + diff(u1, y)
]

ν = 0.35
E = 2.0

H = zeros(3,3)
H[1,1] = E*(1. -ν)/((1+ν)*(1. -2. *ν));
H[1,2] = H[1,1]*ν/(1-ν);
H[2,1] = H[1,2];
H[2,2] = H[1,1];
H[3,3] = H[1,1]*0.5*(1. -2. *ν)/(1. -ν);

σ = H * ε
f1 = u1 - ( diff(σ[1],x) + diff(σ[3], y) ) 
f2 = u2 - ( diff(σ[2],y) + diff(σ[3], x) )

g1 = -σ[3]
g2 = -σ[2]

function code(s)
    s = sympy.julia_code(s)
    replace(replace(s, ".*"=>"*"), ".^"=>"^")
end

println(code(f1))
println(code(f2))
println(code(g1))
println(code(g2))

