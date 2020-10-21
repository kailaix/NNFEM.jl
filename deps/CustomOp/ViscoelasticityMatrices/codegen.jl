using SymPy

mu, eta, dt, lambda = @vars mu eta dt lambda 
S = [
    1+2/3*mu*dt/eta -1/3*mu*dt/eta 0.0
    -1/3*mu*dt/eta 1+2/3*mu*dt/eta 0.0
    0.0 0.0 1+mu*dt/eta
]
invS = inv(S)
H = invS * [
    2*mu+lambda lambda 0.0
    lambda 2mu+lambda 0.0
    0.0 0.0 mu
]
S = simplify(invS)
H = simplify(H)

k = -1
s = ""
for i = 1:3
    for j = 1:3
        global k += 1
        c = sympy.ccode(S[i,j])
        global s *= "s[9*i+$k] = $c;\n"
    end
end
s = s*"\n\n"

k = -1
for i = 1:3
    for j = 1:3
        global k += 1
        c = sympy.ccode(H[i,j])
        global s *= "h[9*i+$k] = $c;\n"
    end
end

println(s)





gs11, gs12, gs13, gs21, gs22, gs23, gs31, gs32, gs33 = @vars gs11 gs12 gs13 gs21 gs22 gs23 gs31 gs32 gs33
gs = [
    gs11 gs12 gs13
    gs21 gs22 gs23
    gs31 gs32 gs33
]

hs11, hs12, hs13, hs21, hs22, hs23, hs31, hs32, hs33 = @vars hs11 hs12 hs13 hs21 hs22 hs23 hs31 hs32 hs33
hs = [
    hs11 hs12 hs13
    hs21 hs22 hs23
    hs31 hs32 hs33
]

a = sum(S .* gs)
gmu = diff(a, mu)|>simplify
geta = diff(a, eta)|>simplify
glambda = diff(a, lambda)|>simplify

s = ""
k = -1
for i = 1:3
    for j = 1:3
        global k += 1
        global s *= "double gs$i$j = grad_s[9*i+$k];\n"
    end
end

s *= "grad_mu_all[i] += "*sympy.ccode(gmu)*";\n"
s *= "grad_eta_all[i] += "*sympy.ccode(geta)*";\n"
s *= "grad_lambda_all[i] += "*sympy.ccode(glambda)*";\n\n"




a = sum(H .* hs)
gmu = diff(a, mu)|>simplify
geta = diff(a, eta)|>simplify
glambda = diff(a, lambda)|>simplify

k = -1
for i = 1:3
    for j = 1:3
        global k += 1
        global s *= "double hs$i$j = grad_h[9*i+$k];\n"
    end
end

s *= "grad_mu_all[i] += "*sympy.ccode(gmu)*";\n"
s *= "grad_eta_all[i] += "*sympy.ccode(geta)*";\n"
s *= "grad_lambda_all[i] += "*sympy.ccode(glambda)*";\n"

println(s)