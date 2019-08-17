using Revise
using NNFEM
using ForwardDiff

E11 = 1.0
E22 = 2.0
G12 = 3.0
C1 = 1.0 
C2 = 1.0

f = x -> PlaneStressIncompressibleRivlinSaundersJacobian(x[1], x[2], x[3], C1, C2)[1]
J0 = ForwardDiff.jacobian(f, [E11;E22;G12])

J = PlaneStressIncompressibleRivlinSaundersJacobian(x[1], x[2], x[3], C1, C2)[2]
@show J
@show J0


# using SymPy
# E11, E22, G12, C1, C2 = @vars E11 E22 G12 C1 C2
# E33 = (1.0/(1 + 2*E11 + G12 + 4*E11*E22 - G12*G12) - 1.0) /2.0
# E =[ E11 G12 0
#      G12 E22 0 
#      0   0   E33]
# lambda = eigvals(2*E+SymPy.I).^2
# I1 = simplify(sum(lambda))
# I2 = simplify(lambda[1]*lambda[2] + lambda[1]*lambda[3] + lambda[2]*lambda[3])

# W = C1*(I1 - 3) + C2*(I2 - 3)
# σ = Array{Sym}(undef,3)
# σ[1] = simplify(diff(W, E11))
# σ[2] = simplify(diff(W, E22))
# σ[3] = simplify(diff(W, G12))
# J = Matrix{Sym}(undef, 3, 3)
# for i = 1:3
#     J[i,1] = simplify(diff(σ[i], E11))
#     J[i,2] = simplify(diff(σ[i], E22))
#     J[i,3] = simplify(diff(σ[i], G12))
# end
# s = "J = zeros(3,3);\n"
# for i = 1:3
#     for j = 1:3
#         global s *= "J[$i,$j]="*sympy.julia_code(J[i,j])*"\n"
#     end
# end
# t = "σ = zeros(3)\n"
# for i = 1:3
#     global t *= "σ[$i]="*sympy.julia_code(σ[i])*"\n"
# end

