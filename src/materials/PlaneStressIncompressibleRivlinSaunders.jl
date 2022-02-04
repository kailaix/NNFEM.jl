export  PlaneStressIncompressibleRivlinSaunders, getStress

@doc """
Pascon, João Paulo. 
"Large deformation analysis of plane-stress hyperelastic problems via triangular membrane finite elements." 
International Journal of Advanced Structural Engineering (2019): 1-20.
"""->
mutable struct PlaneStressIncompressibleRivlinSaunders
    ρ::Float64 # density
    C1::Float64 # Young's modulus
    C2::Float64 # Poisson's ratio
    
    # The classical incompressible PlaneStressIncompressibleRivlinSaunders strain energy function
    # I3 = 0
    # W = C1(I1 - 3) + C2(I2 - 2)

    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} # stress at last time step
    ε0_::Array{Float64} # σ0 to be updated in `commitHistory`
end

function PlaneStressIncompressibleRivlinSaundersJacobian(E11::T, E22::T, G12::T, C1::S, C2::S) where {S<:Number, T<:Number}
    f = x->PlaneStressIncompressibleRivlinSaundersStress(x[1],x[2],x[3], C1, C2)
    J = ForwardDiff.jacobian(f, [E11;E22;G12])
    σ = f([E11;E22;G12])
    σ, J
end

function PlaneStressIncompressibleRivlinSaundersStress(E11::T, E22::T, G12::T, C1::S, C2::S) where {S<:Number, T<:Number}
    C11, C22, C12 = 2*E11+1.0, 2*E22+1.0, G12
    det22C = C11*C22 - C12*C12
    C33 = 1/det22C

    I1 = C11 + C22 + C33
    
    C = [ C11    C12      0
          C12    C22      0
          0       0     C33]

    Cinv = [ C22/det22C    -C12/det22C        0
            -C12/det22C     C11/det22C        0
                  0             0         1.0/C33]
    svol = Cinv
    Siso = 2.0*C1*UniformScaling(1.0) + 2.0*C2*(C + UniformScaling(I1))

    p = -C33*Siso[3,3]

    σ = zeros(T, 3)
    σ[1] = Siso[1,1] + p*svol[1,1]
    σ[2] = Siso[2,2] + p*svol[2,2]
    σ[3] = Siso[1,2] + p*svol[1,2]
    return σ
end



function PlaneStressIncompressibleRivlinSaunders(prop::Dict{String, Any})
    C1 = prop["C1"]; C2 = prop["C2"]; ρ = prop["rho"];
    σ0 = zeros(3); σ0_ = zeros(3); ε0 = zeros(3); ε0_ = zeros(3)
    PlaneStressIncompressibleRivlinSaunders(ρ, C1, C2, σ0, σ0_, ε0, ε0_)
end

function getStress(self::PlaneStressIncompressibleRivlinSaunders,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    local dΔσdΔε
    C1, C2 = self.C1, self.C2
    E11, E22, G12 = strain[1], strain[2], strain[3]
    σ, dΔσdΔε = PlaneStressIncompressibleRivlinSaundersJacobian(E11, E22, G12, C1, C2)

    self.σ0_ = σ[:]
    self.ε0_ = strain

    #@show σ, dΔσdΔε
    return σ, dΔσdΔε
end

function getTangent(self::PlaneStressIncompressibleRivlinSaunders)
    error("Not implemented")
end

function commitHistory(self::PlaneStressIncompressibleRivlinSaunders)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
end





# # W = C1(I1 - 3) + C2(I2 - 2)
# #
# # E11, E22, G12 = strain[1], strain[2], strain[3]
# # E12 = G12/2.0
# #
# #      E11 E12 0
# # E =  E21 E22 0 
# #      0   0   E33
# # incompressibility 
# # det(2E + 1) = det(C) = det(F^T F) = 1  
# # E33 = (1.0/(1 + 2*E11 + G12 + 4*E11*E22 - G12*G12) - 1.0) /2.0
# # C = 2E + 1
# # I1(C) = lambda1 + lambda2 + lambda3 = 3 + 2(E11 + E22 + E33)
# # I2(C) = lambda1*lambda2 + lambda2*lambda3 + lambda3*lambda1 
# #       = (1 + 2E11)*(1 + 2E22) + (1 + 2E22)*(1 + 2E33) + (1 + 2E33)*(1 + 2E11) - G12 * G21

# using SymPy
# E11, E22, G12, C1, C2 = @vars E11 E22 G12 C1 C2
# E33 = (1.0/(1 + 2*E11 + G12 + 4*E11*E22 - G12*G12) - 1.0) /2.0
# I1 = 3 + 2*(E11 + E22 + E33)
# I2 = 3 + 4*(E11 + E22 + E33) + 4*E11*E22 + 4*E11*E22 + 4*E11*E33 - G12*G12
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



