export  PlaneStressMooneyRivlin, getStress

function PlaneStressMooneyRivlinJacobian(E11::Float64, E22::Float64, E12::Float64, C1::Float64, C2::Float64)

    σ = zeros(3)
    σ[1]=(-2*C1*(2*E22 - (4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 1) + C2*(8.0*E11*E22 - 4.0*E11*(2*E22 + 1) + 4.0*E11 - 8.0*E12^2 + 4.0*E12 - 8.0*E22 + (8*E22 + 2.0)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 - 2.0))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2
    σ[2]=-(2*C1*(2*E11 - (4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2) + C2*(8.0*E11^2 + 8.0*E11 - (8*E11 + 4)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2
    σ[3]=(2*C1*(4*E12 - 1) + C2*(4.0*E11*(4*E12 - 1) - 8*E12*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 16.0*E12 - 4.0))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2
    
    J = zeros(3,3);
    J[1,1]=4*(2*E22 + 1)*(2*C1*(2*E22 - (4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 1) - C2*(8.0*E11*E22 - 4.0*E11*(2*E22 + 1) + 4.0*E11 - 8.0*E12^2 + 4.0*E12 - 8.0*E22 + (8*E22 + 2.0)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 - 2.0) + (2*C1 + C2*(8*E22 + 2.0))*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2)/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    J[1,2]=(8*E11*(2*C1*(2*E22 - (4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 1) - C2*(8.0*E11*E22 - 4.0*E11*(2*E22 + 1) + 4.0*E11 - 8.0*E12^2 + 4.0*E12 - 8.0*E22 + (8*E22 + 2.0)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 - 2.0)) + (4*C1*(4*E11*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) - 1) + C2*(8*E11*(8*E22 + 2.0)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) + 8*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 - 8.0))*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    J[1,3]=-(4*(4*E12 - 1)*(2*C1*(2*E22 - (4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 1) - C2*(8.0*E11*E22 - 4.0*E11*(2*E22 + 1) + 4.0*E11 - 8.0*E12^2 + 4.0*E12 - 8.0*E22 + (8*E22 + 2.0)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 - 2.0)) + (8*C1*(4*E12 - 1)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) + C2*(16.0*E12 + 4*(4*E12 - 1)*(8*E22 + 2.0)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) - 4.0))*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    J[2,1]=(4*(2*E22 + 1)*(2*C1*(2*E11 - (4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2) + C2*(8.0*E11^2 + 8.0*E11 + (-8*E11 - 4)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2)) + (4*C1*(2*(2*E22 + 1)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) - 1) - C2*(16.0*E11 - 16*(2*E11 + 1)*(2*E22 + 1)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) - 8*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 8.0))*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    J[2,2]=32*C1*E11^2/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3 + 64.0*C2*E11^3/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3 + 64.0*C2*E11^2/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    J[2,3]=-4*(4*E12 - 1)*(2*C1*(2*E11 - (4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2) + C2*(8.0*E11^2 + 8.0*E11 - (8*E11 + 4)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2) + (2*C1 + 4*C2*(2*E11 + 1))*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2)/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    J[3,1]=-(C2*(32*E12*(2*E22 + 1)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) - 16.0*E12 + 4.0)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) + 4*(2*E22 + 1)*(2*C1*(4*E12 - 1) + C2*(4.0*E11*(4*E12 - 1) - 8*E12*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 16.0*E12 - 4.0)))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    J[3,2]=-8*E11*(2*C1*(4*E12 - 1) + 8*C2*E12*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + C2*(4.0*E11*(4*E12 - 1) - 8*E12*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 16.0*E12 - 4.0))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    J[3,3]=((8*C1 + C2*(16.0*E11 + 32*E12*(4*E12 - 1)*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) - 8*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 16.0))*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1) + 4*(4*E12 - 1)*(2*C1*(4*E12 - 1) + C2*(4.0*E11*(4*E12 - 1) - 8*E12*(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^2 + 16.0*E12 - 4.0)))/(4*E11*E22 + 2*E11 - 4*E12^2 + 2*E12 + 1)^3
    
    return σ, J

end
mutable struct PlaneStressMooneyRivlin
    ρ::Float64 # density
    C1::Float64 # Young's modulus
    C2::Float64 # Poisson's ratio
    
    # The classical incompressible Mooney-Rivlin strain energy function
    # I3 = 0
    # W = C1(I1 - 3) + C2(I2 - 2)

    σ0::Array{Float64} # stress at last time step
    σ0_::Array{Float64} # σ0 to be updated in `commitHistory`
    ε0::Array{Float64} # stress at last time step
    ε0_::Array{Float64} # σ0 to be updated in `commitHistory`
end



function PlaneStressMooneyRivlin(prop::Dict{String, Any})
    C1 = prop["C1"]; C2 = prop["C2"]; ρ = prop["rho"];
    σ0 = zeros(3); σ0_ = zeros(3); ε0 = zeros(3); ε0_ = zeros(3)
    PlaneStressMooneyRivlin(ρ, C1, C2, σ0, σ0_, ε0, ε0_)
end

function getStress(self::PlaneStressMooneyRivlin,  strain::Array{Float64},  Dstrain::Array{Float64}, Δt::Float64 = 0.0)
    local dΔσdΔε
    C1, C2 = self.C1, self.C2
    E11, E22, E12 = strain[1], strain[2], strain[3]/2.0
    σ, dΔσdΔε = PlaneStressMooneyRivlinJacobian(E11, E22, E12, C1, C2)
    # #@show Δγ
    self.σ0_ = σ[:]
    self.ε0_ = strain
    return σ, dΔσdΔε
end

function getTangent(self::PlaneStressMooneyRivlin)
    error("Not implemented")
end

function commitHistory(self::PlaneStressMooneyRivlin)
    self.σ0 = self.σ0_
    self.ε0 = self.ε0_
end


function PlaneStressMooneyRivlinJacobian(E11::Float64, E22::Float64, E12::Float64, E33::Float64,
    C1::Float64, C2::Float64)
    
end



# # W = C1(I1 - 3) + C2(I2 - 2)
# #
# # E11, E22, E12 = strain[1], strain[2], strain[3]/2.0
# #
# #      E11 E12 0
# # E =  E21 E22 0 
# #      0   0   E33
# # incompressibility 
# # det(2E + 1) = det(C) = det(F^T F) = 1  
# # E33 = (1.0/(1 + 2*E11 + 2*E12 + 4*E11*E22 - 4*E12*E12) - 1.0) /2.0
# # C = 2E + 1
# # I1(C) = lambda1 + lambda2 + lambda3 = 3 + 2(E11 + E22 + E33)
# # I2(C) = lambda1*lambda2 + lambda2*lambda3 + lambda3*lambda1 
# #       = (1 + 2E11)*(1 + 2E22) + (1 + 2E22)*(1 + 2E33) + (1 + 2E33)*(1 + 2E11) - 2E12 * 2E21

# using SymPy
# E11, E22, E12,C1,C2 = @vars E11 E22 E12 C1 C2
# E33 = (1.0/(1 + 2*E11 + 2*E12 + 4*E11*E22 - 4*E12*E12) - 1.0) /2.0
# I1 = 3 + 2*(E11 + E22 + E33)
# I2 = 3 + 4*(E11 + E22 + E33) + 4*E11*E22 + 4*E11*E22 + 4*E11*E33 - 4*E12*E12
# W = C1*(I1 - 3) + C2*(I2 - 3)
# σ = Array{Sym}(undef,3)
# σ[1] = simplify(diff(W, E11))
# σ[2] = simplify(diff(W, E22))
# σ[3] = simplify(diff(W, E12))
# J = Matrix{Sym}(undef, 3, 3)
# for i = 1:3
#     J[i,1] = simplify(diff(σ[i], E11))
#     J[i,2] = simplify(diff(σ[i], E22))
#     J[i,3] = simplify(diff(σ[i], E12))
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
