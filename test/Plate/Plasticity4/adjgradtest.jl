include("CommonFuncs.jl")
tid = 200
porder = 2

prop = Dict("name"=> "PlaneStressPlasticity","rho"=> 4.5, "E"=> 1e+6, "nu"=> 0.2,
"sigmaY"=>0.97e+4, "K"=>1e+5)

np = pyimport("numpy")


nx, ny = 10,5

nodes, EBC, g, gt, FBC, fext, ft = BoundaryCondition(tid, nx, ny, porder)
elements = []
for j = 1:ny
    for i = 1:nx 
        n = (nx*porder+1)*(j-1)*porder + (i-1)porder+1
        #element (i,j)
        if porder == 1
            #   4 ---- 3
            #
            #   1 ---- 2

            elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]
        elseif porder == 2
            #   4 --7-- 3
            #   8   9   6 
            #   1 --5-- 2
            elnodes = [n, n + 2, n + 2 + 2*(2*nx+1),  n + 2*(2*nx+1), n+1, n + 2 + (2*nx+1), n + 1 + 2*(2*nx+1), n + (2*nx+1), n+1+(2*nx+1)]
        else
            error("polynomial order error, porder= ", porder)
        end

        coords = nodes[elnodes,:]
        push!(elements,SmallStrainContinuum(coords,elnodes, prop, 3))
    end
end


ndofs = 2
domain = Domain(nodes, elements, ndofs, EBC, g, FBC, fext)
state = zeros(domain.neqs)
∂u = zeros(domain.neqs)


neqs = domain.neqs






# test adjointAssembleStrain(domain::Domain)


function f1(u)
    domain.state[domain.eq_to_dof] = u
    strain, dstrain_dstate_tran = AdjointAssembleStrain(domain)
    strain'[:], dstrain_dstate_tran'
end


u = rand(neqs)
#gradtest(f1, u)



neles = domain.neles
ngps_per_elem = length(domain.elements[1].weights)

stress = zeros(ngps_per_elem*neles, 3)
dstress_dstrain = zeros(ngps_per_elem*neles, 3, 3)
mat = PlaneStress(prop)




# test AssembleStiffAndForce(domain, stress::Array{Float64}, dstress_dstrain::Array{Float64})
function f2(u)
    domain.state[domain.eq_to_dof] = u

    strain, dstrain_dstate_tran = AdjointAssembleStrain(domain)
    for i = 1:ngps_per_elem*neles
        stress[i,:], dstress_dstrain[i,:,:] = getStress(mat, strain[i,:], strain[i,:], 1.0)
    end
    
    fint, stiff = AssembleStiffAndForce(domain, stress, dstress_dstrain)
    fint, stiff
end

u = rand(neqs)
#err1, err2 = gradtest(f2, u); @show err1, err2






# test stiff_tran, dfint_dstress_tran = adjointAssembleStiff(domain, stress::Array{Float64}, dstress_dstrain::Array{Float64})

domain.state[domain.eq_to_dof] = u
strain, dstrain_dstate_tran = AdjointAssembleStrain(domain)
for i = 1:ngps_per_elem*neles
    stress[i,:], dstress_dstrain[i,:,:] = getStress(mat, strain[i,:], strain[i,:], 1.0)
end

fint, stiff = AssembleStiffAndForce(domain, stress, dstress_dstrain)
stiff_tran, dfint_dstress_tran = AdjointAssembleStiff(domain, stress, dstress_dstrain)
@show norm(stiff - stiff_tran')

function f3(stress)
    stress = reshape(stress, 3, ngps_per_elem*neles)'|>Array
    domain.state[domain.eq_to_dof] = u
    fint, _ = AssembleStiffAndForce(domain, stress, dstress_dstrain)
    _, dfint_dstress_tran = AdjointAssembleStiff(domain, stress, dstress_dstrain)
    fint, dfint_dstress_tran'
end

stress = rand(ngps_per_elem*neles*3)
gradtest(f3, stress)



