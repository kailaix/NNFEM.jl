export compute_stress_rivlin_saunders, compute_maxwell_viscoelasticity_matrices

@doc raw""" 
    compute_stress_rivlin_saunders(strain::Union{PyObject, Array{Float64,2}},c1::Union{PyObject, Float64},c2::Union{PyObject, Float64})

Computes the stress using the plane stress incompressible Rivlin Saunders model. 
"""
function compute_stress_rivlin_saunders(strain::Union{PyObject, Array{Float64,2}},c1::Union{PyObject, Float64},c2::Union{PyObject, Float64})
    rivlin_saunders_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/RivlinSaunders/build/libRivlinSaunders","rivlin_saunders")
    strain,c1,c2 = convert_to_tensor([strain,c1,c2], [Float64,Float64,Float64])
    rivlin_saunders_(strain,c1,c2)
end

@doc raw"""
    compute_maxwell_viscoelasticity_matrices(mu::Union{PyObject, Array{Float64,1}},eta::Union{PyObject, Array{Float64,1}},lambda::Union{PyObject, Array{Float64,1}},dt::Union{PyObject, Float64})

Computes the viscoelasticity matrix in the Maxwell model. `mu`, `lambda` are Lamé parameters, and `eta` is the viscosity parameter.
These three inputs are all length $N$ vectors, where $N$ is the total number of Gauss points. 
`dt` is the  time step. 

The time advancing formula is 

$$\sigma^{n+1} = H(\epsilon^{n+1}-\epsilon^n) + S\sigma^n$$

This function returns `S`, `H`, and both are $N\times 3\times 3$ tensors.
"""
function compute_maxwell_viscoelasticity_matrices(mu::Union{PyObject, Array{Float64,1}},lambda::Union{PyObject, Array{Float64,1}},eta::Union{PyObject, Array{Float64,1}},dt::Union{PyObject, Float64})
    viscoelasticity_matrices_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/ViscoelasticityMatrices/build/libViscoelasticityMatrices","viscoelasticity_matrices", multiple=true)
    mu,eta,lambda,dt = convert_to_tensor(Any[mu,eta,lambda,dt], [Float64,Float64,Float64,Float64])
    viscoelasticity_matrices_(mu,eta,lambda,dt)
end