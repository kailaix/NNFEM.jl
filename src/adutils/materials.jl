export compute_stress_rivlin_saunders

@doc raw""" 
    compute_stress_rivlin_saunders(strain::Union{PyObject, Array{Float64,2}},c1::Union{PyObject, Float64},c2::Union{PyObject, Float64})

Computes the stress using the plane stress incompressible Rivlin Saunders model. 
"""
function compute_stress_rivlin_saunders(strain::Union{PyObject, Array{Float64,2}},c1::Union{PyObject, Float64},c2::Union{PyObject, Float64})
    rivlin_saunders_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/RivlinSaunders/build/libRivlinSaunders","rivlin_saunders")
    strain,c1,c2 = convert_to_tensor([strain,c1,c2], [Float64,Float64,Float64])
    rivlin_saunders_(strain,c1,c2)
end