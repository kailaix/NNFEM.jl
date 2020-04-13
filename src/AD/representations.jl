export consistent_tangent_matrix
@doc raw"""
    consistent_tangent_matrix(inputs::Union{Array{Float64, 2}, PyObject},Dc::Union{Array{Float64,2}, PyObject})

Returns the consistent tangent matrices. The size of the return is $N\times 3 \times 3$. 

$$ D_c^{ep} = D_c - \frac{D_c\frac{\partial g}{\partial \sigma}\left(\frac{\partial f}{\partial \sigma}Dc \right)^T D_c }{E_p + \frac{\partial g}{\partial \sigma} Dc \left(\frac{\partial f}{\partial \sigma}Dc \right)^T} $$

Here `inputs` is a $N\times 7$ matrix, where each row is 

$$\left[\frac{\partial g}{\partial \sigma}, \frac{\partial f}{\partial \sigma}, E_p\right]$$

`H` is a $3\times 3$ **row-major** matrix. 
"""
function consistent_tangent_matrix(inputs::Union{Array{Float64, 2}, PyObject},H::Union{Array{Float64,2}, PyObject})
    plasticity_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/Plasticity/build/libPlasticity","plasticity")
    inputs,H = convert_to_tensor([inputs,H], [Float64,Float64])
    out = plasticity_(val,H)
    set_shape(out, (size(inputs, 1), 3, 3))
end