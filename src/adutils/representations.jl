export consistent_tangent_matrix, isotropic_function, strain_voigt_to_tensor
@doc raw"""
    consistent_tangent_matrix(inputs::Union{Array{Float64, 2}, PyObject},Dc::Union{Array{Float64,2}, PyObject})

Returns the consistent tangent matrices. The size of the return is $N\times 3 \times 3$. 


$$D_c^{ep} = D_c - \frac{D_c\frac{\partial g}{\partial \sigma}\left(\frac{\partial f}{\partial \sigma}Dc \right)^T D_c }{E_p + \frac{\partial g}{\partial \sigma} Dc \left(\frac{\partial f}{\partial \sigma}Dc \right)^T}$$


Here `inputs` is a $N\times 7$ matrix, where each row is 

$$\left[\frac{\partial g}{\partial \sigma}, \frac{\partial f}{\partial \sigma}, E_p\right]$$

`Dc` is a $3\times 3$ **row-major** matrix; each row is a linear elasticity matrix.  
"""
function consistent_tangent_matrix(inputs::Union{Array{Float64, 2}, PyObject},H::Union{Array{Float64,2}, PyObject})
    @assert size(inputs,2)==7
    @assert size(H,1)==size(H,2)==3
    plasticity_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/Plasticity/build/libPlasticity","plasticity")
    inputs,H = convert_to_tensor([inputs,H], [Float64,Float64])
    out = plasticity_(val,H)
    set_shape(out, (size(inputs, 1), 3, 3))
end


@doc raw"""
    isotropic_function(coef::Union{Array{Float64,2}, PyObject},strain::Union{Array{Float64,2}, PyObject})

isotropic function of a symmetric tensor.

$$T = s_0 I + s_1 A + s_2 A^2$$

Here 

$$\texttt{coef(i,:)} = [s_0\ s_1\ s_2]$$

$$A = \left[\begin{matrix}
S_{i1} & S_{i3}/2\\ 
S_{i3}/2 & S_{i2}
\end{matrix}\right]$$

where $S_{ij}$ is the $i$-th row and $j$-th column of `strain`
"""
function isotropic_function(coef::Union{Array{Float64,2}, PyObject},strain::Union{Array{Float64,2}, PyObject})
    @assert size(strain,2)==size(coef,2)==3
    @assert size(coef,1)==size(strain,1)
    isotropic_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/Isotropic/build/libIsotropic","isotropic")
    coef,strain = convert_to_tensor([coef,strain], [Float64,Float64])
    out = isotropic_(coef,strain)
    set_shape(out, (size(coef,1), 3))
end


@doc raw"""
    strain_voigt_to_tensor(inp::Union{Array{Float64,2}, PyObject})

Converts Voigt strain tensors to matrix form  

$$\begin{bmatrix}
\epsilon_{11}\\
\epsilon_{22}\\
2\epsilon_{12}\\
\end{bmatrix} \rightarrow \begin{bmatrix}
\epsilon_{11} &  \epsilon_{12}\\
\epsilon_{12} & \epsilon_{22}
\end{bmatrix}$$

The input is $N\times 3$ and the output is $N\times 2 \times 2$
"""
function strain_voigt_to_tensor(inp::Union{Array{Float64,2}, PyObject})
    @assert size(inp,2)==3
    tensor_rep_ = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/TensorRep/build/libTensorRep","tensor_rep")
    inp = convert_to_tensor([inp], [Float64]); inp = inp[1]
    set_shape(tensor_rep_(inp), (size(inp,1), 2,2))
end