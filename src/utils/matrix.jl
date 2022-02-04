export sym_H, orthotropic_H, spd_H, spd_Cholesky, spd_Chol_Orth, spd_zero_to_H


# @doc raw"""
#     sym_H(y::PyObject)
#     sym_H(o::Array)

# Creates a symmetric matrix from 6 parameters
# ```math
# H = \begin{bmatrix}
# y_1 & y_2 & y_3\\ 
# y_2 & y_4 & y_5 \\ 
# y_3 & y_5 & y_6
# \end{bmatrix}
# ```
# """
# function sym_H(y::PyObject)
#     y = sym_op(y)
#     z = tf.reshape(y, (-1,3,3)) 
# end

function sym_H(o::Array)
    [o[1] o[2] o[3];
    o[2] o[4] o[5];
    o[3] o[5] o[6]]
end


@doc raw"""
    orthotropic_H(y::PyObject)
    orthotropic_H(o::Array)

Creates a symmetric matrix from 4 parameters
```math
H = \begin{bmatrix}
y_1 & y_2 & 0\\ 
y_2 & y_3 & 0 \\ 
0 & 0 & y_4
\end{bmatrix}
```
"""
function orthotropic_H(y::PyObject)

    y = orthotropic_op(y)

    z = tf.reshape(y, (-1,3,3)) 

    return z
end

function orthotropic_H(o::Array)
    [o[1] o[2] 0.0;
    o[2] o[3] 0.0;
    0.0 0.0 o[4]]
end

@doc raw"""
    spd_H(o::PyObject, H0::Array{Float64,2})
    spd_H(o::Array, H0::Array)

Creates a SPD matrix from 3 scalars

```math
H = H_0 - \frac{H_0nn'H_0}{1+n'H_0n}
```
"""
function spd_H(o::PyObject, H0::Array{Float64,2})
    if size(o,2)!=3
        error("NNFEM: second dimension of `o` must be 3")
    end
    spd_op = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib", "spd_op")
    ret = spd_op(constant(H0), o)
  
    # ret.set_shape((-1,3,3))
    return ret
end

function spd_H(o::Array, H0::Array)
    # @show size(o'*H0*o)
    o = o[:]
    H0 - H0*(o*o')*H0/(1+o'*H0*o)
end



@doc raw"""
    spd_Cholesky(o::Array)
    spd_Cholesky(o::PyObject)

Creates a SPD matrix from 6 scalars. 

```math
A = LL'
```
where
```math
L = \begin{bmatrix}
o_1 & &  \\
o_2 & o_4 & \\
o_3 & o_5 & o_6 
\end{bmatrix}
```
"""
function spd_Cholesky(o::Array)
    # @show size(o'*H0*o)
    [o[1]*o[1] o[1]*o[2] o[1]*o[3];
     o[1]*o[2] o[2]*o[2]+o[4]*o[4] o[2]*o[3]+o[4]*o[5];
     o[1]*o[3] o[2]*o[3]+o[4]*o[5] o[3]*o[3]+o[5]*o[5]+o[6]*o[6]]
end


function spd_Cholesky(o::PyObject)
    if size(o,2)!=6
        error("NNFEM: second dimension of `o` must be 6")
    end
  
    chol_op = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib", "chol_op")
    ret = chol_op(o)
   
    tf.reshape(ret, (-1,3,3))
end


@doc raw"""
    spd_Chol_Orth(o::Array)
    spd_Chol_Orth(o::PyObject)

Creates a SPD matrix from 4 scalars. 

```math
A = LL'
```
where
```math
L = \begin{bmatrix}
o_1 & & \\
o_2 & o_3 & \\
 &  & o_4 
\end{bmatrix}
```
"""
function spd_Chol_Orth(o::Array)
    # @show size(o'*H0*o)
    [o[1]*o[1] o[1]*o[2] 0;
     o[1]*o[2] o[2]*o[2]+o[3]*o[3] 0.0;
     0.0       0.0                 o[4]*o[4]]
end

function spd_Chol_Orth(o::PyObject)
    if size(o,2)!=4
        error("NNFEM: second dimension of `o` must be 4")
    end
    
    chol_orth_op = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib", "chol_orth_op")
    ret = chol_orth_op(o)
 
    tf.reshape(ret, (-1,3,3))
end

@doc raw"""
    spd_zero_to_H(o::Array)
    spd_zero_to_H(o::Array, H0inv::Array{Float64,2})

Creates a SPD matrix from 4 scalars. 

```math
A = (H_0^{-1} +LL')^{-1}
```
where
```math
L =  \begin{bmatrix}
o_1 & & \\
o_2 & o_3 & \\
 &  & o_4
\end{bmatrix}
```

"""
function spd_zero_to_H(o::PyObject, H0inv::Array{Float64,2})
    if size(o,2)!=4
        error("NNFEM: second dimension of `o` must be 4")
    end

    chol_orth_op = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/build/libDataLib", "chol_orth_op")
    ret = chol_orth_op(o)

    out = tf.reshape(ret, (-1,3,3))
    inv(H0inv+out)
end

function spd_zero_to_H(o::Array, H0inv::Array{Float64,2})
    
    # ret = chol_orth_op(o)
    # @show ret
    # out = tf.reshape(ret, (-1,3,3))
    # inv(H0inv+out)

    spd_o = [o[1]*o[1] o[1]*o[2] 0;
         o[1]*o[2] o[2]*o[2]+o[3]*o[3] 0.0;
         0.0       0.0                 o[4]*o[4]]
    
    inv(H0inv+spd_o)
end