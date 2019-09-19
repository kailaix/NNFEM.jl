export sym_H, orthotropic_H

function sym_H(y::PyObject)
    y = sym_op(y)
    z = tf.reshape(y, (-1,3,3)) 
end

function orthotropic_H(y::PyObject)
    @show y
    y = orthotropic_op(y)
    @show y
    z = tf.reshape(y, (-1,3,3)) 
    @show z 
    return z
end

function sym_H(o::Array{Float64})
    [o[1] o[2] o[3];
    o[2] o[4] o[5];
    o[3] o[5] o[6]]
end

function orthotropic_H(o::Array{Float64})
    [o[1] o[2] 0.0;
    o[2] o[3] 0.0;
    0.0 0.0 o[4]]
end