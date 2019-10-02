export sym_H, orthotropic_H, spd_H

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

function spd_H(o::Array, H0::Array)
    # @show size(o'*H0*o)
    o = o[:]
    H0 - H0*(o*o')*H0/(1+o'*H0*o)
end

function sym_H(o::Array)
    [o[1] o[2] o[3];
    o[2] o[4] o[5];
    o[3] o[5] o[6]]
end

function orthotropic_H(o::Array)
    [o[1] o[2] 0.0;
    o[2] o[3] 0.0;
    0.0 0.0 o[4]]
end

function spd_H(o::PyObject, H0::Array{Float64,2})
    if size(o,2)!=3
        error("NNFEM: second dimension of `o` must be 2")
    end
    spd_op(constant(H0), o)
end