export sym_H, orthotropic_H, spd_H, spd_Cholesky, spd_Chol_Orth

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
        error("NNFEM: second dimension of `o` must be 3")
    end
    ret = spd_op(constant(H0), o)
    @show ret
    # ret.set_shape((-1,3,3))
    return ret
end




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
    @show o
    ret = chol_op(o)
    @show ret
    tf.reshape(ret, (-1,3,3))
end



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
    @show o
    ret = chol_orth_op(o)
    @show ret
    tf.reshape(ret, (-1,3,3))
end