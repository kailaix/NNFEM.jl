using ADCME
using PyCall

if Sys.islinux()
py"""
import tensorflow as tf
libSymOp = tf.load_op_library('Ops/Sym/build/libSymOp.so')
@tf.custom_gradient
def sym_op(x):
    y = libSymOp.sym_op(x)
    def grad(dy):
        return libSymOp.sym_op_grad(dy, y, x)
    return y, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libSymOp = tf.load_op_library('Ops/Sym/build/libSymOp.dylib')
@tf.custom_gradient
def sym_op(x):
    y = libSymOp.sym_op(x)
    def grad(dy):
        return libSymOp.sym_op_grad(dy, y, x)
    return y, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libSymOp = tf.load_op_library('Ops/Sym/build/libSymOp.dll')
@tf.custom_gradient
def sym_op(x):
    y = libSymOp.sym_op(x)
    def grad(dy):
        return libSymOp.sym_op_grad(dy, y, x)
    return y, grad
"""
end
    
sym_op = py"sym_op"