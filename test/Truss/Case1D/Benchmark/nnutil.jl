
using ForwardDiff
using DelimitedFiles
using MAT


nn_width = 20
nn_depth = 3
σY = 0.01e6
dσ = 1.0
activation = "tanh"
exp_id = 1

if length(ARGS)==6
    nn_width = parse(Int64, ARGS[1])
    nn_depth = parse(Int64, ARGS[2])
    σY = parse(Float64, ARGS[3])
    dσ = parse(Float64, ARGS[4])
    activation = ARGS[5]
    exp_id = parse(Int64, ARGS[6])
end
FILEID = "nn_width$(nn_width)nn_depth$(nn_depth)sigmaY$(σY)dsigma$(dσ)activation$(activation)exp_id$exp_id"
FILEID2 = "nn_width$(nn_width)nn_depth$(nn_depth)sigmaY$(σY)dsigma$(dσ)activation$(activation)"

@info FILEID

if activation=="selu"
    global activation = x -> tf.nn.selu(x)
elseif activation=="elu"
    global activation = x -> tf.nn.elu(x)
end

config = [[nn_width for i=1:nn_depth]...,1]

stress_scale = 1e2
strain_scale = 1e-3

E0 = 200.0e3/(stress_scale/strain_scale)


function nn(ε, ε0, σ0)
    @show ε, ε0, σ0
    x = [ε/strain_scale ε0/strain_scale σ0/stress_scale]
    H = ae(x, config, "default", activation = activation)^2
    s = σ0^2
    i = sigmoid((s - σY)/dσ)  

    y = ( H.* i + E0 * (1-i) ) .* (ε-ε0)*stress_scale/strain_scale + σ0
    return y

    
end

function sigmoid_(z)
    return 1.0 / (1.0 + exp(-z))
end

function post_nn(ε::Float64, ε0::Float64, σ0::Float64, Δt::Float64)    
    f = x -> begin
        H = nnpiecewise(reshape([x/strain_scale;ε0/strain_scale;σ0/stress_scale],1,3))[1,1]^2

        @show H
        s = σ0^2
        i = sigmoid_((s - σY)/dσ)

        (H * i + E0 * (1.0-i) )*stress_scale/strain_scale * (x-ε0) + σ0

        
    end
    df = ForwardDiff.derivative(f, ε)
    return f(ε), df
end

