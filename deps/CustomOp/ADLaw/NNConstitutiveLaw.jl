"""
    nn_constitutive_law(input::Array{Float64,2}, θ::Array{Float64,1}, grad_input::Int64, grad_θ::Int64)


Computes the stress with the given neural network weights θ.
# Inputs
`input` is a ``n\\times 9`` array, θ is a `p` dimensional array
`grad_input = 0`: do not compute gradients with respect to input; otherwise compute the gradients, in this case, `g` must have the same size as σ
`grad_θ = 0`    : do not compute gradients with respect to θ; otherwise compute the gradients

output:
- σ
- ``\\frac{\\partial \\sigma}{\\partial ipt}``
- ``\\frac{\\partial \\sigma\\cdot g}{\\partial \\theta}``
"""
function nn_constitutive_law(input::Array{Float64,2}, θ::Array{Float64,1}, 
    g::Union{Array{Float64,2}, Nothing}=nothing, grad_input::Bool=false, grad_θ::Bool=false)
    n = size(input,1)
    @assert size(input,2)==9

    lib = joinpath(@__DIR__, "build/libnnlaw")
    σ = zeros(n*3)
    input = input'[:]

    dinput = zeros()
    if(grad_input)
        dinput = zeros(n*9*3)
        grad_input = 1
    else
        grad_input = 0
    end

    dθ = zeros()
    if(grad_θ)
        @assert size(g,1)==n && size(g,2)==3
        g = g'[:]
        dθ = zeros(length(θ))
        grad_θ = 1
    else
        g = zeros()
        grad_θ = 0
    end

    @eval begin
        ccall((:constitutive_law, $lib), Cvoid,
            (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},
            Cint, Cint, Cint), $σ, $input, $θ, $g, $dinput, $dθ, $n, $grad_input, $grad_θ)
    end

    σ = reshape(σ, 3, n)'|>Array
    if grad_input>0
        # @show "here"
        dinput = permutedims(reshape(dinput,9,3,n),[3,1,2])
        # dinput[:,:,3] -= dinput[:,:,2]
        # dinput[:,:,2] -= dinput[:,:,1]
    end
    
    return σ, dinput, dθ
end

function nn_constitutive_law(input::Array{Float64,2}, θ::Array{Float64,1}, 
        config::Array{Int64},
        g::Union{Array{Float64,2}, Nothing}=nothing, grad_input::Bool=false, grad_θ::Bool=false)
    
    @assert config[1]==9 && config[end]==4
    n_layer = length(config)
    nθ = 0
    for i = 1:n_layer-1
        nθ += config[i]*config[i+1]
        nθ += config[i+1]
    end
    @assert length(θ)==nθ

    n = size(input,1)
    @assert size(input,2)==9

    lib = joinpath(@__DIR__, "build/libnnlaw")
    σ = zeros(n*3)
    input = input'[:]

    dinput = zeros()
    if(grad_input)
        dinput = zeros(n*9*3)
        grad_input = 1
    else
        grad_input = 0
    end

    dθ = zeros()
    if(grad_θ)
        @assert size(g,1)==n && size(g,2)==3
        g = g'[:]
        dθ = zeros(length(θ))
        grad_θ = 1
    else
        g = zeros()
        grad_θ = 0
    end

    @eval begin
        ccall((:constitutive_law_generic, $lib), Cvoid,
            (Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},
            Ptr{Cint}, Cint,
            Cint, Cint, Cint), $σ, $input, $θ, $g, $dinput, $dθ, 
            $(Int32.(config)), $n_layer,
            $n, $grad_input, $grad_θ)
    end

    σ = reshape(σ, 3, n)'|>Array
    if grad_input>0
        # @show "here"
        dinput = permutedims(reshape(dinput,9,3,n),[3,1,2])
        # dinput[:,:,3] -= dinput[:,:,2]
        # dinput[:,:,2] -= dinput[:,:,1]
    end
    
    return σ, dinput, dθ
end


function jl_nn_constitutive_law(input::Array{Float64,2}, θ::Array{Float64,1}, g::Union{Array{Float64,2}, Nothing}=nothing, grad_input::Int64=0, grad_θ::Int64=0)
    x = input
    k = 0
    
    w = reshape(θ[k+1:k+180], 20, 9)'; k+=180
    x = tanh.(x*w) .+ reshape(θ[k+1:k+20], 1, :); k+= 20

    w = reshape(θ[k+1:k+400], 20, 20)'; k+=400
    x = tanh.(x*w) .+ reshape(θ[k+1:k+20], 1, :); k+= 20

    w = reshape(θ[k+1:k+60], 3, 20)'; k+=60
    x = tanh.(x*w) .+ reshape(θ[k+1:k+3], 1, :); k+= 3

    return x
end