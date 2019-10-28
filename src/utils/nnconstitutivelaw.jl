export nn_constitutive_law

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
function nn_constitutive_law(input::Array{Float64,2}, θ::Array{Float64,1}, g::Union{Array{Float64,2}, Nothing}=nothing, grad_input::Int64=0, grad_θ::Int64=0)
    n = size(input,1)
    @assert size(input,2)==9

    lib = joinpath(@__DIR__, "../../deps/CustomOp/ADLaw/build/libnnlaw")
    σ = zeros(n*3)
    input = input'[:]

    dinput = zeros()
    if(grad_input>0)
        dinput = zeros(n*9*3)
    end

    dθ = zeros()
    if(grad_θ>0)
        @assert size(g,1)==n && size(g,2)==3
        g = g'[:]
        dθ = zeros(length(θ))
    else
        g = zeros()
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