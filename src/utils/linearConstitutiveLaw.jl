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
#=
function linear_constitutive_law(input::Array{Float64,2}, θ::Array{Float64,1}, 
    g::Union{Array{Float64,2}, Nothing}=nothing, grad_input::Bool=false, grad_θ::Bool=false)
    
    @assert size(θ,1)==6
    
    H0 = [θ[1] θ[2] θ[3]; 
          θ[2] θ[4] θ[5];  
          θ[3] θ[5] θ[6]]

    n = size(input,1)
    @assert size(input,2)==9

    ε = input[:, 1:3]

    
    σ = zeros(Float64, n, 3)
    for i = 1:n
        σ[i,:] = H0 * ε[i, :] 
    end

    
    dinput = zeros()
    if(grad_input)
        dinput = zeros(Float64, n, 9, 3)
        for i = 1:n
            dinput[i,1:3,:] = H0'
        end

        grad_input = 1
    else
        grad_input = 0
    end

    dθ = zeros()
    if(grad_θ)
        @assert size(g,1)==n && size(g,2)==3
        
        dθ = zeros(length(θ))

        for i = 1:n
        
            dθ[1] += ε[i,1] * g[i,1]
            dθ[2] += ε[i,1] * g[i,2] + ε[i,2] * g[i,1]
            dθ[3] += ε[i,1] * g[i,3] + ε[i,3] * g[i,1]
            dθ[4] += ε[i,2] * g[i,2]
            dθ[5] += ε[i,2] * g[i,3] + ε[i,3] * g[i,2]
            dθ[6] += ε[i,3] * g[i,3]

        end

        grad_θ = 1
    else
        grad_θ = 0
    end


    
    return σ, dinput, dθ
end
=#

function linear_constitutive_law(input::Array{Float64,2}, θ::Array{Float64,1}, 
    g::Union{Array{Float64,2}, Nothing}=nothing, grad_input::Bool=false, grad_θ::Bool=false)
    
    @assert size(θ,1)==6
    
    H0 = [θ[1] θ[2] θ[3]; 
          θ[2] θ[4] θ[5];  
          θ[3] θ[5] θ[6]]

    n = size(input,1)
    @assert size(input,2)==9

    ε ,ε0, σ0 = input[:, 1:3],input[:, 4:6],input[:, 7:9]

    
    σ = zeros(Float64, n, 3)
    for i = 1:n
        σ[i,:] = H0 * (ε[i, :] -  ε0[i, :]) +  σ0[i, :]
        #σ[i,:] = H0 * ε[i, :]  +  σ0[i, :]
    end

    I3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    dinput = zeros()
    if(grad_input)
        dinput = zeros(Float64, n, 9, 3)
        for i = 1:n
            dinput[i,1:3,:] =  H0'
            dinput[i,4:6,:] = -H0'
            dinput[i,7:9,:] =  I3'
        end

        grad_input = 1
    else
        grad_input = 0
    end

    dθ = zeros()
    if(grad_θ)
        @assert size(g,1)==n && size(g,2)==3
        
        dθ = zeros(length(θ))

        for i = 1:n

            # dθ[1] += ε[i,1] * g[i,1]
            # dθ[2] += ε[i,1] * g[i,2] + ε[i,2] * g[i,1]
            # dθ[3] += ε[i,1] * g[i,3] + ε[i,3] * g[i,1]
            # dθ[4] += ε[i,2] * g[i,2]
            # dθ[5] += ε[i,2] * g[i,3] + ε[i,3] * g[i,2]
            # dθ[6] += ε[i,3] * g[i,3]
        
            dθ[1] += (ε[i,1]-ε0[i,1]) * g[i,1]
            dθ[2] += (ε[i,1]-ε0[i,1]) * g[i,2] + (ε[i,2]-ε0[i,2]) * g[i,1]
            dθ[3] += (ε[i,1]-ε0[i,1]) * g[i,3] + (ε[i,3]-ε0[i,3]) * g[i,1]
            dθ[4] += (ε[i,2]-ε0[i,2]) * g[i,2]
            dθ[5] += (ε[i,2]-ε0[i,2]) * g[i,3] + (ε[i,3]-ε0[i,3]) * g[i,2]
            dθ[6] += (ε[i,3]-ε0[i,3]) * g[i,3]

        end

        grad_θ = 1
    else
        grad_θ = 0
    end


    
    return σ, dinput, dθ
end

