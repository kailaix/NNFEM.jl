export gradtest, gradtest_assembleStiffAndForce

@doc raw"""
    gradtest(f::Function, x0::Array{Float64}; scale::Float64 = 1.0)

Testing the gradients of a vector function `f`. 
Here `x0` is $n$-dimensional vector, `f` takes an $n$-dimensional vector as inputs, and
outputs a $m$ dimensional vector. The function tests the second order convergence of `f`
```math
\|f(x_0+\gamma c) - f(x_0) - \gamma \nabla f(x_0) c\|_2 = \mathcal{O}(\gamma^2)
```
"""
function gradtest(f::Function, x0::Array{Float64}; scale::Float64 = 1.0)
    v0 = rand(Float64,size(x0))
    γs = scale ./10 .^(1:5)
    err2 = []
    err1 = []
    f0, J = f(x0)
    for i = 1:5
        f1, _ = f(x0+γs[i]*v0)
        push!(err1, norm(f1-f0))
        # @show f1, f0, 2γs[i]*J*v0
        push!(err2, norm(f1-f0-γs[i]*J*v0))
        # push!(err2, norm((f1-f2)/(2γs[i])-J*v0))
        # #@show "test ", f1, f2, f1-f2
    end
    close("all")
    loglog(γs, err2, label="Automatic Differentiation")
    loglog(γs, err1, label="Finite Difference")
    loglog(γs, γs.^2 * 0.5*abs(err2[1])/γs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(γs, γs * 0.5*abs(err1[1])/γs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
    plt.gca().invert_xaxis()
    legend()
    println("Finite difference: $err1")
    println("Automatic differentiation: $err2")
    return err1, err2
end

function gradtest_assembleStiffAndForce(domain::Domain; scale::Float64 = 1.0)
    function f(s)
        domain.state[domain.dof_to_eq] = s
        rhs, K = assembleStiffAndForce(domain, 0.0)
        return rhs, K 
    end
    gradtest(f, rand(domain.neqs), scale=scale)
end