export ExplicitSolverStep, GeneralizedAlphaSolverStep, ImplicitStaticSolver, LinearStaticSolver,
LinearGeneralizedAlphaSolverStep, LinearGeneralizedAlphaSolverInit!
@doc raw"""
    ExplicitSolverStep(globdat::GlobalData, domain::Domain, Δt::Float64)

Central Difference explicit solver for `M a + fint(u) = fext(u)`. `a`, `v`, `u` are acceleration, velocity and displacement.

```math
\begin{aligned}
u_{n+1} =& u_n + dtv_n + dt^2/2 a_n \\
v_{n+1} =& v_n + dt/2(a_n + a_{n+1}) \\
M a_{n+1} + f^{int}(u_{n+1}) =& f^{ext}_{n+1} \\
M a_{n+1} =& f^{ext}_{n+1} - f^{int}(u_{n+1}) \\
\end{aligned}
```

!!! info 
    You need to call SolverInitial! before the first time step, if $f^{ext}_0 \neq 0$. 
    Otherwise we assume the initial acceleration `globdat.acce[:] = 0`.
"""
function ExplicitSolverStep(globdat::GlobalData, domain::Domain, Δt::Float64)
    assembleMassMatrix!(globdat, domain)

    u = globdat.state[:]
    ∂u  = globdat.velo[:]
    ∂∂u = globdat.acce[:]

    globdat.time  += Δt
    updateTimeDependentEssentialBoundaryCondition!(domain, globdat)
    fext = getExternalForce(domain, globdat)

    u += Δt*∂u + 0.5*Δt*Δt*∂∂u
    ∂u += 0.5*Δt * ∂∂u
    
    domain.state[domain.eq_to_dof] = u[:]
    fint  = assembleInternalForce( globdat, domain, Δt)
    ∂∂up = globdat.M\(fext - fint)

    ∂u += 0.5 * Δt * ∂∂up

    globdat.Dstate = globdat.state[:]
    globdat.state = u[:]
    globdat.velo = ∂u[:]
    globdat.acce = ∂∂up[:]

    commitHistory(domain)
    updateStates!(domain, globdat)

    return globdat, domain
end


@doc raw"""
    ImplicitStaticSolver(globdat::GlobalData, domain::Domain; 
        N::Int64 = 10, ε::Float64 = 1.e-6, maxiterstep::Int64=100)

Solves 
$$K(u) = F$$
using the incremental load method. Specifically, at step $i$, we solve 
```math
f^{int}(u_i) = \frac{i}{N} f^{ext}
```

- `globdat`, GlobalData

- `domain`, Domain

- `N`, an integer, load stepping steps

- `ε`, Float64, absolute error for Newton convergence

-  `maxiterstep`, Int64, maximum iteration number for Newton convergence
"""
function ImplicitStaticSolver(globdat::GlobalData, domain::Domain; 
        N::Int64 = 10, ε::Float64 = 1.e-6, maxiterstep::Int64=100)
    assembleMassMatrix!(globdat, domain)
    fext = getExternalForce(domain, globdat)
    
    for iterstep = 1:N

        # Newton's method
        Newtoniterstep, Newtonconverge = 0, false
        
        while  !Newtonconverge

            Newtoniterstep += 1
            
            fint, stiff = assembleStiffAndForce( globdat, domain )
       
            res = fint - iterstep/N * fext

            Δstate = stiff\res

            globdat.state -= Δstate
            # @show Newtoniterstep, norm(res)
            if (norm(res) < ε  || Newtoniterstep > maxiterstep)
                if Newtoniterstep > maxiterstep
                    @error "$Newtoniterstep Newton iteration does not converge"
                end
                Newtonconverge = true
            end
            updateStates!(domain, globdat)
        end
        commitHistory(domain)
        globdat.Dstate = copy(globdat.state)
    end
    globdat, domain
end

@doc raw"""
    LinearStaticSolver(globdat::GlobalData, domain::Domain)

Solves the linear static problem 
$$\begin{aligned}\div \sigma &= f\\ \sigma = H \epsilon \end{aligned}$$

"""
function LinearStaticSolver(globaldata::GlobalData, domain::Domain)
    @assert globaldata.time ≈ 0.0
    fext = getExternalForce(domain, globaldata)
    fint, stiff = assembleStiffAndForce( globaldata, domain, 0.0)
    res = fext - fint  
    globaldata.state = stiff\res
    updateStates!(domain, globaldata)
    return globaldata, domain 
end



@doc raw"""
    GeneralizedAlphaSolverStep(globdat::GlobalData, domain::Domain, Δt::Float64, 
    ρ::Float64 = 0.0, ε::Float64 = 1e-8, ε0::Float64 = 1e-8, maxiterstep::Int64=100, 
    η::Float64 = 1.0, failsafe::Bool = false, verbose::Bool = false)


Implicit solver for 
$$Ma  + f_{int}(u) = fext$$
Here ``a``, ``v``, ``u`` are acceleration, velocity and displacement respectively.

- `ρ`: controls the damping effect of the α-scheme, ρ∈[0,1], ρ=1 corresponds to the maximum damping
- `ε`: Float64, absolute error for Newton convergence
- `ε0`: Float64, relative error for Newton convergence
- `max_iter`: Int64, maximum iteration number for Newton convergence
- `η`: Float64, Newton step size at the first iteration
- `failsafe`: Bool, if failsafe is true, when the Newton fails to converge, 
              revert back, and return false

The nonlinear $\alpha$
```math
u_{n+1} = u_n + dtv_n + dt^2/2 ((1 - 2\beta)a_n + 2\beta a_{n+1})
v_{n+1} = v_n + dt((1 - \gamma)a_n + \gamma a_{n+1})
2\beta = 0.5*(1 - αm + αf)^2
\gamma = 0.5 - \alpha_m + \alpha_f
```

```math
a_{n+1-\alpha_m} = (1-\alpha_m)a_{n+1} + \alpha_m a_{n} 
v_{n+1-\alpha_f} = (1-\alpha_f)v_{n+1} + \alpha_f v_{n}
u_{n+1-\alpha_f} = (1-\alpha_f)u_{n+1} + \alpha_f u_{n}
M a_{n+1-\alpha_m} + f^{int}(u_{n+1-\alpha_f}) = f^{ext}_{n+1-\alpha_f}
```

'a_{n+1}' is solved by 

```math
M ((1-\alpha_m)a_{n+1} + \alpha_m a_{n})  
+ f^{int}((1-\alpha_f)(u_n + dtv_n + dt^2/2 ((1 - 2\beta)a_n + 2\beta a_{n+1}))) + \alpha_f u_{n}) 
= f^{ext}_{n+1-\alpha_f}
```


As for ```\alpha_m``` and ``\alpha_f``
```math
\alpha_m = (2\rho_{\infty} - 1)/(\rho_{\infty} + 1)
\alpha_f = \rho_{\infty}/(\rho_{\infty} + 1)
```
    
use the current states `a`, `v`, `u`, `time` in globdat, and update these stetes to next time step
update domain history, when failsafe is true, and Newton's solver fails, nothing will be changed.

You need to call SolverInitial! before the first time step, if f^{ext}_0 != 0.
SolverInitial! updates a_0 in the globdat.acce
a_0 = M^{-1}(- f^{int}(u_0) + f^{ext}_0)

We assume globdat.acce[:] = a_0 and so far initialized to 0
We also assume the external force is conservative (it does not depend on the current deformation)
"""
function GeneralizedAlphaSolverStep(globdat::GlobalData, domain::Domain, Δt::Float64; 
        ρ::Float64 = 0.0, ε::Float64 = 1e-8, ε0::Float64 = 1e-8, maxiterstep::Int64=100, 
        η::Float64 = 1.0, failsafe::Bool = false, verbose::Bool = false)
    assembleMassMatrix!(globdat, domain)
    @assert 0<=ρ<=1
    αm = (2ρ-1)/(1+ρ)
    αf = ρ/(1+ρ)
    local res0
    
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf

    # compute solution at uⁿ⁺¹
    failsafe_time = copy(globdat.time)
    globdat.time  += (1 - αf)*Δt

    # domain.Dstate = uⁿ
    failsafe_Dstate = copy(domain.Dstate)
    failsafe_state = copy(domain.state)
    domain.Dstate = domain.state[:]


    updateTimeDependentEssentialBoundaryCondition!(domain, globdat)
    M = globdat.M
    
    ∂∂u = globdat.acce[:] #∂∂uⁿ
    u = globdat.state[:]  #uⁿ
    ∂u  = globdat.velo[:] #∂uⁿ

    fext = getExternalForce(domain, globdat)
    # fext = zeros(domain.neqs)
    ∂∂up = ∂∂u[:]

    Newtoniterstep, Newtonconverge = 0, false

    norm0 = Inf

    while !Newtonconverge
        
        Newtoniterstep += 1
        
        domain.state[domain.eq_to_dof] = (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u
        fint, stiff = assembleStiffAndForce( globdat, domain, Δt)
        res = M * (∂∂up *(1 - αm) + αm*∂∂u)  + fint - fext
        if Newtoniterstep==1
            res0 = res 
        end
        A = M*(1 - αm) + (1 - αf) * 0.5 * β2 * Δt^2 * stiff
        Δ∂∂u = A\res


        while η * norm(Δ∂∂u) > norm0
            η /= 2.0
            verbose && (@info "η", η)
        end
        ∂∂up -= η*Δ∂∂u

        if maxiterstep==1
            break
        end
        verbose && println("$Newtoniterstep/$maxiterstep, abs = $(norm(res)), rel = $(norm(res)/norm(res0))")
        if (norm(res)< ε || norm(res)< ε0*norm(res0) ||Newtoniterstep > maxiterstep)
            if Newtoniterstep > maxiterstep
                # Newton method does not converge
                if failsafe 
                    globdat.time = failsafe_time
                    domain.state = failsafe_state[:]
                    domain.Dstate = failsafe_Dstate[:]
                    return false 
                end
                # When failsafe is not on, test the gradient 
                function f(∂∂up)
                    domain.state[domain.eq_to_dof] = (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u
                    fint, stiff = assembleStiffAndForce( globdat, domain )
                    fint, (1 - αf) * 0.5 * β2 * Δt^2 * stiff
                end
                gradtest(f, ∂∂up)
                @warn("Newton iteration cannot converge $(norm(res))"); Newtonconverge = true
            else
                Newtonconverge = true
                verbose && printstyled("[Newmark] Newton converged $Newtoniterstep\n", color=:green)
            end
        end

        η = min(1.0, 2η)
        norm0 = norm(Δ∂∂u)
    end
    

    #update globdat to the next time step
    globdat.Dstate = globdat.state[:]
    globdat.state += Δt * ∂u + Δt^2/2 * ((1 - β2) * ∂∂u + β2 * ∂∂up)
    globdat.velo += Δt * ((1 - γ) * ∂∂u + γ * ∂∂up)
    globdat.acce = ∂∂up[:]
    globdat.time  += αf*Δt

    #commit history in domain
    commitHistory(domain)
    updateStates!(domain, globdat)
    if options.save_history>=2
        fint, stiff = assembleStiffAndForce( globdat, domain, Δt)
        push!(domain.history["fint"], fint)
        push!(domain.history["fext"], fext)
    end
    if options.save_history>=1
        push!(domain.history["time"], [globdat.time])
    end
    return globdat, domain
    
end 


function LinearGeneralizedAlphaSolverInit!(globdat::GlobalData, domain::Domain, Δt::Float64; 
    ρ::Float64 = 0.0)
    assembleMassMatrix!(globdat, domain)
    @assert 0<=ρ<=1
    αm = (2ρ-1)/(1+ρ)
    αf = ρ/(1+ρ)
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf
    M = globdat.M

    _, stiff = assembleStiffAndForce(domain, Δt)
    A = M*(1 - αm) + (1 - αf) * 0.5 * β2 * Δt^2 * stiff
    STORAGE["LinearGeneralizedAlphaSolverInit!"] = lu(A)
    nothing
end


"""
    LinearGeneralizedAlphaSolverStep(globdat::GlobalData, domain::Domain, Δt::Float64; 
    ρ::Float64 = 0.0)

Solves the dynamic structural equation with time-independent coefficient matrix. This is useful for linear elasticity simulation. 
"""
function LinearGeneralizedAlphaSolverStep(globdat::GlobalData, domain::Domain, Δt::Float64; 
    ρ::Float64 = 0.0)
    if haskey(STORAGE, "LinearGeneralizedAlphaSolverInit")
        error("You must call LinearGeneralizedAlphaSolverInit! before LinearGeneralizedAlphaSolverStep")
    end
    A = STORAGE["LinearGeneralizedAlphaSolverInit!"] 
    M = globdat.M
    @assert 0<=ρ<=1
    αm = (2ρ-1)/(1+ρ)
    αf = ρ/(1+ρ)
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf

    # compute solution at uⁿ⁺¹
    globdat.time  += (1 - αf)*Δt

    # domain.Dstate = uⁿ
    domain.Dstate = domain.state[:]

    updateTimeDependentEssentialBoundaryCondition!(domain, globdat)
    

    ∂∂u = globdat.acce[:] #∂∂uⁿ
    u = globdat.state[:]  #uⁿ
    ∂u  = globdat.velo[:] #∂uⁿ

    fext = getExternalForce(domain, globdat)
    ∂∂up = ∂∂u[:]
    
        
    domain.state[domain.eq_to_dof] = (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u

    fint = assembleInternalForce(domain, Δt)
    res = M * (∂∂up *(1 - αm) + αm*∂∂u)  + fint - fext
    Δ∂∂u = A\res
    ∂∂up -= Δ∂∂u


    #update globdat to the next time step
    globdat.Dstate = globdat.state[:]
    globdat.state += Δt * ∂u + Δt^2/2 * ((1 - β2) * ∂∂u + β2 * ∂∂up)
    globdat.velo += Δt * ((1 - γ) * ∂∂u + γ * ∂∂up)
    globdat.acce = ∂∂up[:]
    globdat.time  += αf*Δt

    #commit history in domain
    commitHistory(domain)
    updateStates!(domain, globdat)
    if options.save_history>=2
        fint, stiff = assembleStiffAndForce( globdat, domain, Δt)
        push!(domain.history["fint"], fint)
        push!(domain.history["fext"], fext)
    end
    if options.save_history>=1
        push!(domain.history["time"], [globdat.time])
    end
    return globdat, domain

end 