export ExplicitSolverStep, GeneralizedAlphaSolverStep
@doc raw"""
    ExplicitSolverStep(globdat::GlobalData, domain::Domain, Δt::Float64)

Central Difference explicit solver for `M a + fint(u) = fext(u)`. `a`, `v`, `u` are acceleration, velocity and displacement.

```math
\begin{align}
u_{n+1} =& u_n + dtv_n + dt^2/2 a_n \\
v_{n+1} =& v_n + dt/2(a_n + a_{n+1}) \\
M a_{n+1} + f^{int}(u_{n+1}) =& f^{ext}_{n+1} \\
M a_{n+1} =& f^{ext}_{n+1} - f^{int}(u_{n+1}) \\
\end{align}
```

!!! info 
    You need to call SolverInitial! before the first time step, if $f^{ext}_0 \neq 0$. 
    Otherwise we assume the initial acceleration `globdat.acce[:] = 0`.
"""
function ExplicitSolverStep(globdat::GlobalData, domain::Domain, Δt::Float64)
    u = globdat.state[:]
    ∂u  = globdat.velo[:]
    ∂∂u = globdat.acce[:]

    fext = getExternalForce!(domain, globdat)

    u += Δt*∂u + 0.5*Δt*Δt*∂∂u
    ∂u += 0.5*Δt * ∂∂u
    
    domain.state[domain.eq_to_dof] = u[:]
    fint  = assembleInternalForce( globdat, domain, Δt)
    ∂∂up = globdat.M\(fext + fbody - fint)

    ∂u += 0.5 * Δt * ∂∂up

    globdat.Dstate = globdat.state[:]
    globdat.state = u[:]
    globdat.velo = ∂u[:]
    globdat.acce = ∂∂up[:]
    globdat.time  += Δt
    commitHistory(domain)
    updateStates!(domain, globdat)

    return globdat, domain
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


    updateDomainStateBoundary!(domain, globdat)
    M = globdat.M
    
    ∂∂u = globdat.acce[:] #∂∂uⁿ
    u = globdat.state[:]  #uⁿ
    ∂u  = globdat.velo[:] #∂uⁿ

    fext = getExternalForce!(domain, globdat)
    # fext = zeros(domain.neqs)
    ∂∂up = ∂∂u[:]

    Newtoniterstep, Newtonconverge = 0, false

    norm0 = Inf

    while !Newtonconverge
        
        Newtoniterstep += 1
        
        domain.state[domain.eq_to_dof] = (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u
        fint, stiff = assembleStiffAndForce( globdat, domain, Δt)
        res = M * (∂∂up *(1 - αm) + αm*∂∂u)  + fint - fext - fbody
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


        verbose && println("$Newtoniterstep/$maxiterstep, $(norm(res))")
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
    fint, stiff = assembleStiffAndForce( globdat, domain, Δt)
    push!(domain.history["fint"], fint)
    push!(domain.history["fext"], fext)
    push!(domain.history["time"], [globdat.time])

    return globdat, domain
    
end 