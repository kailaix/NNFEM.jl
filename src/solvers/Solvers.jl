export ExplicitSolver, NewmarkSolver, AdaptiveSolver, StaticSolver

@doc raw"""
ExplicitSolver(Δt, globdat, domain)

Central Difference Explicit solver for `Ma + fint = fext`, `a`, `v`, `u` are acceleration, velocity and displacement

```math
\begin{align}
u_{n+1} =& u_n + dtv_n + dt^2/2 a_n \\
v_{n+1} =& v_n + dt/2(a_n + a_{n+1}) \\
M a_{n+1} + C v_{n+1} + R(u_{n+1}) =& P_{n+1} \\
(M + dt/2 C) a_{n+1} =& P_{n+1} - R(u_{n+1}) - C(v_n + dt/2 a_{n}) \\
\end{align}
```

    Alternative, todo:
    M a_n + C v_n + R(u_n) = P_n
    M(u_{n+1} - 2u_n + u_{n-1}) + dt/2*C(u_{n+1} - u_{n-1}) + dt^2 R(u_n) = dt^2 P_n
    (M + dt/2 C) u_{n+1} = dt^2(P_n - R(u_n) + dt/2 C u_{n-1} + M(2u_n - u_{n-1})

    For the first step
    u_{-1} = u_0 - dt*v_0 + dt^2/2 a_0
    a_0 = M^{-1}(-Cv_0 - R(u_0) + P_0)
"""->
function ExplicitSolver(Δt, globdat, domain)
    u = globdat.state[:]
    ∂u  = globdat.velo[:]
    ∂∂u = globdat.acce[:]

    fext = domain.fext
    
    ∂u += 0.5*Δt * ∂∂u
    u += Δt * ∂u
    
    domain.state[domain.eq_to_dof] = u[:]
    fint  = assembleInternalForce( globdat, domain, Δt)

    #@show fint, fext

    if length(globdat.M)==0
        error("globalDat is not initialized, call `assembleMassMatrix!(globaldat, domain)`")
    end

    ∂∂up = (fext - fint)./globdat.Mlumped
    ∂u += 0.5 * Δt * ∂∂up

    globdat.Dstate = globdat.state[:]
    globdat.state = u[:]
    globdat.velo = ∂u[:]
    globdat.acce = ∂∂up[:]

    globdat.time  += Δt

    commitHistory(domain)
    updateStates!(domain, globdat)
    fint = assembleInternalForce( globdat, domain, Δt)
    push!(domain.history["fint"], fint)
    push!(domain.history["fext"], fext)
end





@doc raw"""
    NewmarkSolver(Δt, globdat, domain, αm = -1.0, αf = 0.0, ε = 1e-8, ε0 = 1e-8, maxiterstep=100, η = 1.0, failsafe = false)

Implicit solver for ``Ma + C v + R(u) = P``
``a``, ``v``, ``u`` are acceleration, velocity and displacement respectively.
```math
u_{n+1} = u_n + dtv_n + dt^2/2 ((1 - 2\beta)a_n + 2\beta a_{n+1})
v_{n+1} = v_n + dt((1 - gamma)a_n + gamma a_{n+1})
```

```math
M a_{n+0.5} + f_{\mathrm{int}}(u_{n+0.f}) = fext_{n+0.5}
```

    αm = (2\rho_oo - 1)/(\rho_oo + 1)
    αf = \rho_oo/(\rho_oo + 1)
    
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf

    absolution error ε = 1e-8, 
    relative error ε0 = 1e-8  
    
    return true or false indicating converging or not
"""->
function NewmarkSolver(Δt, globdat, domain, αm = -1.0, αf = 0.0, ε = 1e-8, ε0 = 1e-8, maxiterstep=100, η = 1.0, failsafe = false)
    local res0
    # @info maxiterstep
    # error()
    #@info NewmarkSolver
    
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

    fext = similar(u)
    getExternalForce!(domain, globdat, fext)


    ∂∂up = ∂∂u[:]

    Newtoniterstep, Newtonconverge = 0, false

    norm0 = Inf

    # @show "start Newton u ", u
    # @show "start Newton du ", ∂u
    # @show "start Newton ddu ", ∂∂u

    while !Newtonconverge
        
        Newtoniterstep += 1
        
        domain.state[domain.eq_to_dof] = (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u

        fint, stiff = assembleStiffAndForce( globdat, domain, Δt)

        # error()
        res = M * (∂∂up *(1 - αm) + αm*∂∂u)  + fint - fext
        # @show fint, fext
        if Newtoniterstep==1
            res0 = res 
        end

        A = M*(1 - αm) + (1 - αf) * 0.5 * β2 * Δt^2 * stiff
        
        Δ∂∂u = A\res

        #@info " norm(Δ∂∂u) ", norm(Δ∂∂u) 
        while η * norm(Δ∂∂u) > norm0
            η /= 2.0
            @info "η", η
        end
            


        ∂∂up -= η*Δ∂∂u

        # @show norm(A*Δ∂∂u-res)
        # r, B = f(∂∂u)
        # #@show norm(res), norm(r), norm(B)
        # error()
        # #@show Δ∂∂u
        # if globdat.time>0.0001
        #     function f(∂∂up)
        #         domain.state[domain.eq_to_dof] = (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u
        #         fint, stiff = assembleStiffAndForce( globdat, domain )
        #         fint, (1 - αf) * 0.5 * β2 * Δt^2 * stiff
        #     end
        #     gradtest(f, ∂∂up)
        #     error()
        # end
        println("$Newtoniterstep/$maxiterstep, $(norm(res))")
        if (norm(res)< ε || norm(res)< ε0*norm(res0) ||Newtoniterstep > maxiterstep)

            if Newtoniterstep > maxiterstep
                if failsafe 
                    globdat.time = failsafe_time
                    domain.state = failsafe_state[:]
                    domain.Dstate = failsafe_Dstate[:]
                    return false 
                end
                function f(∂∂up)
                    domain.state[domain.eq_to_dof] = (1 - αf)*(u + Δt*∂u + 0.5 * Δt * Δt * ((1 - β2)*∂∂u + β2*∂∂up)) + αf*u
                    fint, stiff = assembleStiffAndForce( globdat, domain )
                    fint, (1 - αf) * 0.5 * β2 * Δt^2 * stiff
                end
                gradtest(f, ∂∂up)
                # error()
                @warn("Newton iteration cannot converge $(norm(res))"); Newtonconverge = true

                
            else
                Newtonconverge = true
                printstyled("[Newmark] Newton converged $Newtoniterstep\n", color=:green)
            end
        end

        η = min(1.0, 2η)
        norm0 = norm(Δ∂∂u)
        # println("================ time = $(globdat.time) $Newtoniterstep =================")
    end
    


    globdat.Dstate = globdat.state[:]
    globdat.state += Δt * ∂u + Δt^2/2 * ((1 - β2) * ∂∂u + β2 * ∂∂up)

    
    globdat.velo += Δt * ((1 - γ) * ∂∂u + γ * ∂∂up)
    globdat.acce = ∂∂up[:]

   
    globdat.time  += αf*Δt
    #@info "After Newmark, at T = ", globdat.time, " the disp is ", domain.state
    commitHistory(domain)
    updateStates!(domain, globdat)

    
    

    fint, stiff = assembleStiffAndForce( globdat, domain, Δt)
    push!(domain.history["fint"], fint)
    push!(domain.history["fext"], fext)

    return true
    
end 


@doc raw"""
    StaticSolver(globdat, domain, loaditerstep = 10, ε = 1.e-8, maxiterstep=100)

Implicit solver for 
```math
f_{\mathrm{int}}(u) = f_{\mathrm{ext}}
```
``u`` is the displacement. We apply the Newton-Raphson algorithm
```math
u_{n+1} = u_n -  \nabla f_{\mathrm{int}}(u^n)^{-1} *( f_{\mathrm{int}}(u^n) -  f_{\mathrm{ext}})
```
"""
function StaticSolver(globdat, domain, loaditerstep = 10, ε = 1.e-8, maxiterstep=100)
    
    fext = domain.fext
    globdat.Dstate = copy(globdat.state)
    for iterstep = 1:loaditerstep

        # Newton's method
        Newtoniterstep, Newtonconverge = 0, false
        
        while  !Newtonconverge

            Newtoniterstep += 1
            
            fint, stiff = assembleStiffAndForce( globdat, domain )
            # #@show "fint", fint, "stiff", stiff
            # #@show "fext", fext
       
            res = fint - iterstep/loaditerstep * fext

            Δstate = stiff\res

            globdat.state -= Δstate
            #@show Newtoniterstep, norm(res)
            if (norm(res) < ε  || Newtoniterstep > maxiterstep)
                if Newtoniterstep > maxiterstep
                    @error "$Newtoniterstep Newton iteration cannot converge"
                end
                Newtonconverge = true
            end
            updateStates!(domain, globdat)
        end
        commitHistory(domain)
        globdat.Dstate = copy(globdat.state)
    end
end



@doc """
    Adaptive Solver, adjust the time step, if this step fails, redo the step with half of
    the time step size
    
    return globdat, domain
"""->
function AdaptiveSolver(solvername, globdat, domain, T, NT, args)

    failsafe = true
    ts = Float64[]

    Δt = T/NT #specified(maximum) time step
    dt = T/NT #current time step
    t = 0.0   #current time
    push!(ts, t)

    if solvername == "NewmarkSolver"

        ρ_oo = args["Newmark_rho"]
        η = args["damped_Newton_eta"]
        maxiterstep = args["Newton_maxiter"]
        ε = args["Newton_Abs_Err"]
        ε0 = args["Newton_Rel_Err"]
       

        αm = (2*ρ_oo - 1)/(ρ_oo + 1)
        αf = ρ_oo/(ρ_oo + 1)

        convergeCounter = 0
        while t < T
            if t + dt > T 
                dt = T - t
            end
            printstyled("dt = $dt, t = $t, T=$T\n", color=:cyan)
            
            convergeOrNot = NewmarkSolver(dt, globdat, domain, αm, αf, ε, ε0, maxiterstep, η, failsafe)
            
            if convergeOrNot
                convergeCounter += 1
                t += dt
                push!(ts, t)
                @assert globdat.time ≈ t
                if dt < 0.8*Δt  && convergeCounter >=5
                    dt = 2*dt
                end

                

            else
                @assert globdat.time ≈ t
                convergeCounter = 0
                dt /= 2.0

                @warn("Repeat time step with dt = ", dt)
            end
        end

    else
        @error("AdaptiveSolve has not implemented for ", solvername)
    end

    return globdat, domain, ts
    
end
    
    
