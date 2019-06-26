export ExplicitSolver, NewmarkSolver, StaticSolver

@doc """
Central Difference Explicit solver for Ma + fint = fext
    a, v, u are acceleration, velocity and displacement

    u_{n+1} = u_n + dtv_n + dt^2/2 a_n
    v_{n+1} = v_n + dt/2(a_n + a_{n+1})

    M a_{n+1} + C v_{n+1} + R(u_{n+1}) = P_{n+1}
    (M + dt/2 C) a_{n+1} = P_{n+1} - R(u_{n+1}) - C(v_n + dt/2 a_{n})


    Alternative, todo:
    M a_n + C v_n + R(u_n) = P_n
    M(u_{n+1} - 2u_n + u_{n-1}) + dt/2*C(u_{n+1} - u_{n-1}) + dt^2 R(u_n) = dt^2 P_n
    (M + dt/2 C) u_{n+1} = dt^2(P_n - R(u_n) + dt/2 C u_{n-1} + M(2u_n - u_{n-1})

    For the first step
    u_{-1} = u_0 - dt*v_0 + dt^2/2 a_0
    a_0 = M^{-1}(-Cv_0 - R(u_0) + P_0)
"""->
function ExplicitSolver(Δt, globdat, domain)
    state = globdat.state
    velo = globdat.velo
    acce = globdat.acce

    fext = domain.fext
    
    velo += 0.5*Δt * acce
    state += Δt * velo
    
    fint  = assembleInternalForce( globdat, domain )

    if length(globdat.M)==0
        error("globalDat is not initialized, call `assembleMassMatrix!(globaldat, domain)`")
    end

    acce = (fext-fint)./globdat.Mlumped
    velo += 0.5 * Δt * acce
    globdat.acce[:] = acce[:]

    globdat.time  += Δt
    # globdat.elements.commitHistory()
end




@doc """
    Implicit solver for Ma + C v + R(u) = P
    a, v, u are acceleration, velocity and displacement

    u_{n+1} = u_n + dtv_n + dt^2/2 ((1 - 2\beta)a_n + 2\beta a_{n+1})
    v_{n+1} = v_n + dt((1 - gamma)a_n + gamma a_{n+1})

    M a_{n+0.5} + C v_{n+0.5} + R(u_{n+0.5}) = P_{n+0.5}
"""->
function NewmarkSolver(Δt, globdat, domain, β2 = 0.5, γ = 0.5, ε = 1e-8, maxiterstep=100)
    
    globdat.time  += Δt

    M = globdat.M
    ∂∂uk = copy(globdat.acce)
    u = copy(globdat.state)
    ∂u  = copy(globdat.velo)

    fext = domain.fext
    #Newton solve for a_{n+1}

    ∂∂u = copy(∂∂uk)
    Newtoniterstep, Newtonconverge = 0, false
    while !Newtonconverge

        #validate gradient
        #app.check_derivative(u_n)
        Newtoniterstep += 1

        domain.state[domain.eq_to_dof] = u + Δt * ∂u + (1-β2)/2*Δt^2*∂∂uk + β2/2*Δt^2*∂∂u
        # t_nh = t_n + Δt/2.0
        fint, stiff = assembleStiffAndForce( globdat, domain )
        
        res = M * ∂∂u + fint - fext

        A = M + β2/2 * Δt^2 * stiff

        Δ∂∂u = A\res
        ∂∂u -= Δ∂∂u

        println("$Newtoniterstep, $(norm(res))")
        if (norm(res) < ε || Newtoniterstep > maxiterstep)
            if Newtoniterstep > maxiterstep
                error("Newton iteration cannot converge $(norm(res))")
            end
            Newtonconverge = true
            printstyled("Newton converged $Newtoniterstep\n", color=:green)
        end
    end
    globdat.Dstate = copy(globdat.state)
    globdat.state += Δt * ∂u + 0.5 *Δt * Δt * ((1 - β2) * ∂∂uk + β2 * ∂∂u)
    globdat.velo += Δt * ((1 - γ) * ∂∂uk + γ * ∂∂u)

    # todo update historic parameters
    commitHistory(domain)
    updateStates(domain, globdat)
end



@doc """
    Implicit solver for fint(u) = fext
    u is the displacement
    Newton iteration, with time-stepping at P
    u_{n+1} = u_n -  dR(u)^{-1} *(R(u) - P)

    :param app:
    :param u_0:
    :return: u_{oo}
"""->
function StaticSolver(globdat, domain, loaditerstep = 10, ε = 1.e-8, maxiterstep=100)
    
    fext = domain.fext
    globdat.Dstate = copy(globdat.state)
    for iterstep = 1:loaditerstep

        # Newton's method
        Newtoniterstep, Newtonconverge = 0, false
        
        while  !Newtonconverge

            Newtoniterstep += 1
            
            fint, stiff = assembleStiffAndForce( globdat, domain )
            # @info "fint", fint, "stiff", stiff
            # @info "fext", fext
       
            res = fint - iterstep/loaditerstep * fext

            Δstate = stiff\res

            globdat.state -= Δstate
            @show Newtoniterstep, norm(res)
            if (norm(res) < ε  || Newtoniterstep > maxiterstep)
                if Newtoniterstep > maxiterstep
                    @error "$Newtoniterstep Newton iteration cannot converge"
                end
                Newtonconverge = true
            end
            updateStates(domain, globdat)
        end
        commitHistory(domain)
        globdat.Dstate = copy(globdat.state)
    end
end
