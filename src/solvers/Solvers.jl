using LinearAlgebra
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
    disp = globdat.state
    velo = globdat.velo
    acce = globdat.acce

    fext = domain.fext
    
    velo += 0.5*Δt * acce
    disp += Δt * velo
    
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
function NewmarkSolver(Δt, globdat, domain, β = 0.25, γ = 0.5, ε = 1e-8, maxiterstep=100)
    
    globdat.time  += Δt

    M = domain.M
    acce_n = globdat.acce; acce_np = globdat.acce
    disp_n = globdat.disp
    vel_n  = globdat.vel

    fext = domain.fext
    #Newton solve for a_{n+1}
    Newtoniterstep, Newtonconverge = 0, false
    while !Newtonconverge

        #validate gradient
        #app.check_derivative(u_n)
        Newtoniterstep += 1

        acce_nh = 0.5*(acce_n + acce_np)
        vel_nh = vel_n + 0.5*Δt*((1 - γ)*acce_n + γ*acce_np)
        disp_nh = disp_n + 0.5*Δt*vel_n + 0.25*Δt*Δt*((1 - 2*β)*acce_n + 2*β*acce_np)

        t_nh = t_n + Δt/2.0

        fint, stiff = assembleStiffAndForce( globdat, domain )

        res = M * acce_nh + fint - fext

        A = 0.5*M/2.0 + 0.5*Δt*Δt*β * stiff

        Δacce_np = np.linalg.solve(A, res)

        acce_np -= Δacce_np

        if (norm(res) < ε || Newtoniterstep > maxiterstep)
            if Newtoniterstep > maxiterstep
                @info "Newton iteration cannot converge"
            end
            Newtonconverge = true
        end
    end

    globdat.acce[:] = acce_np[:] 
    globdat.disp[:] += Δt * vel_n + Δt * Δt / 2.0 * ((1 - 2β) * acce_n + 2 * β * acce_np)
    globdat.vel[:] += Δt * ((1 - γ) * acce_n + γ * acce_np)

    # todo update historic parameters
    # globdat.elements.commitHistory()
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

    for iterstep = 1:loaditerstep

        # Newton's method
        Newtoniterstep, Newtonconverge = 0, false

        while  !Newtonconverge

            NewtonIterstep += 1

            fint, stiff = assembleStiffAndForce( globdat, domain )

            res = fint - iterstep/loaditerstep * fext

            Δdisp = stiff\res

            globdat.disp[:] -= Δdisp[:]


            if (np.linalg.norm(RHS) < ε  || Newtoniterstep > maxiterstep)
                if Newtoniterstep > maxiterstep
                    @info "Newton iteration cannot converge"
                end
                NewtonConverge = true
            end
        end
    end
end
