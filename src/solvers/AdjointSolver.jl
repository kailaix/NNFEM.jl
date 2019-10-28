export  AdjNewmarkSolver 

# return the strain, dstrain_dstate
function computeStrain(globdat::GlobalData, domain::Domain)

end

# return the Fint, pFint_pstate, pFint_pstress
function assembleStiffAndForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)
    Fint = zeros(Float64, domain.neqs)
    # K = zeros(Float64, domain.neqs, domain.neqs)
    ii = Int64[]; jj = Int64[]; vv = Float64[]
    neles = domain.neles
  
    # Loop over the elements in the elementGroup
    for iele  = 1:neles
      element = domain.elements[iele]
  
      # Get the element nodes
      el_nodes = getNodes(element)
  
      # Get the element nodes
      el_eqns = getEqns(domain,iele)
  
      el_dofs = getDofs(domain,iele)
  
      el_state  = getState(domain,el_dofs)
  
      el_Dstate = getDstate(domain,el_dofs)
      
  
      # Get the element contribution by calling the specified action
      #@info "ele id is ", iele
      fint, stiff  = getStiffAndForce(element, el_state, el_Dstate, Δt)
  
      # Assemble in the global array
      el_eqns_active = el_eqns .>= 1
      # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
      Slocal = stiff[el_eqns_active,el_eqns_active]
      Idx = el_eqns[el_eqns_active]
      for i = 1:length(Idx)
        for j = 1:length(Idx)
          push!(ii, Idx[i])
          push!(jj, Idx[j])
          push!(vv, Slocal[i,j])
        end
      end
      Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]
      # @info "Fint is ", Fint
    end
    Ksparse = sparse(ii, jj, vv, domain.neqs, domain.neqs)
    # @show norm(K-Array(Ksparse))
    return Fint, Ksparse
  end


  @doc """
  Implicit solver for Ma + C v + R(u) = P
  a, v, u are acceleration, velocity and displacement

  u_{n+1} = u_n + dtv_n + dt^2/2 ((1 - 2\beta)a_n + 2\beta a_{n+1})
  v_{n+1} = v_n + dt((1 - gamma)a_n + gamma a_{n+1})

  M a_{n+0.5} + fint(u_{n+0.f}) = fext_{n+0.5}

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

  
  fext = getExternalForce(domain, globdat)


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




function assembleAdjointMatrix(globdat::GlobalData, domain::Domain, state, Dstate, stress, dstress_dstrain)
    DFintDstate = zeros(Float64, domain.neqs)
    # K = zeros(Float64, domain.neqs, domain.neqs)
    ii = Int64[]; jj = Int64[]; vv = Float64[]
    neles = domain.neles
  
    # Loop over the elements in the elementGroup
    for iele  = 1:neles
      element = domain.elements[iele]
  
      # Get the element nodes
      el_nodes = getNodes(element)
  
      # Get the element nodes
      el_eqns = getEqns(domain,iele)
  
      el_dofs = getDofs(domain,iele)
  
      el_state  = getState(domain, el_dofs)
  
  
      # Get the element contribution by calling the specified action
      DfintDstate, DfintDstress  = getStiffAndForce(element, el_state, el_Dstate, Δt)
  
      # Assemble in the global array
      el_eqns_active = el_eqns .>= 1
      # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
      Slocal = stiff[el_eqns_active,el_eqns_active]
      Idx = el_eqns[el_eqns_active]
      for i = 1:length(Idx)
        for j = 1:length(Idx)
          push!(ii, Idx[i])
          push!(jj, Idx[j])
          push!(vv, Slocal[i,j])
        end
      end
     
    end
     
    dfint_dstress_tran =  sparse(ii, jj, vv, ngp*nstrain, neqs)
    dstrain_dstate_tran = sparse(ii, jj, vv, neqs, ngp*nstrain)
    stiff_tran = sparse(ii, jj, vv, domain.neqs, domain.neqs)
    
    return dfint_dstress_tran, stiff_tran, dstrain_dstate_tran
  end

  

function assembleAdjointNNStressMatrix(strain, Dstrain, stress, nn)
    # return ngp by 3 by 3 matrix

    pnn_pstrain_tran,  pnn_pdstrain_tran, pnn_pstress_tran, pnn_ptheta_tran 
end

function dJ_dstate(state[i, :], obs_state[i,:])
end

@doc """
    Implicit solver for Ma + C v + R(u) = P
    a, v, u are acceleration, velocity and displacement

    u_{n+1} = u_n + dtv_n + dt^2/2 ((1 - 2\beta)a_n + 2\beta a_{n+1})
    v_{n+1} = v_n + dt((1 - gamma)a_n + gamma a_{n+1})

    M a_{n+0.5} + fint(u_{n+0.f}) = fext_{n+0.5}

    αm = (2\rho_oo - 1)/(\rho_oo + 1)
    αf = \rho_oo/(\rho_oo + 1)
    
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf

    absolution error ε = 1e-8, 
    relative error ε0 = 1e-8  
    
    return true or false indicating converging or not
"""->



function AdjointNewmarkSolver(Δt, globdat, domain, αm = -1.0, αf = 0.0)
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf

    adj_lambda = zeros(nt+2,ndof)
    adj_tau = zeros(nt+2,ndof)
    adj_kappa = zeros(nt+2,ndof)
    adj_sigma = zeros(nt+2,ngp*3)

    rhs, temp = zeros(nt), zeros(nt)

    J = 0.0

    for i = NT:1
        
        #compute tau^i
        adj_tau[i,:] = dt * adj_lambda[i+1,:] + adj_tau[i+1,:]

        #compute kappa^i
        temp[:] = (dt*dt*(1-β2)/2.0*adj_lambda[i+1,:] + adj_tau[i,:]*dt*γ + adj_tau[i+1,:]*dt*(1.0-γ)) - MT*(αm*adj_kappa[i+1,:]) 

        # dstrain_dstate_tran = dE_i/d d_i
        # dstrain_dstate_tran_p = dE_{i+1}/d d_{i+1}

        # pnn_pstrain_tran = pnn(E^i, E^{i-1}, S^{i-1})/pE^i
        # pnn_pdstrain_tran = pnn(E^i, E^{i-1}, S^{i-1})/pE^{i-1}
        # pnn_pstress_tran = pnn(E^i, E^{i-1}, S^{i-1})/pS^{i-1}

        rhs[:] = (dJ_dstate(state[i, :], obs_state[i,:]) + adj_lambda[i+1,:] 
               +  dstrain_dstate_tran*(pnn_pdstrain_tran_p * adj_sigma[i+1,:]) 
               +  dstrain_dstate_tran*(pnn_pstrain_tran * (pnn_pstress_tran_p * adj_sigma[i+1,:]))*(dt*dt/2.0*β2) + temp

        adj_kappa[i,:] = (MT*(1 - αm) + stiff_tran*(dt*dt/2.0*β2))\res


        rhs[:] = MT * ((1 - αm)*adj_kappa[i,:]) - temp 
        adj_lambda[i,:] = rhs/(dt*dt/2.0 * β2)

        adj_sigma[i,:] = -dfint_dstress_tran*adj_kappa[i,:] + pnn_pstress_tran_p*adj_sigma[i+1,:]

        J += 
        dJ += pnn_ptheta_tran * adj_sigma[i,:]

        dstrain_dstate_tran_p = dstrain_dstate_tran
        pnn_pdstrain_tran_p = pnn_pdstrain_tran
        pnn_pstress_tran_p = pnn_pstress_tran

    end

    
end 


  
  @doc """
    Implicit solver for Ma + C v + R(u) = P
    a, v, u are acceleration, velocity and displacement

    u_{n+1} = u_n + dtv_n + dt^2/2 ((1 - 2\beta)a_n + 2\beta a_{n+1})
    v_{n+1} = v_n + dt((1 - gamma)a_n + gamma a_{n+1})

    M a_{n+0.5} + fint(u_{n+0.f}) = fext_{n+0.5}

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

    
    fext = getExternalForce(domain, globdat)


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
