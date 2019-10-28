export AdjointAssembleStrain, AssembleStiffAndForce, AdjointAssembleStiff
@doc """
Compute the strain, based on the state in domain
and dstrain_dstate
"""->
function AdjointAssembleStrain(domain)
  neles = domain.neles
  eledim = domain.elements[1].eledim
  nstrain = div((eledim + 1)*eledim, 2)
  ngps_per_elem = length(domain.elements[1].weights)
  neqs = domain.neqs


  strain = zeros(Float64, neles*ngps_per_elem, nstrain)
  # dstrain_dstate = zeros(Float64, neles*ngps_per_elem, domain.neqs)
  ii = Int64[]; jj = Int64[]; vv = Float64[]

  # Loop over the elements in the elementGroup
  for iele  = 1:neles
    element = domain.elements[iele]

    # Get the element nodes
    el_nodes = getNodes(element)

    # Get the element nodes
    el_eqns = getEqns(domain,iele)

    el_dofs = getDofs(domain,iele)

    el_state  = getState(domain, el_dofs)

    # Get strain{ngps_per_elem, nstrain} 
    #     dstrain_dstate{ngps_per_elem*nstrain, neqs_per_elem}  
    strain[iele*ngps_per_elem+1 : iele*ngps_per_elem+ngps_per_elem,:], ldstrain_dstate = 
    getStrainState(element, el_state)


      # Assemble in the global array
      el_eqns_active = el_eqns .>= 1
      el_eqns_active_idx = el_eqns[el_eqns_active]
    
      ldstrain_dstate_active = ldstrain_dstate[:,el_eqns_active]
    
      for i = 1:ngps_per_elem*nstrain
        for j = 1:length(el_eqns_active_idx)
          push!(ii, iele*ngps_per_elem*nstrain+i)
          push!(jj, el_eqns_active_idx[j])
          push!(vv, ldstrain_dstate_active[i,j])
        end
      end
  end
   
  dstrain_dstate_tran = sparse(jj, ii, vv, neqs, neles*ngps_per_elem*nstrain)
  return strain, dstrain_dstate_tran
end


@doc """
Compute the fint and stiff, based on the state and Dstate in domain
"""->
function AssembleStiffAndForce(domain, stress::Array{Float64}, dstress_dstrain::Array{Float64})
  neles = domain.neles
  ngps_per_elem = length(domain.elements[1].weights)
  neqs = domain.neqs
  
  
  fint = zeros(Float64, domain.neqs)
  # K = zeros(Float64, domain.neqs, domain.neqs)
  ii = Int64[]; jj = Int64[]; vv = Float64[]

  # Loop over the elements in the elementGroup
  for iele  = 1:neles
    element = domain.elements[iele]

    # Get the element nodes
    el_nodes = getNodes(element)

    # Get the element nodes
    el_eqns = getEqns(domain,iele)

    el_dofs = getDofs(domain,iele)

    #@show "iele", iele, el_dofs 
    
    #@show "domain.state", iele, domain.state 

    el_state  = getState(domain,el_dofs)

    el_Dstate = getDstate(domain,el_dofs)

    gp_ids = iele*ngps_per_elem+1 : iele*ngps_per_elem+ngps_per_elem
    lfint, lstiff  = getStiffAndForce(element, el_state, el_Dstate, stress[gp_ids,:], dstress_dstrain[gp_ids,:,:])

    # Assemble in the global array
    el_eqns_active = el_eqns .>= 1
    # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
    lstiff_active = lstiff[el_eqns_active,el_eqns_active]
    el_eqns_active_idx = el_eqns[el_eqns_active]
    for i = 1:length(el_eqns_active_idx)
      for j = 1:length(el_eqns_active_idx)
        push!(ii, el_eqns_active_idx[i])
        push!(jj, el_eqns_active_idx[j])
        push!(vv, lstiff_active[i,j])
      end
    end
    fint[el_eqns[el_eqns_active]] += lfint[el_eqns_active]
    # @info "Fint is ", Fint
  end
  stiff = sparse(ii, jj, vv, neqs, neqs)
  # @show norm(K-Array(Ksparse))
  return fint, stiff
end




@doc """
Compute the stiff and dfint_dstress, based on the state in domain
and dstrain_dstate
"""->
function adjointAssembleStiff(domain, stress::Array{Float64}, dstress_dstrain::Array{Float64})
    neles = domain.neles
    eledim = domain.elements[1].eledim
    nstrain = div((eledim + 1)*eledim, 2)
    ngps_per_elem = length(domain.elements[1].weights)
    neqs = domain.neqs


    ii_stiff = Int64[]; jj_stiff = Int64[]; vv_stiff = Float64[]
    ii_dfint_dstress = Int64[]; jj_dfint_dstress = Int64[]; vv_dfint_dstress = Float64[]


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
  

      gp_ids = iele*ngps_per_elem : iele*ngps_per_elem+ngps_per_elem-1
      # Get the element contribution by calling the specified action
      stiff, dfint_dstress = getStiffAndDforceDstress(element, el_state, el_Dstate, stress[gp_ids,:], dstress_dstrain[gp_ids,:,:])
  
      # Assemble in the global array
      el_eqns_active = el_eqns .>= 1
      el_eqns_active_idx = el_eqns[el_eqns_active]
      # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
      stiff_active = stiff[el_eqns_active,el_eqns_active]
      dfint_dstress_active = dfint_dstress[el_eqns_active,:]
      el_eqns_active_idx = el_eqns[el_eqns_active]

      for i = 1:length(el_eqns_active_idx)
        for j = 1:length(el_eqns_active_idx)
          push!(ii_stiff, el_eqns_active_idx[i])
          push!(jj_stiff, el_eqns_active_idx[j])
          push!(vv_stiff, stiff_active[i,j])
        end
      end


      for i = 1:length(el_eqns_active_idx)
        for j = 1:ngps_per_elem*nstrain
          push!(ii_dfint_dstress, el_eqns_active_idx[i])
          push!(jj_dfint_dstress, iele*ngps_per_elem*nstrain+j)
          push!(vv_dfint_dstress, dfint_dstress_active[i,j])
        end
      end

     
    end

    stiff_tran = sparse(jj_stiff, jj_stiff, vv_stiff, neqs, neqs) 
    dfint_dstress_tran =  sparse(jj_dfint_dstress, ii_dfint_dstress, vv_dfint_dstress, neles*ngps_per_elem*nstrain, neqs)
  
    return stiff_tran, dfint_dstress_tran
  end

  
#=
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

function BackwardNewmarkSolver(Δt, globdat, domain, αm = -1.0, αf = 0.0)
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

=#