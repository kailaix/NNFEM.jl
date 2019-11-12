export AdjointAssembleStrain, AssembleStiffAndForce, AdjointAssembleStiff,
ForwardNewmarkSolver, BackwardNewmarkSolver,constitutive_law


function constitutive_law(input::Array{Float64,2}, θ::Array{Float64,1}, 
  g::Union{Array{Float64,2}, Nothing}=nothing, grad_input::Bool=false, grad_θ::Bool=false; 
  strain_scale::Float64, stress_scale::Float64)
  input_ = zero(input)
  input_[:,1:6,:] = input[:,1:6,:]/strain_scale
  input_[:,7:9,:] = input[:,7:9,:]/stress_scale

  config = [9, 20, 20, 20, 4]      
  out, g_input, g_θ = nn_constitutive_law(input_, θ, config, g, grad_input, grad_θ)
  out *= stress_scale
  if grad_θ
    g_θ *= stress_scale
  end
  if grad_input
    g_input[:,1:6,:] *= stress_scale/strain_scale
  end
  out, g_input, g_θ
end

@doc """
Compute the strain, based on the state in domain
and dstrain_dstate
"""->
function AdjointAssembleStrain(domain, computeDstrain::Bool=true)
  neles = domain.neles
  eledim = domain.elements[1].eledim
  nstrain = div((eledim + 1)*eledim, 2)
  ngps_per_elem = length(domain.elements[1].weights)
  neqs = domain.neqs
  
  
  strain = zeros(Float64, neles*ngps_per_elem, nstrain)
  # dstrain_dstate = zeros(Float64, neles*ngps_per_elem, domain.neqs)
  
  
  # Loop over the elements in the elementGroup
  for iele  = 1:neles
    element = domain.elements[iele]
    
    # Get the element nodes
    el_eqns = getEqns(domain,iele)
    
    el_dofs = getDofs(domain,iele)
    
    el_state  = getState(domain, el_dofs)
    
    # Get strain{ngps_per_elem, nstrain} 
    #     dstrain_dstate{ngps_per_elem*nstrain, neqs_per_elem}  
    strain[(iele-1)*ngps_per_elem+1 : iele*ngps_per_elem,:], ldstrain_dstate = 
    getStrainState(element, el_state)
    
    if computeDstrain
      # Assemble in the global array
      el_eqns_active = el_eqns .>= 1
      el_eqns_active_idx = el_eqns[el_eqns_active]
      
      ldstrain_dstate_active = ldstrain_dstate[:,el_eqns_active]
      
      domain.vv_dstrain_dstate[domain.vv_dstrain_dstate_ele_indptr[iele]:domain.vv_dstrain_dstate_ele_indptr[iele+1] - 1] = ldstrain_dstate_active[:]
      
    end
  end
  dstrain_dstate_tran = computeDstrain ? sparse(domain.jj_dstrain_dstate, domain.ii_dstrain_dstate, domain.vv_dstrain_dstate, neqs, neles*ngps_per_elem*nstrain) : nothing;
  return strain, dstrain_dstate_tran
end


@doc """
Compute the fint and stiff, based on the state and Dstate in domain
"""->
function AssembleStiffAndForce(domain, stress::Array{Float64}, dstress_dstrain_T::Array{Float64})
  neles = domain.neles
  ngps_per_elem = length(domain.elements[1].weights)
  neqs = domain.neqs
  
  
  fint = zeros(Float64, domain.neqs)
  # K = zeros(Float64, domain.neqs, domain.neqs)
  
  
  # Loop over the elements in the elementGroup
  for iele  = 1:neles
    element = domain.elements[iele]
    
    # Get the element nodes
    el_eqns = getEqns(domain,iele)
    
    el_dofs = getDofs(domain,iele)
    
    #@show "iele", iele, el_dofs 
    
    #@show "domain.state", iele, domain.state 
    
    el_state  = getState(domain,el_dofs)
    
    gp_ids = (iele-1)*ngps_per_elem+1 : iele*ngps_per_elem
    
    lfint, lstiff  = getStiffAndForce(element, el_state, stress[gp_ids,:], dstress_dstrain_T[gp_ids,:,:])
    
    # Assemble in the global array
    el_eqns_active = el_eqns .>= 1
    lstiff_active = lstiff[el_eqns_active,el_eqns_active]
    domain.vv_stiff[domain.vv_stiff_ele_indptr[iele]:domain.vv_stiff_ele_indptr[iele+1] - 1] = lstiff_active[:]
    
    
    fint[el_eqns[el_eqns_active]] += lfint[el_eqns_active]
    # @info "Fint is ", Fint
  end
  #@assert (norm(vv_stiff - domain.vv_stiff)) == 0.0
  stiff = sparse(domain.ii_stiff, domain.jj_stiff, domain.vv_stiff, neqs, neqs)
  # @show norm(K-Array(Ksparse))
  return fint, stiff
end




@doc """
Compute the stiff and dfint_dstress, based on the state in domain
and dstrain_dstate
"""->
function AdjointAssembleStiff(domain, stress::Array{Float64}, dstress_dstrain_T::Array{Float64})
  neles = domain.neles
  eledim = domain.elements[1].eledim
  nstrain = div((eledim + 1)*eledim, 2)
  ngps_per_elem = length(domain.elements[1].weights)
  neqs = domain.neqs
  
  
  
  neles = domain.neles
  
  # Loop over the elements in the elementGroup
  for iele  = 1:neles
    element = domain.elements[iele]
    
    # Get the element nodes
    el_eqns = getEqns(domain,iele)
    
    el_dofs = getDofs(domain,iele)
    
    el_state  = getState(domain, el_dofs)
    
    
    gp_ids = (iele-1)*ngps_per_elem+1 : iele*ngps_per_elem
    # Get the element contribution by calling the specified action
    stiff, dfint_dstress = getStiffAndDforceDstress(element, el_state, stress[gp_ids,:], dstress_dstrain_T[gp_ids,:,:])
    
    
    # Assemble in the global array
    el_eqns_active = el_eqns .>= 1
    el_eqns_active_idx = el_eqns[el_eqns_active]
    # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
    stiff_active = stiff[el_eqns_active,el_eqns_active]
    dfint_dstress_active = dfint_dstress[el_eqns_active,:]
    
    domain.vv_stiff[domain.vv_stiff_ele_indptr[iele]:domain.vv_stiff_ele_indptr[iele+1] - 1] = stiff_active[:]
    domain.vv_dfint_dstress[domain.vv_dfint_dstress_ele_indptr[iele]:domain.vv_dfint_dstress_ele_indptr[iele+1] - 1] = dfint_dstress_active[:]
    
    
    
  end
  
  
  stiff_tran = sparse(domain.jj_stiff, domain.ii_stiff, domain.vv_stiff, neqs, neqs) 
  dfint_dstress_tran =  sparse(domain.jj_dfint_dstress, domain.ii_dfint_dstress, domain.vv_dfint_dstress, neles*ngps_per_elem*nstrain, neqs)
  
  return stiff_tran, dfint_dstress_tran
end



function computDJDstate(state, obs_state)
  #J = (state - obs_state).^2
  #@show  norm(2.0*(state - obs_state))
  2.0*(state - obs_state)
end

function computeJ(state, obs_state)
  
  #@show sum((state - obs_state).^2)
  sum((state - obs_state).^2)
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
  
  make sure
  state[NT+1, neqs], strain[NT+1, ngps_per_elem*neles, nstrain], stress[NT+1, ngps_per_elem*neles, nstrain]
  state, obs_state strain, stress have NT+1 frames, the first time step corresponds to the initial condition
  
  return dJ
  """->
  function BackwardNewmarkSolver(globdat, domain, theta::Array{Float64},
    T::Float64, NT::Int64, state::Array{Float64}, strain::Array{Float64}, stress::Array{Float64},
    strain_scale::Float64, stress_scale::Float64, obs_state::Array{Float64}, αm::Float64 = -1.0, αf::Float64 = 0.0)
    Δt = T/NT
    β2 = 0.5*(1 - αm + αf)^2
    γ = 0.5 - αm + αf
    neles, ngps_per_elem, neqs = domain.neles, length(domain.elements[1].weights), domain.neqs
    nstrain = 3
    
    adj_lambda = zeros(Float64, NT+1,neqs)
    adj_tau = zeros(Float64,NT+1,neqs)
    adj_kappa = zeros(Float64,NT+1,neqs)
    adj_sigma = zeros(Float64,NT+1,neles*ngps_per_elem, nstrain)
    
    MT = (globdat.M)'
    dJ = zeros(Float64,length(theta))
    
    
    
    
    # dstrain_dstate_tran = dE_i/d d_i
    # dstrain_dstate_tran_p = dE_{i+1}/d d_{i+1}
    
    # pnn_pstrain_tran = pnn(E^i, E^{i-1}, S^{i-1})/pE^i
    # pnn_pstrain0_tran = pnn(E^i, E^{i-1}, S^{i-1})/pE^{i-1}
    # pnn_pstress0_tran = pnn(E^i, E^{i-1}, S^{i-1})/pS^{i-1}
    
    # pnn_pstrain_tran_p = pnn(E^{i+1}, E^{i}, S^{i})/pE^{i+1}
    # pnn_pstrain0_tran_p = pnn(E^{i+1}, E^{i}, S^{i})/pE^{i}
    # pnn_pstress0_tran_p = pnn(E^{i+1}, E^{i}, S^{i})/pS^{i}
    
    pnn_pstrain0_tran_p = zeros(Float64, neles*ngps_per_elem, nstrain, nstrain) 
    pnn_pstrain0_tran = Array{Float64}(undef, neles*ngps_per_elem, nstrain, nstrain)
    pnn_pstress0_tran_p = zeros(Float64, neles*ngps_per_elem, nstrain, nstrain) 
    pnn_pstress0_tran = Array{Float64}(undef, neles*ngps_per_elem, nstrain, nstrain)
    pnn_pstrain_tran = Array{Float64}(undef, neles*ngps_per_elem, nstrain, nstrain)
    
    # temporal variables
    rhs = Array{Float64}(undef, neqs)  
    temp = Array{Float64}(undef, neqs)
    tempmult = Array{Float64}(undef, nstrain, neles*ngps_per_elem)
    output = Array{Float64}(undef, neles*ngps_per_elem, 3*nstrain,  nstrain)
    sigmaTdstressdtheta = similar(dJ) 
    
    
    for i = NT:-1:1
      # get strain
      domain.state[domain.dof_to_eq] = state[i+1,:]
      _, dstrain_dstate_tran = AdjointAssembleStrain(domain)
      
      #@show size(strain[i+1,:,:]), size(strain[i,:,:]), size(stress[i,:,:])
      _, output[:,:,:], _ =  constitutive_law([strain[i+1,:,:] strain[i,:,:] stress[i,:,:]], theta, nothing, true, false, strain_scale=strain_scale, stress_scale=stress_scale)
      
      pnn_pstrain_tran[:,:,:], pnn_pstrain0_tran[:,:,:], pnn_pstress0_tran[:,:,:] = output[:,1:3,:], output[:,4:6,:], output[:,7:9,:]
      
      stiff_tran, dfint_dstress_tran = AdjointAssembleStiff(domain, stress[i+1,:,:], pnn_pstrain_tran)
      
      
      #compute tau^i
      adj_tau[i,:] = Δt * adj_lambda[i+1,:] + adj_tau[i+1,:]
      
      #compute kappa^i
      temp[:] = (Δt*Δt*(1-β2)/2.0*adj_lambda[i+1,:] + Δt*γ*adj_tau[i,:] + Δt*(1.0-γ)*adj_tau[i+1,:]) - MT*(αm*adj_kappa[i+1,:]) 
      
      
      
      rhs[:] = computDJDstate(state[i+1, :], obs_state[i+1,:]) + adj_lambda[i+1,:] 
      
      for j = 1:neles*ngps_per_elem
        tempmult[:,j] = pnn_pstrain0_tran_p[j,:,:]*adj_sigma[i+1,j,:] 
      end
      rhs +=  dstrain_dstate_tran* tempmult[:]
      
      for j = 1:neles*ngps_per_elem
        tempmult[:,j] = pnn_pstrain_tran[j, :, :] *(pnn_pstress0_tran_p[j,:,:]*adj_sigma[i+1,j,:])
      end
      rhs +=  dstrain_dstate_tran*tempmult[:]
      
      
      rhs[:] = rhs*(Δt*Δt/2.0*β2) + temp
      
      adj_kappa[i,:] = (MT*(1 - αm) + stiff_tran*(Δt*Δt/2.0*β2))\rhs
      
      
      rhs[:] = MT * ((1 - αm)*adj_kappa[i,:]) - temp 
      adj_lambda[i,:] = rhs/(Δt*Δt/2.0 * β2)
      
      
      for j = 1:neles*ngps_per_elem
        tempmult[:,j] = pnn_pstress0_tran_p[j,:,:]*adj_sigma[i+1,j,:]
      end
      
      #@assert norm(tempmult)==0.0
      
      adj_sigma[i,:,:] = (reshape(-dfint_dstress_tran*adj_kappa[i,:], nstrain, neles*ngps_per_elem) + tempmult)'
      
      _, _, sigmaTdstressdtheta[:] =  constitutive_law([strain[i+1,:,:] strain[i,:,:] stress[i,:,:]], theta, adj_sigma[i,:,:], false, true, strain_scale=strain_scale, stress_scale=stress_scale)
      
      
      dJ += sigmaTdstressdtheta
      
      pnn_pstrain0_tran_p[:,:,:] = pnn_pstrain0_tran
      
      pnn_pstress0_tran_p[:,:,:] = pnn_pstress0_tran
      
    end
    return dJ
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
    
    make sure
    globdat.time  = 0.0
    domain.state, domain.velo, and domain.acce neqs parts are the initial conditions
    globdat.acce, globdat.state, globdat.velo are the intial conditions
    
    obs_state has NT+1, the first time step corresponds to the initial condition
    
    return J
    state[NT+1, neqs], strain[NT+1, ngps_per_elem*neles, nstrain], stress[NT+1, ngps_per_elem*neles, nstrain]
    the first time step corresponds to the initial condition
    """->
    function ForwardNewmarkSolver(globdat, domain, theta::Array{Float64},
      T::Float64, NT::Int64, strain_scale::Float64, stress_scale::Float64, 
      obs_state::Array{Float64}, 
      #output 
      state::Array{Float64, 2}, strain::Array{Float64, 3}, stress::Array{Float64, 3},
      #Newmark and Newton Parameter
      αm::Float64 = -1.0, αf::Float64 = 0.0, ε::Float64 = 1e-8, 
      ε0::Float64 = 1e-8, maxiterstep::Int64=10)
      
      Δti = T/NT
      β2 = 0.5*(1 - αm + αf)^2
      γ = 0.5 - αm + αf
      neles, ngps_per_elem, neqs = domain.neles, length(domain.elements[1].weights), domain.neqs
      nstrain = 3
      local norm_res0      
      
      # initialize globdat and domain
      fill!(globdat.state, 0.0)
      fill!(globdat.velo, 0.0)
      fill!(globdat.acce, 0.0)
      globdat.time = 0.0
      
      M = globdat.M
      J = 0.0 
      
      # 1: initial condition, compute 2, 3, 4 ... NT+1
      # state = zeros(Float64, NT+1,neqs)
      # strain = zeros(Float64, NT+1, neles*ngps_per_elem, nstrain)
      # stress = zeros(Float64, NT+1, neles*ngps_per_elem, nstrain) 
      @assert(size(state) == (NT+1, neqs))
      @assert(size(strain) == (NT+1, neles*ngps_per_elem, nstrain))
      @assert(size(stress) == (NT+1, neles*ngps_per_elem, nstrain))
      
      # temporal variables
      ∂∂up = Array{Float64}(undef, neqs)  
      Δ∂∂u = Array{Float64}(undef, neqs)
      res = Array{Float64}(undef, neqs)
      fint = Array{Float64}(undef, neqs)
      fext = Array{Float64}(undef, neqs)
      output = Array{Float64}(undef, neles*ngps_per_elem, 3*nstrain,  nstrain) 
      
      Ni, i = 0.0, 1
      MinStepSize = 1.0/2.0^10
      stepsize = 1.0
      convergeCounter = 0
      
      while i < NT+1
        
        Δt = Δti*stepsize
        
        failSafeTime =  globdat.time 
        globdat.time  += (1 - αf)*Δt
        
        updateDomainStateBoundary!(domain, globdat)
        
        getExternalForce!(domain, globdat, fext)
        
        ∂∂up[:] = globdat.acce
        
        Newtoniterstep, Newtonconverge = 0, false
        
        while !Newtonconverge && Newtoniterstep < maxiterstep 
          
          Newtoniterstep += 1
     
          
          domain.state[domain.eq_to_dof] = (1 - αf)*(Δt*globdat.velo + 0.5 * Δt * Δt * ((1 - β2)*globdat.acce + β2*∂∂up)) + globdat.state
          
          strain[i+1, :,:], _ = AdjointAssembleStrain(domain, false)
          stress[i+1, :,:], output[:,:,:], _ =  constitutive_law([strain[i+1,:,:] strain[i,:,:] stress[i,:,:]], theta, nothing, true, false, strain_scale=strain_scale, stress_scale=stress_scale)
          pnn_pstrain_tran = output[:,1:3,:]
          
          fint, stiff = AssembleStiffAndForce(domain, stress[i+1, :,:], pnn_pstrain_tran)

          @show norm(fint), norm(fext)
          
          res[:] = M * (∂∂up *(1 - αm) + αm*globdat.acce)  + fint - fext
          
          norm_res = norm(res)

          if Newtoniterstep==1
            norm_res0 = norm_res 
          end
          
          A = (M*(1 - αm) + (1 - αf) * 0.5 * β2 * Δt^2 * stiff)
          
          Δ∂∂u[:] = A\res
          
          ∂∂up -= Δ∂∂u
          
          
          
          #println("$Newtoniterstep/$maxiterstep, $(norm(res))")
          if (norm_res < ε || norm_res < ε0*norm_res0) 
              Newtonconverge = true
          end
          @show norm_res, " / " , norm_res0
          
        end

        @show "After Newton, Ni = ", Ni
        
     
        
        if !Newtonconverge
          @show "!Newtonconverge"
          #revert the globdat time
          globdat.time  = failSafeTime
          stepsize /= 2.0
          convergeCounter = 0

          if stepsize < MinStepSize
            J = Inf
            return J
          end
        else  
          
          Ni += stepsize  
          convergeCounter += 1
          
          globdat.state += Δt * globdat.velo + Δt^2/2 * ((1 - β2) * globdat.acce + β2 * ∂∂up)
          globdat.velo += Δt * ((1 - γ) * globdat.acce + γ * ∂∂up)
          globdat.acce[:] = ∂∂up
          globdat.time  += αf*Δt
          
          
          
          if Ni ≈ i
            
            
            #save data 
            state[i+1,:] = globdat.state
            
            domain.state[domain.eq_to_dof] = globdat.state
            strain[i+1, :,:], _ = AdjointAssembleStrain(domain, false)
            stress[i+1, :,:], _, _ =  constitutive_law([strain[i+1,:,:] strain[i,:,:] stress[i,:,:]], theta, nothing, false, false, strain_scale=strain_scale, stress_scale=stress_scale)
            
            
            
            #update J 
            J += computeJ(state[i+1,:], obs_state[i+1,:])

            i += 1  

            @show i, Ni, size(state)
          
          end

          if convergeCounter  == 4
            stepsize = min(1.0, stepsize*2.0)
            convergeCounter = 0 
          end
        end


      end
      
      return J
    end