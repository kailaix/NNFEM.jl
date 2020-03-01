export assembleStiffAndForce,assembleInternalForce,assembleMassMatrix!,tfAssembleInternalForce



@doc """
    Numerically assemble internal force vector, compute local internal force f_int and then assemble to F_int
    - 'globdat': GlobalData
    - 'domain': Domain, finite element domain, for data structure
    - 'Δt':  Float64, current time step size
    Return F_int: Float64[neqs], internal force vector
"""->function assembleInternalForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)
    Fint = zeros(Float64, domain.neqs)
    neles = domain.neles
  
    # Loop over the elements in the elementGroup
    for iele  = 1:neles
        element = domain.elements[iele]
  
      # Get the element nodes
        el_nodes = getNodes(element)
  
      # Get the element nodes
        el_eqns = getEqns(domain, iele)
  
        el_dofs = getDofs(domain, iele)
  
        el_state  = getState(domain, el_dofs)
  
        el_Dstate = getDstate(domain, el_dofs)
  
      # Get the element contribution by calling the specified action
        fint = getInternalForce(element, el_state, el_Dstate, Δt)
  
      # Assemble in the global array
        el_eqns_active = (el_eqns .>= 1)
        Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]
    end
  
    return Fint
end


@doc """
    Tensorflow version of assembling internal force vector, compute local internal 
    force f_int and then assemble to F_int, which generates inverse problem automatically.
    
    - 'domain': Domain, finite element domain, for data structure
    - 'nn': Function strain -> stress, neural network constitutive law function 
    - 'E_all': PyObject(Float64)[neles*nGauss, nstrains], neles*nGauss is the number of Gaussian quadrature points, 
               nstrain is the number of strain components. All strains for the current time-step
    - 'DE_all': PyObject(Float64)[neles*nGauss, nstrains], neles*nGauss is the number of Gaussian quadrature points, 
                nstrain is the number of strain components. All strains for the previous time-step
    - 'w∂E∂u_all': PyObject(Float64)[neles*nGauss, ndofs_per_element, nstrains], neles*nGauss is the number of Gaussian quadrature points, 
                   ndofs_per_element is the number of freedoms per element, nstrain is the number of strain components.
                   Multiplication of the Gaussian weight and ∂E∂u^T for current time-step, 
            
    - 'σ0_all': PyObject(Float64)[neles*nGauss, nstrains], neles*nGauss is the number of Gaussian quadrature points, 
    nstrain is the number of strain components.  All stresses for the last time-step

    Return: internal force vector F_int, PyObject(Float64)[neqns] and the predicted stresses at the 
            current step σ_all, PyObject(Float64)[neles*nGauss, nstrains]
"""->function tfAssembleInternalForce(domain::Domain, nn::Function, E_all::PyObject, DE_all::PyObject, w∂E∂u_all::PyObject, σ0_all::PyObject)
    neles = domain.neles
    nGauss = length(domain.elements[1].weights)
    neqns_per_elem = length(getEqns(domain, 1))
    nstrains = size(E_all, 2) 
 
  
    @assert size(E_all) == (neles * nGauss, nstrains)
    @assert size(DE_all) == (neles * nGauss, nstrains)
    @assert size(σ0_all) == (neles * nGauss, nstrains)
  
  # el_eqns_all, the equation numbers related to the Gaussian points in the element, negative value means Drichlet boundary
    el_eqns_all = zeros(Int32, neles * nGauss, neqns_per_elem)
  # el_eqns_active_all = el_eqns_all > 0
    el_eqns_active_all = zeros(Bool, neles * nGauss, neqns_per_elem)
  
  # Loop over the elements in the elementGroup to construct el_eqns_active_all , el_eqns_all and w∂E∂u_all
    for iele  = 1:neles
    # Get the element nodes
        el_eqns = getEqns(domain, iele)
 
    # Assemble in the global array
        el_eqns_active_all[(iele - 1) * nGauss + 1:iele * nGauss,:] = repeat((el_eqns .>= 1)', nGauss, 1)
        el_eqns_all[(iele - 1) * nGauss + 1:iele * nGauss,:] = repeat(el_eqns', nGauss, 1)
    end

  # get stress at each Gaussian points
    σ_all = nn(E_all, DE_all, σ0_all)

  # compute fint at each Gaussian quadrature points, fints[igp] = w∂E∂u_all[igp] * σ_all[igp]
    fints = squeeze(tf.matmul(w∂E∂u_all, tf.expand_dims(σ_all, 2)))

  # call the Cpp assembler to construct Fint
    Fint = cpp_fint(fints, constant(el_eqns_all, dtype = Int32), constant(domain.neqs, dtype = Int32))

    return Fint, σ_all
end




# function tfAssembleInternalForce(domain::Domain, nn::Function, E_all::PyObject, DE_all::PyObject,
#      w∂E∂u_all::PyObject, σ0_all::PyObject, α::PyObject)
#   neles = domain.neles
#   nGauss = length(domain.elements[1].weights)
#   neqns_per_elem = length(getEqns(domain,1))
#   nstrains = size(E_all,2) #todo
 
  
#   @assert size(E_all)==(neles*nGauss, nstrains)
#   @assert size(DE_all)==(neles*nGauss, nstrains)
#   @assert size(σ0_all)==(neles*nGauss, nstrains)
  
#   # el_eqns_all, the equation numbers related to the Gaussian point, negative value means Drichlet boundary
#   el_eqns_all = zeros(Int32, neles*nGauss, neqns_per_elem)
#   # el_eqns_active_all = el_eqns_all > 0
#   el_eqns_active_all = zeros(Bool, neles*nGauss, neqns_per_elem)
#   # Loop over the elements in the elementGroup to construct el_eqns_active_all , el_eqns_all and w∂E∂u_all
#   for iele  = 1:neles
#     # Get the element nodes
#     el_eqns = getEqns(domain,iele)
 
#     # Assemble in the global array
#     el_eqns_active_all[(iele-1)*nGauss+1:iele*nGauss,:] = repeat((el_eqns .>= 1)', nGauss, 1)
#     el_eqns_all[(iele-1)*nGauss+1:iele*nGauss,:] = repeat(el_eqns', nGauss, 1)
#   end

#   # get stress at each Gaussian points
#   # @info "* ", E_all, DE_all, σ0_all
#   σ_all, α = nn(E_all, DE_all, σ0_all, α) # α is the internal variable
#   fints = squeeze(tf.matmul(w∂E∂u_all, tf.expand_dims(σ_all,2)))
#   Fint = cpp_fint(fints,constant(el_eqns_all, dtype=Int32),constant(domain.neqs, dtype=Int32))
#   return Fint, σ_all, α
# end





@doc """
    Numerically assemble internal force vector and stiffness matrix, compute local internal force f_int/ 
    stiffness matrix Slocal and then assemble to F_int/Ksparse
    - 'globdat': GlobalData
    - 'domain': Domain, finite element domain, for data structure
    - 'Δt':  Float64, current time step size
    Return F_int: Float64[neqs], internal force vector
    Return Ksparse: Float64[neqs, neqns], sparse stiffness matrix
"""->function assembleStiffAndForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)
    Fint = zeros(Float64, domain.neqs)
  # K = zeros(Float64, domain.neqs, domain.neqs)
    ii = Int64[]; jj = Int64[]; vv = Float64[]
    neles = domain.neles

  # Loop over the elements in the elementGroup
    for iele  = 1:neles
        element = domain.elements[iele]

    # Get the element nodes
        el_nodes = getNodes(element)

    # Get the element nodes equation numbers
        el_eqns = getEqns(domain, iele)
    
    # Get the element nodes dof numbers
        el_dofs = getDofs(domain, iele)

        el_state  = getState(domain, el_dofs)

        el_Dstate = getDstate(domain, el_dofs)

    # Get the element contribution by calling the specified action
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
    end
    Ksparse = sparse(ii, jj, vv, domain.neqs, domain.neqs)

    return Fint, Ksparse
end

# function assembleStiffAndForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)
#   Fint = zeros(Float64, domain.neqs)
#   K = zeros(Float64, domain.neqs, domain.neqs)
#   neles = domain.neles

#   # Loop over the elements in the elementGroup
#   for iele  = 1:neles
#     element = domain.elements[iele]

#     # Get the element nodes
#     el_nodes = getNodes(element)

#     # Get the element nodes
#     el_eqns = getEqns(domain,iele)

#     el_dofs = getDofs(domain,iele)

#     #@show "iele", iele, el_dofs 
    
#     #@show "domain.state", iele, domain.state 

#     el_state  = getState(domain,el_dofs)

#     el_Dstate = getDstate(domain,el_dofs)
#     # #@show "+++++", el_state, el_Dstate

#     # Get the element contribution by calling the specified action
#     #@info "ele id is ", iele
#     fint, stiff  = getStiffAndForce(element, el_state, el_Dstate, Δt)

#     # Assemble in the global array
#     el_eqns_active = el_eqns .>= 1
#     K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
#     Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]

#     # @info "Fint is ", Fint
#   end
#   return Fint, sparse(K)
# end

@doc """
    compute constant sparse mass matrix
    due to the time-dependent Dirichlet boundary condition
    mass matrix = M,    MID
                  MID'  MDD

    - 'globdat': GlobalData
    - 'domain': Domain, finite element domain, for data structure
    here M is Float64[neqns, neqns]
         MID is Float64[neqns, nd1]
         Mlumped is Float64[neqns]

    update M and MID and Mlumpe in globaldat
"""->function assembleMassMatrix!(globaldat::GlobalData, domain::Domain)
    Mlumped = zeros(Float64, domain.neqs)
    # M = zeros(Float64, domain.neqs, domain.neqs)
    iiM = Int64[]; jjM = Int64[]; vvM = Float64[]
    Mlumped = zeros(Float64, domain.neqs)
    neles = domain.neles

    nnodes, ndims = domain.nnodes, domain.ndims

    # MID = zeros(domain.neqs, sum(domain.EBC .== -2))
    iiMID = Int64[]; jjMID = Int64[]; vvMID = Float64[]

    # construct map from freedoms(first direction, second direction ...) to time-dependent Dirichlet freedoms
    dofs_to_EBCdofs = zeros(Int64, nnodes*ndims)
    if globaldat.EBC_func != nothing
        dof_id = 0
        for idof = 1:ndims
            for inode = 1:nnodes
                if (domain.EBC[inode, idof] == -2)
                    dof_id += 1
                    dofs_to_EBCdofs[inode + (idof - 1) * nnodes] = dof_id
                end
            end
        end
    end


    # Loop over the elements in the elementGroup
    for iele = 1:neles
        element = domain.elements[iele]

        # Get the element nodes
        el_nodes = getNodes(element)
    
        # Get the element nodes equation numbers
        el_eqns = getEqns(domain, iele)

        # Get the element nodes dof numbers
        el_dofs = getDofs(domain, iele)

        # Get the element contribution by calling the specified action
        lM, lumped = getMassMatrix(element)

        # Assemble in the global array
        
        el_eqns_active = (el_eqns .>= 1)

        # time-dependent Dirichlet boundary condition
        el_eqns_acc_active = (el_eqns .== -2)
        
        # M[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += lM[el_eqns_active, el_eqns_active]
        Idx = el_eqns[el_eqns_active]
        Mlocal = lM[el_eqns_active, el_eqns_active]
        for i = 1:length(Idx)
            for j = 1:length(Idx)
                push!(iiM, Idx[i])
                push!(jjM, Idx[j])
                push!(vvM, Mlocal[i,j])
            end
        end

        Mlumped[el_eqns[el_eqns_active]] += lumped[el_eqns_active]


        if globaldat.EBC_func != nothing
            Idx = el_eqns[el_eqns_active]
            Idy = dofs_to_EBCdofs[el_dofs[el_eqns_acc_active]]
            if length(Idy) > 0
            @show el_eqns_acc_active, el_dofs[el_eqns_acc_active], Idy
            end
            Mlocal = lM[el_eqns_active, el_eqns_acc_active]
            for i = 1:length(Idx)
                for j = 1:length(Idy)
                    push!(iiMID, Idx[i])
                    push!(jjMID, Idy[j])
                    push!(vvMID, Mlocal[i,j])
                end
            end
        end
        
    end

    globaldat.M = sparse(iiM, jjM, vvM, domain.neqs, domain.neqs)
    globaldat.Mlumped = Mlumped
    globaldat.MID = sparse(iiMID, jjMID, vvMID, domain.neqs, sum(domain.EBC .== -2))


end


# @doc """
#     compute constant mass matrix as sparse matrix
#     due to the time-dependent Dirichlet boundary condition
#     mass matrix = M,    MID
#                   MID'  MDD
#     save M and MID and lump(M)
# """->

# function assembleMassMatrix!(globaldat::GlobalData, domain::Domain)
#   Mlumped = zeros(Float64, domain.neqs)
#   M = zeros(Float64, domain.neqs, domain.neqs)
#   Mlumped = zeros(Float64, domain.neqs)
#   neles = domain.neles

#   MID = zeros(domain.neqs, sum(domain.EBC .== -2))
#   # Loop over the elements in the elementGroup
#   for iele = 1:neles
#       element = domain.elements[iele]

#       # Get the element nodes
#       el_nodes = getNodes(element)
  
#       # Get the element nodes
#       el_eqns = getEqns(domain,iele)

#       # Get the element contribution by calling the specified action
#       lM, lumped = getMassMatrix(element)

#       # Assemble in the global array
      
#       el_eqns_active = (el_eqns .>= 1)
#       el_eqns_acc_active = (el_eqns .== -2)
      
#       M[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += lM[el_eqns_active, el_eqns_active]
#       Mlumped[el_eqns[el_eqns_active]] += lumped[el_eqns_active]
#       MID[el_eqns[el_eqns_active], el_eqns[el_eqns_acc_active]] += lM[el_eqns_active, el_eqns_acc_active]
      
#   end

#   globaldat.M = sparse(M)
#   globaldat.Mlumped = sparse(Mlumped)
#   globaldat.MID = MID
# end