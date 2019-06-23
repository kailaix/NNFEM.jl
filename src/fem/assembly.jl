export assembleInternalForce,assembleStiffAndForce,assembleMassMatrix
function assembleInternalForce(globdat::GlobalData, domain::Domain)
    Fint = zeros(Int64, domain.neqs)
    neles = domain.neles
  
    # Loop over the elements in the elementGroup
    for iele  = 1:neles
      element = domain.elements[iele]
  
      # Get the element nodes
      el_nodes = getNodes(element)
  
      el_coords = getCoords(domain,el_nodes)
  
      # Get the element nodes
      el_eqns = getEqns(domain,iele)
  
      el_dofs = getDofs(domain,iele)
  
      el_state  = getState(domain,el_dofs)
  
      el_Dstate = getDstate(domain,el_dofs)
  
      # Get the element contribution by calling the specified action
      fint = getInternalForce(element, el_coords, el_state, el_Dstate)
  
      # Assemble in the global array
      el_eqns_active = (el_eqns .>= 0)
      Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]
    end
  
    return Fint
end

function assembleStiffAndForce(globdat::GlobalData, domain::Domain)
    Fint = zeros(Int64, domain.neqs)
    K = zeros(Int64, domain.neqs, domain.neqs)
    neles = domain.neles
  
    # Loop over the elements in the elementGroup
    for iele  = 1:neles
      element = domain.elements[iele]
  
      # Get the element nodes
      el_nodes = getNodes(element)
  
      el_coords = getCoords(domain,el_nodes)
  
      # Get the element nodes
      el_eqns = getEqns(domain,iele)
  
      el_dofs = getDofs(domain,iele)
  
      el_state  = getState(domain,el_dofs)
  
      el_Dstate = getDstate(domain,el_dofs)
  
      # Get the element contribution by calling the specified action
      fint, stiff  = getInternalForce(element, el_state, el_Dstate)

      # Assemble in the global array
      el_eqns_active = el_eqns .>= 0
      K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
      Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]
    end
    return Fint,K
end

function assembleMassMatrix(globdat::GlobalData, domain::Domain)
    Mlumped = zeros(Int64, domain.neqs)
    M = zeros(Int64, domain.neqs, domain.neqs)

    neles = domain.neles

    # Loop over the elements in the elementGroup
    for iele = 1:neles
        element = domain.elements[iele]

        # Get the element nodes
        el_nodes = getNodes(element)
    
        el_coords = getCoords(domain,el_nodes)
    
        # Get the element nodes
        el_eqns = getEqns(domain,iele)

        # Get the element contribution by calling the specified action
        lM, lMlumped = getMassMatrix(element, el_coords)


        # Assemble in the global array
        el_eqns_active = (el_eqns .>= 0)

        M[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += lM[el_eqns_active, el_eqns_active]

        Mlumped[el_eqns[el_eqns_active]] += lMlumped[el_eqns_active]
    end

    return M, Mlumped
  
end

