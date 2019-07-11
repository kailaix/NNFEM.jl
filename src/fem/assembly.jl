using SparseArrays
export assembleStiffAndForce,assembleInternalForce,assembleMassMatrix!,tfAssembleInternalForce
function assembleInternalForce(globdat::GlobalData, domain::Domain)
    Fint = zeros(Float64, domain.neqs)
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
      fint = getInternalForce(element, el_state, el_Dstate)
  
      # Assemble in the global array
      el_eqns_active = (el_eqns .>= 1)
      Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]
    end
  
    return Fint
end

function tfAssembleInternalForce(domain::Domain, nn::Function,
  E_all::PyObject, DE_all::PyObject, σ0_all::PyObject)
  neles = domain.neles
  nGauss = length(domain.elements[1].weights)
  neqns_per_elem = length(getEqns(domain,1))
  nstrains = 3

  @assert size(E_all)==(neles*nGauss, nstrains)
  @assert size(DE_all)==(neles*nGauss, nstrains)
  @assert size(σ0_all)==(neles*nGauss, nstrains)
  
  el_eqns_active_all = zeros(Bool, neles*nGauss, neqns_per_elem)
  el_eqns_all = zeros(Int32, neles*nGauss, neqns_per_elem)
  w∂E∂u_all = zeros(neles*nGauss, neqns_per_elem, nstrains)
  
  # Loop over the elements in the elementGroup to construct strain and geo-matrix E_all and w∂E∂u_all
  for iele  = 1:neles
    element = domain.elements[iele]

    # Get the element nodes
    el_nodes = getNodes(element)

    # Get the element nodes
    el_eqns = getEqns(domain,iele)

    el_dofs = getDofs(domain,iele)

    el_state  = getState(domain,el_dofs)

    # Get the element contribution by calling the specified action
    _, w∂E∂u = getStrain(element, el_state)
    
    # # Assemble in the global array
    el_eqns_active_all[(iele-1)*nGauss+1:iele*nGauss,:] = repeat((el_eqns .>= 1)', nGauss, 1)
    el_eqns_all[(iele-1)*nGauss+1:iele*nGauss,:] = repeat(el_eqns', nGauss, 1)
    
    w∂E∂u_all[(iele-1)*nGauss+1:iele*nGauss,:,:] = w∂E∂u
  end
  # @info "here"
  σ_all = nn(E_all, DE_all, σ0_all)
  # @info "here"
  el_eqns_active_all = constant(el_eqns_active_all, dtype=Bool)
  #while loop only aa
  el_eqns_all[el_eqns_all .<= 0] .= 1 
  el_eqns_all = constant(el_eqns_all, dtype=Int64)
  w∂E∂u_all = constant(w∂E∂u_all)
  # @info "here"

  # * while loop
  function cond0(i, tensor_array)
    i<=neles*nGauss+1
  end
  function body(i, tensor_array)
    x = constant(zeros(Float64, domain.neqs))
    fint = w∂E∂u_all[i - 1] * σ_all[i - 1]
    
    # op = tf.print("+",i,fint, summarize=-1)
    # fint = bind(fint, op)

    fint = fint * cast(Float64, el_eqns_active_all[i - 1]) # 8D

    # op = tf.print("-",i, el_eqns_all[i-1], summarize=-1)
    # fint = bind(fint, op)

    x = scatter_add(x, el_eqns_all[i-1], fint)
    # op = tf.print("-",i, x, summarize=-1)
    # x = bind(x, op)
    tensor_array = write(tensor_array, i, x)
    i+1,tensor_array
  end
  tensor_array = TensorArray(neles*nGauss+1)
  Fint = constant(zeros(Float64, domain.neqs))
  tensor_array = write(tensor_array, 1, Fint)
  i = constant(2, dtype=Int32)
  _, out = while_loop(cond0, body, [i, tensor_array]; parallel_iterations=10)
  out = stack(out)


  Fint = sum(out, dims=1)
  # op = tf.print("*",Fint, summarize=-1)
  # Fint = bind(Fint, op)
  return Fint, σ_all
end

function assembleStiffAndForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)
    # Fint = zeros(Float64, domain.neqs)
    
    # K = zeros(Float64, domain.neqs, domain.neqs)
    neles = domain.neles

    FII = Array{Array{Int64}}(undef, neles)
    FVV = Array{Array{Float64}}(undef, neles)
    II = Array{Array{Int64}}(undef, neles)
    JJ = Array{Array{Int64}}(undef, neles)
    VV = Array{Array{Float64}}(undef, neles)
    # Loop over the elements in the elementGroup
    Threads.@threads for iele  = 1:neles
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
      # #@show "+++++", el_state, el_Dstate
  
      # Get the element contribution by calling the specified action
      #@info "ele id is ", iele
      fint, stiff  = getStiffAndForce(element, el_state, el_Dstate, Δt)

      # Assemble in the global array
      el_eqns_active = el_eqns .>= 1
      # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]

      el_act = el_eqns[el_eqns_active]
      # el_act = reshape(el_eqns[el_eqns_active], length(el_eqns[el_eqns_active]), 1)
      II[iele] = (el_act*ones(Int64, 1, length(el_act)))[:]
      JJ[iele] = (el_act*ones(Int64, 1, length(el_act)))'[:]
      VV[iele] = stiff[el_eqns_active,el_eqns_active][:]
      FII[iele] = el_act
      FVV[iele] = fint[el_eqns_active]
      # Fint[el_act] += fint[el_eqns_active]
    end
    II = vcat(II...); JJ = vcat(JJ...); VV = vcat(VV...); FII=vcat(FII...); FVV = vcat(FVV...)
    K = sparse(II,JJ,VV,domain.neqs,domain.neqs)
    Fint = sparse(FII, ones(length(FII)), FVV, domain.neqs, 1)|>Array
    # Ksp = sparse(II,JJ,VV)
    # @show norm(K-Ksp)
    return Fint, K
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
#   end
#   return Fint, sparse(K)
# end


function assembleMassMatrix!(globaldat::GlobalData, domain::Domain)
    Mlumped = zeros(Float64, domain.neqs)
    M = zeros(Float64, domain.neqs, domain.neqs)
    Mlumped = zeros(Float64, domain.neqs)
    neles = domain.neles

    # Loop over the elements in the elementGroup
    for iele = 1:neles
        element = domain.elements[iele]

        # Get the element nodes
        el_nodes = getNodes(element)
    
        # Get the element nodes
        el_eqns = getEqns(domain,iele)

        # Get the element contribution by calling the specified action
        lM, lumped = getMassMatrix(element)

        # Assemble in the global array
        
        el_eqns_active = (el_eqns .>= 1)
        M[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += lM[el_eqns_active, el_eqns_active]
        Mlumped[el_eqns[el_eqns_active]] += lumped[el_eqns_active]

        
    end

    globaldat.M = sparse(M)
    globaldat.Mlumped = sparse(Mlumped)
  
end

