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


@doc """
    domain   : finite element domain, for data structure
    nn: Neural network constitutive law, with 
    E_all    : all strains for the current time-step, with size (neles*nGauss, nstrains)
    DE_all   : all strains for the last time-step, with size (neles*nGauss, nstrains)
    w∂E∂u_all: multiplication of the Gaussian weight and ∂E∂u^T for current time-step, 
               with size (neles*nGauss, ndofs_per_element, nstrains)
    σ0_all   : all stresses for the last time-step, with size (neles*nGauss, nstrains)

    compute local internal force f_int and then assemble to F_int
"""->
function tfAssembleInternalForce(domain::Domain, nn::Function, E_all::PyObject, DE_all::PyObject, w∂E∂u_all::PyObject, σ0_all::PyObject)
  neles = domain.neles
  nGauss = length(domain.elements[1].weights)
  neqns_per_elem = length(getEqns(domain,1))
  nstrains = size(E_all,2) #todo
 
  
  @assert size(E_all)==(neles*nGauss, nstrains)
  @assert size(DE_all)==(neles*nGauss, nstrains)
  @assert size(σ0_all)==(neles*nGauss, nstrains)
  
  # el_eqns_all, the equation numbers related to the Gaussian point, negative value means Drichlet boundary
  el_eqns_all = zeros(Int32, neles*nGauss, neqns_per_elem)
  # el_eqns_active_all = el_eqns_all > 0
  el_eqns_active_all = zeros(Bool, neles*nGauss, neqns_per_elem)
  
  # Loop over the elements in the elementGroup to construct el_eqns_active_all , el_eqns_all and w∂E∂u_all
  for iele  = 1:neles
    # Get the element nodes
    el_eqns = getEqns(domain,iele)
 
    # Assemble in the global array
    el_eqns_active_all[(iele-1)*nGauss+1:iele*nGauss,:] = repeat((el_eqns .>= 1)', nGauss, 1)
    el_eqns_all[(iele-1)*nGauss+1:iele*nGauss,:] = repeat(el_eqns', nGauss, 1)
  end

  # get stress at each Gaussian points
  # @info "* ", E_all, DE_all, σ0_all
  σ_all = nn(E_all, DE_all, σ0_all)

  # @info "* *** "

  # cast to tensorflow variable
  el_eqns_active_all = constant(el_eqns_active_all, dtype=Bool)
  # trick, set Dirichlet equation number to 1, when assemble, add 0 to equation 1.
  el_eqns_all[el_eqns_all .<= 0] .= 1 
  el_eqns_all = constant(el_eqns_all, dtype=Int64)
  # cast to tensorflow variable  

  # * while loop
  function cond0(i, tensor_array_Fint)
    i<=neles*nGauss+1
  end
  function body(i, tensor_array_Fint)
    x = constant(zeros(Float64, domain.neqs))
    # fint in the ith Gaussian point
    fint = w∂E∂u_all[i - 1] * σ_all[i - 1]


    # set fint entries to 0, when the dof is Dirichlet boundary
    fint = fint .* cast(Float64, el_eqns_active_all[i - 1]) # 8D
    # puth fint in the Fint at address el_eqns_all[i-1]
    x = scatter_add(x, el_eqns_all[i-1], fint)
    
    # op = tf.print("w∂E∂u_all", w∂E∂u_all[i - 1], summarize=-1)
    # fint = bind(fint, op)

    # op = tf.print("fint", x, summarize=-1)
    # x = bind(x, op)

    # op = tf.print(x, summarize=-1)
    # x = bind(x, op)
    
    # write x to tensor_array_Fint
    tensor_array_Fint = write(tensor_array_Fint, i, x)
    i+1,tensor_array_Fint
  end
  tensor_array_Fint = TensorArray(neles*nGauss+1)
  Fint = constant(zeros(Float64, domain.neqs))

  tensor_array_Fint = write(tensor_array_Fint, 1, Fint)
  i = constant(2, dtype=Int32)
  _, out = while_loop(cond0, body, [i, tensor_array_Fint]; parallel_iterations=1)
  out = stack(out)


  Fint = sum(out, dims=1)


  # op = tf.print("E_all", E_all, summarize=-1)
  # Fint = bind(Fint, op)
  # op = tf.print("σ", σ_all, summarize=-1)
  # Fint = bind(Fint, op)
  # op = tf.print("Fint", Fint, summarize=-1)
  # Fint = bind(Fint, op)
  
  # op = tf.print("*",Fint, summarize=-1)
  # Fint = bind(Fint, op)
  return Fint, σ_all
end

# function assembleStiffAndForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)
#     # Fint = zeros(Float64, domain.neqs)
    
#     # K = zeros(Float64, domain.neqs, domain.neqs)
#     neles = domain.neles

#     FII = Array{Array{Int64}}(undef, neles)
#     FVV = Array{Array{Float64}}(undef, neles)
#     II = Array{Array{Int64}}(undef, neles)
#     JJ = Array{Array{Int64}}(undef, neles)
#     VV = Array{Array{Float64}}(undef, neles)
#     # Loop over the elements in the elementGroup
#     # Threads.@threads 
#     for iele  = 1:neles
#       element = domain.elements[iele]
  
#       # Get the element nodes
#       el_nodes = getNodes(element)
  
#       # Get the element nodes
#       el_eqns = getEqns(domain,iele)
  
#       el_dofs = getDofs(domain,iele)

#       #@show "iele", iele, el_dofs 
      
#       #@show "domain.state", iele, domain.state 

#       el_state  = getState(domain,el_dofs)
  
#       el_Dstate = getDstate(domain,el_dofs)
#       # #@show "+++++", el_state, el_Dstate
  
#       # Get the element contribution by calling the specified action
#       #@info "ele id is ", iele
#       fint, stiff  = getStiffAndForce(element, el_state, el_Dstate, Δt)

#       # Assemble in the global array
#       el_eqns_active = el_eqns .>= 1
#       # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]

#       el_act = el_eqns[el_eqns_active]
#       # el_act = reshape(el_eqns[el_eqns_active], length(el_eqns[el_eqns_active]), 1)
#       II[iele] = (el_act*ones(Int64, 1, length(el_act)))[:]
#       JJ[iele] = (el_act*ones(Int64, 1, length(el_act)))'[:]
#       VV[iele] = stiff[el_eqns_active,el_eqns_active][:]
#       FII[iele] = el_act
#       FVV[iele] = fint[el_eqns_active]
#       # Fint[el_act] += fint[el_eqns_active]
#     end
#     II = vcat(II...); JJ = vcat(JJ...); VV = vcat(VV...); FII=vcat(FII...); FVV = vcat(FVV...)
#     K = sparse(II,JJ,VV,domain.neqs,domain.neqs)
#     Fint = sparse(FII, ones(length(FII)), FVV, domain.neqs, 1)|>Array
#     # Ksp = sparse(II,JJ,VV)
#     # @show norm(K-Ksp)
#     return Fint, K
# end

function assembleStiffAndForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)
  Fint = zeros(Float64, domain.neqs)
  K = zeros(Float64, domain.neqs, domain.neqs)
  neles = domain.neles

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
    # #@show "+++++", el_state, el_Dstate

    # Get the element contribution by calling the specified action
    #@info "ele id is ", iele
    fint, stiff  = getStiffAndForce(element, el_state, el_Dstate, Δt)

    # Assemble in the global array
    el_eqns_active = el_eqns .>= 1
    K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
    Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]

    # @info "Fint is ", Fint
  end
  return Fint, sparse(K)
end

@doc """
    compute constant mass matrix
    due to the time-dependent Dirichlet boundary condition
    mass matrix = M,    MID
                  MID'  MDD
    save M and MID and lump(M)
"""->
function assembleMassMatrix!(globaldat::GlobalData, domain::Domain)
    Mlumped = zeros(Float64, domain.neqs)
    M = zeros(Float64, domain.neqs, domain.neqs)
    Mlumped = zeros(Float64, domain.neqs)
    neles = domain.neles

    MID = zeros(domain.neqs, sum(domain.EBC .== -2))
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
        el_eqns_acc_active = (el_eqns .== -2)
        
        M[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += lM[el_eqns_active, el_eqns_active]
        Mlumped[el_eqns[el_eqns_active]] += lumped[el_eqns_active]
        MID[el_eqns[el_eqns_active], el_eqns[el_eqns_acc_active]] += lM[el_eqns_active, el_eqns_acc_active]
        
    end

    globaldat.M = sparse(M)
    globaldat.Mlumped = sparse(Mlumped)
    globaldat.MID = MID
end

