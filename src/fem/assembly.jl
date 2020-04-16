export assembleStiffAndForce,assembleMassMatrix!,assembleInternalForce, getEqns

@doc raw"""
    assembleInternalForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)

Computes the internal force vector $F_\mathrm{int}$ of length `neqs`
- `globdat`: GlobalData
- `domain`: Domain, finite element domain, for data structure
- `Δt`:  Float64, current time step size

Only the information in `domain` is used for computing internal force. 
Therefore, the boundary conditions in `domain` must be set appropriately.
"""
function assembleInternalForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)
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


@doc raw"""
    assembleInternalForce(domain::Domain, nn::Function, E_all::PyObject, DE_all::PyObject, w∂E∂u_all::PyObject, σ0_all::PyObject)

Computes local internal force f_int and then assemble to F_int, which generates inverse problem automatically.

- `domain`: finite element domain
- `nn`: constitutive relation for expressing `stress = f(strain)`, assuming `stress` and `strain` are defined on Gauss points (`(neles*nGauss) × nstrains`).
- `E_all`: strain data of size `(neles*nGauss) × nstrains` at the **current** time step.
- `DE_all`: strain data of size `(neles*nGauss) × nstrains` at the **last** time step.
- `w∂E∂u_all`: sensitivity matrix of size `(neles*nGauss) x ndofs_per_element x nstrains`; `neles*nGauss` is the number of Gaussian quadrature points, 
  `ndofs_per_element` is the number of freedoms per element, and `nstrain` is the number of strain components.
  The sensitivity matrix already considers the quadrature weights. 

```math
s_{g,j,i}^e = w_g^e\frac{\partial \epsilon_g^e}{\partial u_j^e}
```
where $e$ is the element index, $g$ is the Gaussian index. 
        
- `σ0_all`: stress data of size `neles*nGauss×nstrains` at the **last** time step. 

Return: 

- $F_{\mathrm{int}}$:  internal force vector of length `neqns`
- $\sigma_{\mathrm{all}}$: predicted stress at **current** step, a matrix of size `(neles*nGauss) × nstrains`
"""
function assembleInternalForce(domain::Domain, nn::Function, 
        E_all::Union{Array{Float64, 2}, PyObject}, DE_all::Union{Array{Float64, 2}, PyObject}, 
        w∂E∂u_all::Union{Array{Float64, 3}, PyObject}, σ0_all::Union{Array{Float64, 2}, PyObject})

    E_all, DE_all, w∂E∂u_all, σ0_all = convert_to_tensor([E_all, DE_all, w∂E∂u_all, σ0_all], [Float64, Float64, Float64, Float64])
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
    cpp_fint = load_op_and_grad("$(@__DIR__)/../../deps/CustomOp/FintComp/build/libFintComp", "fint_comp")
    Fint = cpp_fint(fints, constant(el_eqns_all, dtype = Int32), constant(domain.neqs, dtype = Int32))

    return Fint, σ_all
end


@doc raw"""
    assembleStiffAndForce(globdat::GlobalData, domain::Domain, Δt::Float64 = 0.0)

Computes the internal force and stiffness matrix. 

- `globdat`: GlobalData
- `domain`: Domain, finite element domain, for data structure
- `Δt`:  Float64, current time step size

Returns a length `neqs` vector $F_{\mathrm{int}}$ and `neqs×neqs` sparse stiffness matrix. 
"""
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


@doc raw"""
    assembleMassMatrix!(globaldat::GlobalData, domain::Domain)

Computes the constant sparse mass matrix $M_{\mathrm{mass}}$, the lumped mass matrix $M_{\mathrm{lump}}$
due to time-dependent Dirichlet boundary conditions, and store them in `globaldat`. 

```math
M_{\mathrm{mass}}\begin{bmatrix}
M & M_{ID}\\ 
M_{ID}^T & M_{DD} 
\end{bmatrix}
```

Here M is a `neqns×neqns` matrix, and $M_{ID}$ is a `neqns×nd` matrix. $M_{\mathrm{lump}}$ assumes that the local mass matrix is a diagonal matrix. 

- `globdat`: `GlobalData`
- `domain`: `Domain`, finite element domain, for data structure

![](./assets/massmatrix.png)
"""
function assembleMassMatrix!(globaldat::GlobalData, domain::Domain)
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
