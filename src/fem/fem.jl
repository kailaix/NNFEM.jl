export Domain,GlobalData,updateStates!,updateDomainStateBoundary!,getExternalForce,
    setNeumannBoundary!, setGeometryPoints!


@doc """
GlobalData

Store data for finite element update, assume the problem has n freedoms

- `state`: Float64[n],  displacement array at the current time, only for active freedoms, 
   the ordering is based on the equation number, here ndims=2 is the dimension of the problem space
- `state`: Float64[n],  displacement array at the previous time
- `velo`: Float64[n],  velocity array at the current 
- `acce`: Float64[n],  acceleration array at the current 
- `time`: float, current time
- `M`: Float64[n,n] spares mass matrix
- `Mlumped`: Float64[n] lumped mass array
- `MID`: Float64[n, nd1] off-diagonal part of the mass matrix, between the active freedoms and the time-dependent Dirichlet freedoms, assume there are nd time-dependent Dirichlet freedoms
- `EBC_func`: function Float64:t-> Float64[n_d1] float array, time-dependent Dirichlet boundary condition (ordering is direction first then node number, u1, u3, ... v1, v4 ...)
- `FBC_func`: function Float64:t-> Float64[n_t1] float array, time-dependent load boundary condition (ordering is direction first then node number, u1, u3, ... v1, v4 ...)
"""->   
mutable struct GlobalData
    state::Array{Float64}    #u
    Dstate::Array{Float64}   #uk
    velo::Array{Float64}     #∂u
    acce::Array{Float64}     #∂∂u
    time::Float64
    M::Union{SparseMatrixCSC{Float64,Int64},Array{Float64}}
    Mlumped::Array{Float64}
    MID::Array{Float64}

    EBC_func::Union{Function,Nothing}  #time dependent Dirichlet boundary condition
    FBC_func::Union{Function,Nothing}  #time force load boundary condition
    
end


function GlobalData(state::Array{Float64},Dstate::Array{Float64},velo::Array{Float64},acce::Array{Float64}, neqs::Int64,
        EBC_func::Union{Function, Nothing}=nothing, FBC_func::Union{Function, Nothing}=nothing)
    time = 0.0
    M = Float64[]
    Mlumped = Float64[]
    MID = Float64[]
    GlobalData(state, Dstate, velo, acce, time, M, Mlumped, MID, EBC_func, FBC_func)
end


@doc """
Domain

Date structure for the computatational domain

- `nnodes`: Int64, number of nodes (each quadratical quad element has 9 nodes)
- `nodes`: Float64[nnodes, ndims], coordinate array of all nodes
- `neles`: number of elements 
- `elements`: Element[neles], element array, each element is a struct 
- `ndims`: Int64, dimension of the problem space 
- `state`: Float64[nnodes*ndims] current displacement of all nodal freedoms, Float64[1:nnodes] are for the first direction.
- `Dstate`: Float64[nnodes*ndims] previous displacement of all nodal freedoms, Float64[1:nnodes] are for the first direction.
- 'LM':  Int64[neles][ndims], LM(e,d) is the global equation number(active freedom number) of element e's d th freedom, 
         -1 means fixed (time-independent) Dirichlet
         -2 means time-dependent Dirichlet
- 'DOF': Int64[neles][ndims], DOF(e,d) is the global freedom number of element e's d th freedom
- 'ID':  Int64[nnodes, ndims], ID(n,d) is the equation number(active freedom number) of node n's dth freedom, 
         -1 means fixed (time-independent) Dirichlet
         -2 means time-dependent Dirichlet
- 'neqs':  Int64,  number of equations or active freedoms
- 'eq_to_dof':  Int64[neqs], map from to equation number(active freedom number) to the freedom number (Int64[1:nnodes] are for the first direction) 
- 'dof_to_eq':  Bool[nnodes*ndims], map from freedom number(Int64[1:nnodes] are for the first direction) to booleans (active freedoms(equation number) are true)
- 'EBC':  Int64[nnodes, ndims], EBC[n,d] is the displacement boundary condition of node n's dth freedom,
           -1 means fixed(time-independent) Dirichlet boundary nodes
           -2 means time-dependent Dirichlet boundary nodes
- 'g':  Float64[nnodes, ndims], values for fixed(time-independent) Dirichlet boundary conditions of node n's dth freedom,
- 'FBC': Int64[nnodes, ndims], FBC[n,d] is the force load boundary condition of node n's dth freedom,
           -1 means constant(time-independent) force load boundary nodes
           -2 means time-dependent force load boundary nodes
- 'fext':  Float64[neqs], constant(time-independent) force load boundary conditions for these freedoms
- 'time': Float64, current time
- 'npoints': Int64, number of points (each quadratical quad element has 4 points, npoints==nnodes, when porder==1)
- 'node_to_point': Int64[nnodes]:map from node number to point point, -1 means the node is not a geometry point

- 'ii_stiff': Int64[], first index of the sparse matrix representation of the stiffness matrix
- 'jj_stiff': Int64[], second index of the sparse matrix representation of the stiffness matrix
- 'vv_stiff_ele_indptr': Int64[], Int64[e] is the first index entry for the e's element of the sparse matrix representation of the stiffness matrix
- 'vv_stiff': Float64[], values of the sparse matrix representation of the stiffness matrix

- 'ii_dfint_dstress': Int64[], first index of the sparse matrix representation of the dfint_dstress matrix 
- 'jj_dfint_dstress': Int64[], second index of the sparse matrix representation of the dfint_dstress matrix
- 'vv_dfint_dstress_ele_indptr': Int64[], Int64[e] is the first index entry for the e's element of the sparse matrix representation of the dfint_dstress matrix
- 'vv_dfint_dstress': Float64[], values of the sparse matrix representation of the dfint_dstress matrix 

- 'ii_dstrain_dstate': Int64[], first index of the sparse matrix representation of the dstrain_dstate matrix
- 'jj_dstrain_dstate': Int64[], second index of the sparse matrix representation of the dstrain_dstate matrix
- 'vv_dstrain_dstate_ele_indptr': Int64[], Int64[e] is the first index entry for the e's element of the sparse matrix representation of the stiffness matrix
- 'vv_dstrain_dstate': Float64[], values of the sparse matrix representation of the dstrain_dstate matrix

- 'history': Dict{String, Array{Array{Float64}}}, dictionary between string and its time-histories quantity Float64[ntime][]
"""-> 
mutable struct Domain
    nnodes::Int64
    nodes::Array{Float64}
    neles::Int64
    elements::Array
    ndims::Int64
    state::Array{Float64}
    Dstate::Array{Float64}
    LM::Array{Array{Int64}}
    DOF::Array{Array{Int64}}
    ID::Array{Int64}
    neqs::Int64
    eq_to_dof::Array{Int64}
    dof_to_eq::Array{Bool}
    EBC::Array{Int64}  # Dirichlet boundary condition
    g::Array{Float64}  # Value for Dirichlet boundary condition
    FBC::Array{Int64}  # Nodal force boundary condition
    fext::Array{Float64}  # Value for Nodal force boundary condition
    time::Float64

    npoints::Int64     # number of mesh points(the same as nodes, when porder==1)
    node_to_point::Array{Int64} # map from node to point, -1 means the node is not a geometry point
    

    ii_stiff::Array{Int64} 
    jj_stiff::Array{Int64} 
    vv_stiff_ele_indptr::Array{Int64} 
    vv_stiff::Array{Float64} 

    ii_dfint_dstress::Array{Int64}  
    jj_dfint_dstress::Array{Int64}
    vv_dfint_dstress_ele_indptr::Array{Int64}    
    vv_dfint_dstress::Array{Float64}   

    ii_dstrain_dstate::Array{Int64}
    jj_dstrain_dstate::Array{Int64}
    vv_dstrain_dstate_ele_indptr::Array{Int64} 
    vv_dstrain_dstate::Array{Float64}   

    history::Dict{String, Array{Array{Float64}}}



end

function Base.:copy(g::Union{GlobalData, Domain}) 
    names = fieldnames(g)
    args = [copy(getproperty(g, n)) for n in names]
    GlobalData(args...)
end

@doc """
    Creating a finite element domain

    - `nodes`: Float64[nnodes, ndims], coordinate array of all nodes
    - `elements`: Element[neles], element array, each element is a struct 
    - `ndims`: Int64, dimension of the problem space 
    - 'EBC':  Int64[nnodes, ndims], EBC[n,d] is the displacement boundary condition of node n's dth freedom,
           -1 means fixed(time-independent) Dirichlet boundary nodes
           -2 means time-dependent Dirichlet boundary nodes
    - 'g':  Float64[nnodes, ndims], values for fixed(time-independent) Dirichlet boundary conditions of node n's dth freedom,
    - 'FBC': Int64[nnodes, ndims], FBC[n,d] is the force load boundary condition of node n's dth freedom,
           -1 means constant(time-independent) force load boundary nodes
           -2 means time-dependent force load boundary nodes
    - 'f':  Float64[nnodes, ndims], values for constant(time-independent) force load boundary conditions of node n's dth freedom,

    Return: Domain, the finite element domain
"""->
function Domain(nodes::Array{Float64}, elements::Array, ndims::Int64, EBC::Array{Int64}, g::Array{Float64}, FBC::Array{Int64}, f::Array{Float64})
    nnodes = size(nodes,1)
    neles = size(elements,1)
    state = zeros(nnodes * ndims)
    Dstate = zeros(nnodes * ndims)
    LM = Array{Int64}[]
    DOF = Array{Int64}[]
    ID = Int64[]
    neqs = 0
    eq_to_dof = Int64[]
    dof_to_eq = zeros(Bool, nnodes * ndims)
    fext = Float64[]

    npoints = -1
    node_to_point = Int64[]
    
    history = Dict("state"=>Array{Float64}[], "acc"=>Array{Float64}[], "fint"=>Array{Float64}[],
                "fext"=>Array{Float64}[], "strain"=>[], "stress"=>[], "time"=>Array{Float64}[])
    
    domain = Domain(nnodes, nodes, neles, elements, ndims, state, Dstate, 
    LM, DOF, ID, neqs, eq_to_dof, dof_to_eq, 
    EBC, g, FBC, fext, 0.0, npoints, node_to_point,
    Int64[], Int64[], Int64[], Float64[], 
    Int64[], Int64[], Int64[], Float64[], 
    Int64[], Int64[], Int64[], Float64[], history)

    #set fixed(time-independent) Dirichlet boundary conditions
    setDirichletBoundary!(domain, EBC, g)
    #set constant(time-independent) force load boundary conditions
    setNeumannBoundary!(domain, FBC, f)

    assembleSparseMatrixPattern!(domain)

    domain
end

@doc """
    In the constructor 
    Update the node_to_point map 

    - `self`: Domain, finit element domain
    - 'npoints': Int64, number of points (each quadratical quad element has 4 points, npoints==nnodes, when porder==1)
    - 'node_to_point': Int64[nnodes]:map from node number to point point, -1 means the node is not a geometry point

""" -> 
function setGeometryPoints!(self::Domain, npoints::Int64, node_to_point::Array{Int64})
    self.npoints = npoints
    self.node_to_point = node_to_point
end



@doc """
    Update current step strain, stress in the history map of the Domain
    strain and stress are both Float[ngp, nstrain], ngp is the number of Gaussian quadrature point
    nstrain=1 for 1D and nstrain = 3 for 2D 
    The strain is in Voigt notation
""" -> 
function commitHistory(domain::Domain)
    for e in domain.elements
        commitHistory(e)
    end
    
    # 1D, nstrain=1; 2D, nstrain=3
    eledim = domain.elements[1].eledim
    nstrain = div((eledim + 1)*eledim, 2)
    ngp = domain.neles * length(domain.elements[1].weights)
    if nstrain==1
        strain = zeros(ngp)
        stress = zeros(ngp)
        k = 1
        for e in domain.elements
            for igp in e.mat
                strain[k] = igp.ε0
                stress[k] = igp.σ0
                k += 1
            end
        end
    else
        strain = zeros(ngp, nstrain)
        stress = zeros(ngp, nstrain)
        k = 1
        for e in domain.elements
            for igp in e.mat
                strain[k,:] = igp.ε0
                stress[k,:] = igp.σ0
                k += 1
            end
        end
    end

    push!(domain.history["strain"], strain)
    push!(domain.history["stress"], stress)
end



@doc """
    In the constructor 
    Update the fixed (time-independent Dirichlet boundary) state entries 
    Build LM, and DOF array
    - 'self': Domain
    - 'EBC':  Int64[nnodes, ndims], EBC[n,d] is the displacement boundary condition of node n's dth freedom,
           -1 means fixed(time-independent) Dirichlet boundary nodes
           -2 means time-dependent Dirichlet boundary nodes
    - 'g':  Float64[nnodes, ndims], values for fixed(time-independent) Dirichlet boundary conditions of node n's dth freedom,
    ''
""" -> 
function setDirichletBoundary!(self::Domain, EBC::Array{Int64}, g::Array{Float64})

    # ID(n,d) is the global equation number of node n's dth freedom, 
    # -1 means fixed (time-independent) Dirichlet
    # -2 means time-dependent Dirichlet

    nnodes, ndims = self.nnodes, self.ndims
    neles, elements = self.neles, self.elements
    #ID = zeros(Int64, nnodes, ndims) .- 1

    ID = copy(EBC)

    eq_to_dof, dof_to_eq = Int64[], zeros(Bool, nnodes * ndims)
    neqs = 0
    for idof = 1:ndims
      for inode = 1:nnodes
          if (EBC[inode, idof] == 0)
              neqs += 1
              ID[inode, idof] = neqs
              push!(eq_to_dof,inode + (idof-1)*nnodes)
              dof_to_eq[(idof - 1)*nnodes + inode] = true
          elseif (EBC[inode, idof] == -1)
              #update state fixed (time-independent) Dirichlet boundary conditions
              self.state[inode + (idof-1)*nnodes] = g[inode, idof]
          end
      end
    end

    self.ID, self.neqs, self.eq_to_dof, self.dof_to_eq = ID, neqs, eq_to_dof, dof_to_eq


    # LM(e,d) is the global equation number of element e's d th freedom
    LM = Array{Array{Int64}}(undef, neles)
    for iele = 1:neles
      el_nodes = getNodes(elements[iele])
      ieqns = ID[el_nodes, :][:]
      LM[iele] = ieqns
    end
    self.LM = LM

    # DOF(e,d) is the global dof number of element e's d th freedom

    DOF = Array{Array{Int64}}(undef, neles)
    for iele = 1:neles
      el_nodes = getNodes(elements[iele])
      DOF[iele] = [el_nodes;[idof + nnodes for idof in el_nodes]]
    end
    self.DOF = DOF
    
end


@doc """
    In the constructor
    Update the external force vector fext
    - 'self': Domain
    - 'FBC': Int64[nnodes, ndims], FBC[n,d] is the force load boundary condition of node n's dth freedom,
           -1 means constant(time-independent) force load boundary nodes
           -2 means time-dependent force load boundary nodes
    - 'f':  Float64[nnodes, ndims], values for constant(time-independent) force load boundary conditions of node n's dth freedom,

""" -> 
function setNeumannBoundary!(self::Domain, FBC::Array{Int64}, f::Array{Float64})

    fext = zeros(Float64, self.neqs)
    # ID(n,d) is the global equation number of node n's dth freedom, -1 means no freedom

    nnodes, ndims, ID = self.nnodes, self.ndims, self.ID
    for idof = 1:ndims
      for inode = 1:nnodes
          if (FBC[inode, idof] == -1)
              @assert ID[inode, idof] > 0
              fext[ID[inode, idof]] += f[inode, idof]
          end
        end
    end
    self.fext = fext
end


@doc """
    At each time step
    Update the state and Dstate in Domain from GlobalData
    state and Dstate in GlobalData are only for active freedoms (equations)
    state and Dstate in Domain are only for all freedoms, they are used for constructing the internal force and/or stiffness matrix
    update the state and acc history of the Domain
    - 'self': Domain
    - 'globaldat': GlobalData
""" ->
function updateStates!(self::Domain, globaldat::GlobalData)
    
    self.state[self.eq_to_dof] = globaldat.state[:]
    
    self.time = globaldat.time
    push!(self.history["state"], copy(self.state))
    push!(self.history["acc"], copy(globaldat.acce))

    updateDomainStateBoundary!(self, globaldat)
    
    self.Dstate = self.state[:]
end


# @doc """
#     :param self: Domain
#     :param state : 1D array to convert
#     :param compress_or_expand  : "Compress" or "Expand" 

#     "Compress", the state has all freedoms on all nodes, remove these freedoms on EBC
#     "Expand",   the state has only active freedoms on active nodes (active means not prescribed), 
#                 set these freedoms on EBC to 0

#     :return:
# """ ->
# function convertState(self::Domain, state::Array{Float64}, compress_or_expand::String)
    
#     if compress_or_expand == "Expand"
#         new_state = zeros(Float64, self.nnodes*self.ndims)
#         new_state[self.eq_to_dof] = state[:]
#         return new_state

#     elseif compress_or_expand == "Compress"
#         return state[self.eq_to_dof]
    
#     else
#         error("convertStats error, compress_or_expand is ", compress_or_expand)
#     end

# end



@doc """
    Update time dependent boundary of state fext in Domain based on the time-dependent boundary functions in GlobalData.
    - 'self': Domain
    - 'globaldat': GlobalData
    """ ->
function updateDomainStateBoundary!(self::Domain, globaldat::GlobalData)
    if globaldat.EBC_func != nothing
        disp, acce = globaldat.EBC_func(globaldat.time) # user defined time-dependent boundary
        dof_id = 0

        #update state of all nodes
        for idof = 1:self.ndims
            for inode = 1:self.nnodes
                if (self.EBC[inode, idof] == -2)
                    dof_id += 1
                    self.state[inode + (idof-1)*self.nnodes] = disp[dof_id]
                end
            end
        end
    end

    if globaldat.FBC_func != nothing
        ID = self.ID
        nodal_force = globaldat.FBC_func(globaldat.time) # user defined time-dependent boundary
        # @info nodal_force
        dof_id = 0
        #update fext for active nodes (length of neqs)
        for idof = 1:self.ndims
            for inode = 1:self.nnodes
                if (self.FBC[inode, idof] == -2)
                    dof_id += 1
                    @assert ID[inode, idof] > 0
                    self.fext[ID[inode, idof]] = nodal_force[dof_id]
                end
            end
        end
    end
end


@doc """
    Compute external force vector, including external force load and time-dependent Dirichlet boundary conditions
    The function needs to be called after
    function updateDomainStateBoundary!(self::Domain, globaldat::GlobalData)
    which computes the external force vector from external force load

    - 'self': Domain
    - 'globaldat': GlobalData
    - 'fext': Float64[neqs], container for the external force vector

    return external force vector at the present step
""" ->
function getExternalForce!(self::Domain, globaldat::GlobalData, fext::Array{Float64})
    fext[:] = self.fext
    if globaldat.EBC_func != nothing
        MID = globaldat.MID
        _, acce = globaldat.EBC_func(globaldat.time)

        fext -= MID * acce
    end
    fext
end


@doc """
    Get the coordinates of several nodes (possibly in one element)
    - 'self': Domain
    - 'el_nodes': Int64[n], node array

    Return: Float64[n, ndims], the coordinates of these nodes
""" ->
function getCoords(self::Domain, el_nodes::Array{Int64})
    return self.nodes[el_nodes, :]
end

@doc """
    Get the global freedom numbers of the element
    - 'self': Domain
    - 'iele': Int64, element number

    Return: Int64[], the global freedom numbers of the element (ordering in local element ordering)

""" ->
function getDofs(self::Domain, iele::Int64)    
    return self.DOF[iele]
end

@doc """
    Get the equation numbers(active freedom numbers) of the element
    - 'self': Domain
    - 'iele': Int64, element number

    Return: Int64[], the equation numbers(active freedom numbers) of the element (ordering in local element ordering)

""" ->
function getEqns(self::Domain, iele::Int64)
    return self.LM[iele]
end


@doc """
    Get the displacements of several nodes (possibly in one element)
    - 'self': Domain
    - 'el_nodes': Int64[n], node array

    Return: Float64[n, ndims], the displacements of these nodes
""" ->
function getState(self::Domain, el_dofs::Array{Int64})
    return self.state[el_dofs]
end

@doc """
    Get the displacements of several nodes (possibly in one element) at the previous time step
    - 'self': Domain
    - 'el_nodes': Int64[n], node array

    Return: Float64[n, ndims], the displacements of these nodes at the previous time step
""" ->

function getDstate(self::Domain, el_dofs::Array{Int64})
    return self.Dstate[el_dofs]
end

@doc """
    Compute constant stiff, dfint_dstress, dstrain_dstate matrix patterns
    - 'self': Domain
"""->
function assembleSparseMatrixPattern!(self::Domain)
    
    neles = self.neles
    eledim = self.elements[1].eledim
    nstrain = div((eledim + 1)*eledim, 2)
    ngps_per_elem = length(self.elements[1].weights)
    neqs = self.neqs


    ii_stiff = Int64[]; jj_stiff = Int64[]; vv_stiff_ele_indptr = ones(Int64, neles+1);
    ii_dfint_dstress = Int64[]; jj_dfint_dstress = Int64[]; vv_dfint_dstress_ele_indptr = ones(Int64, neles+1);
    ii_dstrain_dstate = Int64[]; jj_dstrain_dstate = Int64[]; vv_dstrain_dstate_ele_indptr = ones(Int64, neles+1);


    neles = self.neles
  
    # Loop over the elements in the elementGroup
    for iele  = 1:neles
      element = self.elements[iele]

      el_eqns = getEqns(self,iele)
  
      el_dofs = getDofs(self,iele)
  
      el_state  = getState(self, el_dofs)
  
      gp_ids = (iele-1)*ngps_per_elem+1 : iele*ngps_per_elem
      
   
      # Assemble in the global array
      el_eqns_active = el_eqns .>= 1
      el_eqns_active_idx = el_eqns[el_eqns_active]
      # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]

      el_eqns_active_idx = el_eqns[el_eqns_active]

      for j = 1:length(el_eqns_active_idx)
        for i = 1:length(el_eqns_active_idx)
          push!(ii_stiff, el_eqns_active_idx[i])
          push!(jj_stiff, el_eqns_active_idx[j])
          #push!(vv_stiff, stiff_active[i,j])
        end
      end
      vv_stiff_ele_indptr[iele+1] = vv_stiff_ele_indptr[iele] + length(el_eqns_active_idx)*length(el_eqns_active_idx)

      for j = 1:ngps_per_elem*nstrain
        for i = 1:length(el_eqns_active_idx) 
          push!(ii_dfint_dstress, el_eqns_active_idx[i])
          push!(jj_dfint_dstress, (iele-1)*ngps_per_elem*nstrain+j)
          #push!(vv_dfint_dstress, dfint_dstress_active[i,j])
        end
      end
      vv_dfint_dstress_ele_indptr[iele+1] = vv_dfint_dstress_ele_indptr[iele] + ngps_per_elem*nstrain*length(el_eqns_active_idx)

      for j = 1:length(el_eqns_active_idx)
        for i = 1:ngps_per_elem*nstrain
        
          push!(ii_dstrain_dstate, (iele-1)*ngps_per_elem*nstrain+i)
          push!(jj_dstrain_dstate, el_eqns_active_idx[j])
          #push!(vv_dstrain_dstate, dstrain_dstate_active[i,j])
        end
      end
      vv_dstrain_dstate_ele_indptr[iele+1] = vv_dstrain_dstate_ele_indptr[iele] + ngps_per_elem*nstrain*length(el_eqns_active_idx)


    end

    self.ii_stiff = ii_stiff; self.jj_stiff = jj_stiff; 
    self.vv_stiff_ele_indptr = vv_stiff_ele_indptr; self.vv_stiff = similar(ii_stiff)

    self.ii_dfint_dstress = ii_dfint_dstress; self.jj_dfint_dstress = jj_dfint_dstress; 
    self.vv_dfint_dstress_ele_indptr = vv_dfint_dstress_ele_indptr; self.vv_dfint_dstress= similar(ii_dfint_dstress)
    
    self.ii_dstrain_dstate = ii_dstrain_dstate; self.jj_dstrain_dstate = jj_dstrain_dstate; 
    self.vv_dstrain_dstate_ele_indptr = vv_dstrain_dstate_ele_indptr; self.vv_dstrain_dstate = similar(ii_dstrain_dstate)

  end