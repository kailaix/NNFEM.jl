export Domain,GlobalData,updateStates!,updateTimeDependentEssentialBoundaryCondition!,
    setConstantNodalForces!, setGeometryPoints!, setConstantDirichletBoundary!, getExternalForce,
    commitHistory, getBodyForce, getEdgeForce, getNGauss, getDofs, getDStrain, getCoords, getState,
    getGaussPoints, getElems


@doc raw"""
    GlobalData

Store data for finite element updates. Assume the problem has n freedoms,

- `state`: a vector of length $n$. Displacement array at the **current** time, only for **active** freedoms.
   The ordering is based on the equation number.
- `Dstate`: a vector of length $n$. Displacement array at the **previous** time.
- `velo`: a vector of length $n$. Velocity array at the **current** time.
- `acce`: a vector of length $n$. Acceleration array at the **current** time.
- `time`: float, current time.
- `M`: a matrix of size $n\times n$ spares mass matrix
- `Mlumped`: a vector of length $n$ lumped mass array
- `MID`: Float64[n, nd1] off-diagonal part of the mass matrix, between the active freedoms and the time-dependent Dirichlet freedoms, assume there are nd time-dependent Dirichlet freedoms
- `EBC_func`: displacement $d$, velocity $v$, and acceleration $a$ from time-dependent Dirichlet boundary conditions 

$$d, v, a = \text{EBC\_func}(\text{time})$$

The length of each output is the same as number of "-2" in `EBC` array. The ordering is direction major, i.e., $u_1, u_3, \ldots, v_1, v_3, \ldots$ 

- `FBC_func`: time-dependent load boundary condition. 

$$f = \text{FBC\_func}(\text{time})$$

Here $f$ is a vector. Its length is the same as number of "-2" in `FBC` array. The ordering is direction major, i.e., $u_1, u_3, \ldots, v_1, v_3, \ldots$ 

- `Body_func`: time-dependent/independent body force function. 

$$f = \text{Body\_func}(x_{\text{array}}, y_{\text{array}}, \text{time})$$

Here $f$ is a vector or a matrix (second dimension is 2) depending on the dimension of state variables. 
The output is a $N\times n_{\text{dim}}$ matrix, where $N$ is the length of $x_{\text{array}}$ or $y_{\text{array}}$, and $n_{\text{dim}}$ is the dimension of the problem (1 or 2).

- `Edge_func`: time-dependent/independent traction load. 

$$f = \text{Edge\_func}(x_{\text{array}}, y_{\text{array}}, \text{time}, \text{id})$$

Here $f$ is a vector. Its length is the same as the length of $x_{\text{array}}$ or $y_{\text{array}}$.

"""
mutable struct GlobalData
    state::Array{Float64}    #u
    Dstate::Array{Float64}   #uk
    velo::Array{Float64}     #∂u
    acce::Array{Float64}     #∂∂u
    time::Float64
    M::Union{SparseMatrixCSC{Float64,Int64},Array{Float64}}
    Mlumped::Array{Float64}
    MID::Array{Float64}

    EBC_func::Union{Function,Nothing}  #time dependent Dirichlet boundary condition f(t)
    FBC_func::Union{Function,Nothing}  #time dependent nodal force  boundary condition f(t)
    Body_func::Union{Function,Nothing} #body force function f(x_array, y_array, t)
    Edge_func::Union{Function,Nothing} #edge traction function f(x_array, y_array, t, id)
    
end

function Base.show(io::IO, z::GlobalData)  
    yes = "✔️"
    no = "✘"
print(io, """GlobalData with $(length(z.state)) active DOFs, time=$(z.time)
Mass matrix ... $(length(z.M)==0 ? no : yes)
EBC_func    ... $(isnothing(z.EBC_func) ? no : yes)
FBC_func    ... $(isnothing(z.FBC_func) ? no : yes)
Body_func   ... $(isnothing(z.Body_func) ? no : yes)
Edge_func   ... $(isnothing(z.Edge_func) ? no : yes)
""")
end



@doc raw"""
    GlobalData(state::Union{Array{Float64,1},Missing},Dstate::Union{Array{Float64,1},Missing},
            velo::Union{Array{Float64,1},Missing},acce::Union{Array{Float64,1},Missing}, 
            neqs::Int64,
            EBC_func::Union{Function, Nothing}=nothing, FBC_func::Union{Function, Nothing}=nothing,
            Body_func::Union{Function,Nothing}=nothing, Edge_func::Union{Function,Nothing}=nothing)

The size of `state`, `Dstate`, `velo`, `acce` must be `neqs`, i.e., the active DOFs.
If they are missing, they are treated as zeros. 
"""
function GlobalData(state::Union{Array{Float64,1},Missing},Dstate::Union{Array{Float64,1},Missing},
        velo::Union{Array{Float64,1},Missing},acce::Union{Array{Float64,1},Missing}, 
        neqs::Int64,
        EBC_func::Union{Function, Nothing}=nothing, FBC_func::Union{Function, Nothing}=nothing,
        Body_func::Union{Function,Nothing}=nothing, Edge_func::Union{Function,Nothing}=nothing)
    time = 0.0
    M = Float64[]
    Mlumped = Float64[]
    MID = Float64[]
    state = coalesce(state, zeros(neqs))
    Dstate = coalesce(state, zeros(neqs))
    velo = coalesce(state, zeros(neqs))
    acce = coalesce(state, zeros(neqs))
    GlobalData(state, Dstate, velo, acce, time, M, Mlumped, MID, EBC_func, FBC_func, Body_func, Edge_func)
end




@doc raw"""
    Domain

Date structure for the computatational domain.

- `nnodes`: Int64, number of nodes (each quadratical quad element has 9 nodes)
- `nodes`: Float64[nnodes, ndims], coordinate array of all nodes
- `neles`: number of elements 
- `elements`: a list of `neles` element arrays, each element is a struct 
- `ndims`: Int64, dimension of the problem space 
- `state`: a matrix of size `nnodes×ndims`. **Current** displacement of all nodal freedoms, `state[1:nnodes]` are for the first direction.
- `Dstate`: `nnodes×ndims`. **Previous** displacement of all nodal freedoms, `Dstate[1:nnodes]` are for the first direction.
- `LM`:  `Int64[neles][ndims]`, LM(e,d) is the global equation number (active freedom number) of element e's d th freedom, 
         
         ∘ -1 means fixed (time-independent) Dirichlet

         ∘ -2 means time-dependent Dirichlet

         ∘ >0 means the global equation number

- `DOF`: a matrix of size `neles×ndims`, DOF(e,d) is the global freedom (including both active and inactive DOFs) number of element e's d th freedom.
- `ID`:  a matrix of size `nnodes×ndims`. `ID(n,d)` is the equation number (active freedom number) of node n's $d$-th freedom, 
         
         ∘ -1 means fixed (time-independent) Dirichlet

         ∘ -2 means time-dependent Dirichlet

         ∘ >0 means the global equation number

- `neqs`:  Int64,  number of equations, a.k.a., active freedoms
- `eq_to_dof`:  an integer vector of length `neqs`, map from to equation number (active freedom number) to the freedom number (Int64[1:nnodes] are for the first direction) 
- `dof_to_eq`:  a bolean array of size `nnodes×ndims`, map from freedom number(Int64[1:nnodes] are for the first direction) to booleans (active freedoms(equation number) are true)
- `EBC`:  Int64[nnodes, ndims], EBC[n,d] is the displacement boundary condition of node n's dth freedom,
           -1 means fixed(time-independent) Dirichlet boundary nodes
           -2 means time-dependent Dirichlet boundary nodes
- `g`:  Float64[nnodes, ndims], values for fixed(time-independent) Dirichlet boundary conditions of node n's dth freedom,
- `FBC`: Int64[nnodes, ndims], `FBC[n,d]`` is the force load boundary condition of node n's dth freedom,
           -1 means constant(time-independent) force load boundary nodes
           -2 means time-dependent force load boundary nodes
- `fext`:  Float64[neqs], constant (time-independent) nodal forces on these freedoms
- `Edge_Traction_Data`: `n × 3` integer matrix for natural boundary conditions.

      1. `Edge_Traction_Data[i,1]` is the element id,

      2. `Edge_Traction_Data[i,2]` is the local edge id in the element, where the force is exterted (should be on the boundary, but not required)

      3. `Edge_Traction_Data[i,3]` is the force id, which should be consistent with the last component of the `Edge_func` in the `Globdat`

- `time`: Float64, current time
- `npoints`: Int64, number of points (each quadratical quad element has 4 points, npoints==nnodes, when porder==1)
- `node_to_point`: Int64[nnodes]:map from node number to point point, -1 means the node is not a geometry point
"""
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
    edge_traction_data::Array{Int64} # traction force location, [element id, local edge id, force id]
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

function Base.show(io::IO, z::Domain) 
    yes = "✔️"
    no = "✘"
    print(io, """Domain with $(length(z.elements)) elements, $(z.nnodes) nodes and $(z.neqs) active equations
    edge_traction_data ... $(size(z.edge_traction_data, 1)==0 ? no : yes)
    """)
end


function Base.:copy(g::GlobalData) 
    names = propertynames(g)
    args = [copy(getproperty(g, n)) for n in names]
    GlobalData(args...)
end

function Base.:copy(g::Domain) 
    names = propertynames(g)
    args = [copy(getproperty(g, n)) for n in names]
    Domain(args...)
end

@doc raw"""
    Domain(nodes::Array{Float64}, elements::Array, ndims::Int64, EBC::Array{Int64}, g::Array{Float64}, FBC::Array{Int64}, f::Array{Float64})

Creating a finite element domain.

- `nodes`: coordinate array of all nodes, a `nnodes × 2` matrix
- `elements`: element array. Each element is a material struct, e.g., [`PlaneStrain`](@ref). 
- `ndims`: dimension of the problem space. For 2D problems, ndims = 2. 
- `EBC`:  `nnodes × ndims` integer matrix for essential boundary conditions
    `EBC[n,d]`` is the displacement boundary condition of node `n`'s $d$-th freedom,
    
    ∘ -1: fixed (time-independent) Dirichlet boundary nodes

    ∘ -2: time-dependent Dirichlet boundary nodes

- `g`:  `nnodes × ndims` double matrix, values for fixed (time-independent) Dirichlet boundary conditions of node `n`'s $d$-th freedom,
- `FBC`: `nnodes × ndims` integer matrix for nodal force boundary conditions.
    FBC[n,d] is the force load boundary condition of node n's dth freedom,

    ∘ -1 means constant(time-independent) force load boundary nodes

    ∘ -2 means time-dependent force load boundary nodes

- `f`:  `nnodes × ndims` double matrix, values for constant (time-independent) force load boundary conditions of node n's $d$-th freedom,

- `Edge_Traction_Data`: `n × 3` integer matrix for natural boundary conditions.
    Edge_Traction_Data[i,1] is the element id,
    Edge_Traction_Data[i,2] is the local edge id in the element, where the force is exterted (should be on the boundary, but not required)
    Edge_Traction_Data[i,3] is the force id, which should be consistent with the last component of the Edge_func in the Globdat

For time-dependent boundary conditions (`EBC` or `FBC` entries are -2), the corresponding `f` or `g` entries are not used.
"""
function Domain(nodes::Array{Float64}, elements::Array, ndims::Int64,
    EBC::Array{Int64}, g::Array{Float64}, FBC::Array{Int64}, 
    f::Array{Float64}, edge_traction_data::Array{Int64,2}=zeros(Int64,0,3))
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
    EBC, g, FBC, fext, edge_traction_data, 0.0, npoints, node_to_point,
    Int64[], Int64[], Int64[], Float64[], 
    Int64[], Int64[], Int64[], Float64[], 
    Int64[], Int64[], Int64[], Float64[], history)

    #set fixed(time-independent) Dirichlet boundary conditions
    setConstantDirichletBoundary!(domain, EBC, g)
    #set constant(time-independent) force load boundary conditions
    setConstantNodalForces!(domain, FBC, f)
    assembleSparseMatrixPattern!(domain)

    domain
end




@doc """
    commitHistory(domain::Domain)

Update current step strain and stress in the history map of the `domain`. 
This is essential for visualization and time dependent constitutive relations. 
""" 
function commitHistory(domain::Domain)
    for e in domain.elements
        commitHistory(e)
    end
    
    if options.save_history>=1
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
end



@doc """
    setConstantDirichletBoundary!(self::Domain, EBC::Array{Int64}, g::Array{Float64})

Bookkeepings for time-independent Dirichlet boundary conditions. Only called once in the constructor of `domain`. 
It updates the fixed (time-independent Dirichlet boundary) state entries and builds both LM and DOF arrays.

- `self`: Domain
- `EBC`:  Int64[nnodes, ndims], EBC[n,d] is the displacement boundary condition of node n's dth freedom,
        
  ∘ -1 means fixed(time-independent) Dirichlet boundary nodes

  ∘ -2 means time-dependent Dirichlet boundary nodes

- `g`:  Float64[nnodes, ndims], values for fixed (time-independent) Dirichlet boundary conditions of node n's dth freedom,
""" -> 
function setConstantDirichletBoundary!(domain::Domain, EBC::Array{Int64}, g::Array{Float64})

    # ID(n,d) is the global equation number of node n's dth freedom, 
    # -1 means fixed (time-independent) Dirichlet
    # -2 means time-dependent Dirichlet

    nnodes, ndims = domain.nnodes, domain.ndims
    neles, elements = domain.neles, domain.elements
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
              domain.state[inode + (idof-1)*nnodes] = g[inode, idof]
          end
      end
    end

    domain.ID, domain.neqs, domain.eq_to_dof, domain.dof_to_eq = ID, neqs, eq_to_dof, dof_to_eq


    # LM(e,d) is the global equation number of element e's d th freedom
    LM = Array{Array{Int64}}(undef, neles)
    for iele = 1:neles
      el_nodes = getNodes(elements[iele])
      ieqns = ID[el_nodes, :][:]
      LM[iele] = ieqns
    end
    domain.LM = LM

    # DOF(e,d) is the global dof number of element e's d th freedom

    DOF = Array{Array{Int64}}(undef, neles)
    for iele = 1:neles
      el_nodes = getNodes(elements[iele])
      if domain.ndims==1
        DOF[iele] = el_nodes
      elseif domain.ndims==2
        DOF[iele] = [el_nodes;[idof + nnodes for idof in el_nodes]]
      else
        error("NOT IMPLEMENTED YET")
      end
      
    end
    domain.DOF = DOF
    
end


@doc """

Bookkeepings for time-independent Nodal force boundary conditions. Only called once in the constructor of `domain`. 
It updates the fixed (time-independent Nodal forces) state entries and builds both LM and DOF arrays.

- `self`: Domain
- `FBC`:  Int64[nnodes, ndims], FBC[n,d] is the displacement boundary condition of node n's dth freedom,
        
    ∘ -1 means fixed (time-independent) Nodal force freedoms

    ∘ -2 means time-dependent Nodal force freedoms

- `f`:  Float64[nnodes, ndims], values for fixed (time-independent) Neumann boundary conditions of node n's dth freedom,

#The name is misleading
""" 
function setConstantNodalForces!(self::Domain, FBC::Array{Int64}, f::Array{Float64})

    fext = zeros(Float64, self.neqs)
    # ID(n,d) is the global equation number of node n's dth freedom, -1 means no freedom

    nnodes, ndims, ID = self.nnodes, self.ndims, self.ID
    for idof = 1:ndims
      for inode = 1:nnodes
          if (FBC[inode, idof] == -1)
              if  ID[inode, idof] <= 0
                error("Node $inode is both a Dirichlet node and a force node.")
              end
              fext[ID[inode, idof]] += f[inode, idof]
          end
        end
    end
    self.fext = fext
end


@doc """
    updateStates!(domain::Domain, globaldat::GlobalData)
    update time-dependent Dirichlet boundary condition to globaldat.time

Update `state` and `Dstate` in `domain`. This includes 

- Copy state variable values for active DOFs from `globaldat`

- Set time-dependent essential boundary conditions using `globaldat.EBC`

The time-independent boundary conditions are inherented from last time step.
""" 
function updateStates!(domain::Domain, globaldat::GlobalData)
    
    domain.state[domain.eq_to_dof] = globaldat.state[:]
    
    domain.time = globaldat.time
    if options.save_history>=1
        push!(domain.history["state"], copy(domain.state))
        push!(domain.history["acc"], copy(globaldat.acce))
    end

    updateTimeDependentEssentialBoundaryCondition!(domain, globaldat)
    
    domain.Dstate[:] = domain.state
end


@doc """
    updateTimeDependentEssentialBoundaryCondition!(domain::Domain, globaldat::GlobalData)
    
If there exists time-dependent Dirichlet boundary conditions, `updateTimeDependentEssentialBoundaryCondition!` must be called to update 
the boundaries in `domain`. This function is called by [`updateStates!`](@ref)

This function updates `state` data in `domain`.
"""
function updateTimeDependentEssentialBoundaryCondition!(domain::Domain, globaldat::GlobalData)
    if globaldat.EBC_func ≠ nothing
        disp, _, _ = globaldat.EBC_func(globaldat.time) # user defined time-dependent boundary
        dof_id = 0

        #update state of all nodes
        for idof = 1:domain.ndims
            for inode = 1:domain.nnodes
                if (domain.EBC[inode, idof] == -2)
                    dof_id += 1
                    domain.state[inode + (idof-1)*domain.nnodes] = disp[dof_id]
                end
            end
        end
    end

end


@doc """
    getExternalForce(self::Domain, globaldat::GlobalData, fext::Union{Missing,Array{Float64}}=missing)

Computes external force vector at `globaldat.time`, 
This includes all the body force, external load, and internal force caused by acceleration.
"""
function getExternalForce(domain::Domain, globaldat::GlobalData)
    #Update time-independent nodal force
    fext = copy(domain.fext)

    #Update time-dependent nodal force
    if !isnothing(globaldat.FBC_func)
        ID = domain.ID
        nodal_force = globaldat.FBC_func(globaldat.time) # user defined time-dependent boundary
        # @info nodal_force
        dof_id = 0
        #update fext for active nodes (length of neqs)
        for idof = 1:domain.ndims
            for inode = 1:domain.nnodes
                if (domain.FBC[inode, idof] == -2)
                    dof_id += 1
                    @assert ID[inode, idof] > 0
                    fext[ID[inode, idof]] += nodal_force[dof_id]
                end
            end
        end
    end


    #Update the acceleration effect from the time-dependent Dirichlet boundary condition
    if !isnothing(globaldat.EBC_func)        
        MID = globaldat.MID
        _, _, acce = globaldat.EBC_func(globaldat.time)
        fext -= MID * acce
    end

    #Update time-dependent and time-independent body force
    fbody = getBodyForce(domain, globaldat, globaldat.time)

    #Update time-dependent and time-independent edge traction force
    fedge = getEdgeForce(domain, globaldat, globaldat.time)
    
    fext + fbody + fedge
end

@doc raw"""
    getBodyForce(domain::Domain, globdat::GlobalData, time::Float64)

Computes the body force vector $F_\mathrm{body}$ of length `neqs`
- `globdat`: GlobalData
- `domain`: Domain, finite element domain, for data structure
- `Δt`:  Float64, current time step size
"""
function getBodyForce(domain::Domain, globdat::GlobalData, time::Float64)
    
    Fbody = zeros(Float64, domain.neqs)
    neles = domain.neles
    

    if isnothing(globdat.Body_func)
        return Fbody
    end

    # Loop over the elements in the elementGroup
    for iele  = 1:neles
        element = domain.elements[iele]
  
        gauss_pts = getGaussPoints(element)
        fvalue = globdat.Body_func(gauss_pts[:,1], gauss_pts[:,2], time)
  
        fbody = getBodyForce(element, fvalue)

      # Assemble in the global array
        el_eqns = getEqns(domain, iele)
        el_eqns_active = (el_eqns .>= 1)
        Fbody[el_eqns[el_eqns_active]] += fbody[el_eqns_active]
    end
  
    return Fbody
end


@doc raw"""
    getEdgeForce(domain::Domain, globdat::GlobalData, time::Float64)

Computes the edge force vector $F_\mathrm{edge}$ defined in `domain.edge_traction_data`
- `globdat`: GlobalData
- `domain`: Domain, finite element domain, for data structure
- `time`:  Float64, current time step size
"""
function getEdgeForce(domain::Domain, globdat::GlobalData, time::Float64)
    Fedge = zeros(Float64, domain.neqs)
    neles = domain.neles

    if (isnothing(globdat.Edge_func) && size(domain.edge_traction_data, 1)!=0) ||
        (!isnothing(globdat.Edge_func) && size(domain.edge_traction_data, 1)==0)
        @warn("`GlobalData` does not have (has) `Edge_func` but `Domain` has (does not have) `edge_traction_data`")
    end

    if isnothing(globdat.Edge_func)
        return Fedge
    end

    # Loop over the elements in the elementGroup

    for i = 1:size(domain.edge_traction_data)[1]

        iele, iedge, ifunc = domain.edge_traction_data[i, :]

        element = domain.elements[iele]
  
        gauss_pts = getEdgeGaussPoints(element, iedge)

        fvalue = globdat.Edge_func(gauss_pts[:,1], gauss_pts[:,2], time, ifunc)
  
        fedge = getEdgeForce(element, iedge, fvalue)

      # Assemble in the global array
        el_eqns = getEqns(domain, iele)
        el_eqns_active = (el_eqns .>= 1)
        Fedge[el_eqns[el_eqns_active]] += fedge[el_eqns_active]
    end
  
    return Fedge
end


@doc """
    getCoords(domain::Domain, el_nodes::Array{Int64})

Get the coordinates of several nodes (possibly in one element)
- `domain`: Domain
- `el_nodes`: Int64[n], node array

Return: Float64[n, ndims], the coordinates of these nodes
"""
function getCoords(domain::Domain, el_nodes::Array{Int64})
    return domain.nodes[el_nodes, :]
end

@doc """
    getDofs(domain::Domain, iele::Int64)   

Get the global freedom numbers of the element
- `domain`: Domain
- `iele`: Int64, element number

Return: Int64[], the global freedom numbers of the element (ordering in local element ordering)
""" 
function getDofs(domain::Domain, iele::Int64)    
    return domain.DOF[iele]
end

@doc """
    getNGauss(domain::Domain)

Gets the total number of Gauss quadrature points. 
"""
function getNGauss(domain::Domain)
    ng = 0
    for e in domain.elements
        ng += length(e.weights)
    end
    ng
end

@doc """
    getEqns(domain::Domain, iele::Int64)

Gets the equation numbers (active freedom numbers) of the element. 
This excludes both the time-dependent and time-independent Dirichlet boundary conditions. 
""" 
function getEqns(domain::Domain, iele::Int64)
    return domain.LM[iele]
end


@doc """
    getState(domain::Domain, el_dofs::Array{Int64})

Get the displacements of several nodes (possibly in one element)
- `domain`: Domain
- `el_nodes`: Int64[n], node array

Return: Float64[n, ndims], the displacements of these nodes
""" 
function getState(domain::Domain, el_dofs::Array{Int64})
    return domain.state[el_dofs]
end

@doc """
    getDstate(domain::Domain, el_dofs::Array{Int64})
    
Get the displacements of several nodes (possibly in one element) at the previous time step
- `domain`: Domain
- `el_nodes`: Int64[n], node array

Return: Float64[n, ndims], the displacements of these nodes at the previous time step
""" ->

function getDstate(domain::Domain, el_dofs::Array{Int64})
    return domain.Dstate[el_dofs]
end


@doc raw"""
    getGaussPoints(domain::Domain)

Returns all Gauss points as a $n_g\times 2$ matrix, where $n_g$ is the total number of Gauss points.
"""
function getGaussPoints(domain::Domain)
    v = []
    for e in domain.elements
        vg = getGaussPoints(e) 
        push!(v, vg)
    end 
    vcat(v...)
end

@doc raw"""
    getStrain(domain::Domain)

Computes the strain from the `domain` data. The output is $n_g\times 3$ matrix, where $n_g$ is the total number of Gauss points. 
Each row is the strain tensor $\begin{bmatrix} \epsilon_{xx} & \epsilon_{yy} & \gamma_{xy} \end{bmatrix}$
"""
function getStrain(domain::Domain)
    E = Array{Array{Float64}}(undef, domain.neles)
    for (k,e) in enumerate(domain.elements)
        el_dof = getDofs(domain, k)
        state = getState(domain, el_dof)
        E[k], _ = getStrain(e, state)
    end
    vcat(E...)
end

@doc raw"""
    getDStrain(domain::Domain)

Computes the strain **at last time step** from the `domain` data. The output is $n_g\times 3$ matrix, where $n_g$ is the total number of Gauss points. 
Each row is the strain tensor $\begin{bmatrix} \epsilon_{xx} & \epsilon_{yy} & \gamma_{xy} \end{bmatrix}$
"""
function getDStrain(domain::Domain)
    E = Array{Array{Float64}}(undef, domain.neles)
    for (k,e) in enumerate(domain.elements)
        el_dof = getDofs(domain, k)
        Dstate = getDstate(domain, el_dof)
        E[k], _ = getDStrain(e, Dstate)
    end
    vcat(E...)
end

@doc raw"""
    getStress(domain::Domain, Δt::Float64 = 0.0; save_trace::Bool = false)

Returns the stress based on `domain.state` and `domain.Dstate`. If `save_trace` is true, 
the stress is also saved to `domain.stress`, which is useful for visualization. 

The output is $n_g\times 3$ matrix, where $n_g$ is the total number of Gauss points. 
Each row is the strain tensor $\begin{bmatrix} \sigma_{xx} & \sigma_{yy} & \sigma_{xy} \end{bmatrix}$
"""
function getStress(domain::Domain, Δt::Float64 = 0.0; save_trace::Bool = false)
    Ret = zeros(getNGauss(domain), 3)
    local ∂E∂u, E, DE
    l = 1
    for (k,elem) in enumerate(domain.elements)
        nnodes = length(elem.elnodes)
        el_dof = getDofs(domain,k)
        state = getState(domain, el_dof)
        Dstate = getDstate(domain, el_dof)
        u = state[1:nnodes]; v = state[nnodes+1:2*nnodes]
        Du = Dstate[1:nnodes]; Dv = Dstate[nnodes+1:2*nnodes]

        for k = 1:length(elem.weights)
            # #@show "Gaussian point ", k
            g1 = elem.dhdx[k][:,1]; g2 = elem.dhdx[k][:,2]
            
            ux = u'*g1; uy = u'*g2; vx = v'*g1; vy = v'*g2
            Dux = Du'*g1; Duy = Du'*g2; Dvx = Dv'*g1; Dvy = Dv'*g2
            
            if isa(elem, SmallStrainContinuum)
                ∂E∂u = [g1   zeros(nnodes)    g2;
                        zeros(nnodes)    g2   g1;] 
                E = [ux; vy; uy+vx]
                DE = [Dux; Dvy; Duy+Dvx]
            elseif isa(elem, FiniteStrainContinuum)
                ∂E∂u = [g1+ux*g1 uy*g2    g2 + g2*ux+g1*uy;
                vx*g1    g2+vy*g2 g1 + g1*vy+g2*vx;] 
                E = [ux+0.5*(ux*ux+vx*vx); vy+0.5*(uy*uy+vy*vy); uy+vx+ux*uy+vx*vy]
                DE = [Dux+0.5*(Dux*Dux+Dvx*Dvx); Dvy+0.5*(Duy*Duy+Dvy*Dvy); Duy+Dvx+Dux*Duy+Dvx*Dvy]
            else 
                error("NOT IMPLEMENTED")
            end

            S, dS_dE = getStress(elem.mat[k], E, DE, Δt)
            Ret[l,:] = S 
            l += 1

            if save_trace
                elem.stress[k] = S
            end
        end
    end
    Ret
end

@doc raw"""
    getElems(domain::Domain)

Returns the element connectivity matrix $n_e \times 4$. This function implicitly assumes that all elements are quadrilateral.
"""
function getElems(domain::Domain)
    elem = zeros(Int64, domain.neles, 4)
    for (k,e) in enumerate(domain.elements)
        elem[k,:] = e.elnodes
    end
    return elem
end


#------------------------------------------------------------------------------------
# Maybe useful in the future


@doc """
    Compute constant stiff, dfint_dstress, dstrain_dstate matrix patterns
    - 'self': Domain
"""
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



@doc """
    In the constructor 
    Update the node_to_point map 

    - `self`: Domain, finit element domain
    - 'npoints': Int64, number of points (each quadratical quad element has 4 points, npoints==nnodes, when porder==1)
    - 'node_to_point': Int64[nnodes]:map from node number to point point, -1 means the node is not a geometry point

""" 
function setGeometryPoints!(self::Domain, npoints::Int64, node_to_point::Array{Int64})
    self.npoints = npoints
    self.node_to_point = node_to_point
end
