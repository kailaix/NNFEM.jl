export Domain,GlobalData,updateStates


mutable struct GlobalData
    state::Array{Float64}
    Dstate::Array{Float64}
    velo::Array{Float64}
    acce::Array{Float64}
    time::Float64
    M::Array{Float64}
    Mlumped::Array{Float64}
end

function GlobalData(state::Array{Float64},Dstate::Array{Float64},velo::Array{Float64},acce::Array{Float64}, neqs::Int64)
    time = 0.0
    M = Float64[]
    Mlumped = Float64[]
    GlobalData(state, Dstate, velo, acce, time, M, Mlumped)
end


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
    EBC::Array{Int64}  # Dirichlet boundary condition
    g::Array{Float64}  # Value for Dirichlet boundary condition
    NBC::Array{Int64}  # Nodal force boundary condition
    fext::Array{Float64}  # Value for Nodal force boundary condition
    time::Float64
    
end

@doc """
    Creating a finite element domain
    nodes: n by 2 float 64, node coordinates
    elements: element list
    ndims: 2
    EBC: n by 2 Int64, nodal Dirichlet boundary condition, -1 time-independent, -2 time-dependent
    g: n by 2 Float64, values for nodal time-independent Dirichlet boundary condition
    NBC: n by 2 Int64, nodal force boundary condition, -1 time-independent, -2 time-dependent
    f: n by 2 Float64, values for nodal force time independent force boundary condition

"""->
function Domain(nodes::Array{Float64}, elements::Array, ndims::Int64, EBC::Array{Int64}, g::Array{Float64}, NBC::Array{Int64}, f::Array{Float64})
    nnodes = size(nodes,1)
    neles = size(elements,1)
    state = zeros(nnodes * ndims)
    Dstate = zeros(nnodes * ndims)
    LM = Array{Int64}[]
    DOF = Array{Int64}[]
    ID = Int64[]
    neqs = 0
    eq_to_dof = Int64[]
    fext = Float64[]
    
    domain = Domain(nnodes, nodes, neles, elements, ndims, state, Dstate, LM, DOF, ID, neqs, eq_to_dof, EBC, g, NBC, fext, 0.0)
    setDirichletBoundary!(domain, EBC, g)
    setNeumannBoundary!(domain, NBC, f)
    domain
end

function commitHistory(domain::Domain)
    for e in domain.elements
        commitHistory(e.mat)
    end
end

@doc """

    :param EBC[n, d] is the boundary condition of of node n's dth freedom,
        -1 means fixed Dirichlet boundary nodes
        -2 means time dependent Dirichlet boundary nodes
    :param g[n, d] is the fixed Dirichlet boundary value

    :param nbc:
    :return:
""" -> 
function setDirichletBoundary!(self::Domain, EBC::Array{Int64}, g::Array{Float64})

    # ID(n,d) is the global equation number of node n's dth freedom, -1 means no freedom
    nnodes, ndims = self.nnodes, self.ndims
    neles, elements = self.neles, self.elements
    ID = zeros(Int64, nnodes, ndims) .- 1

    eq_to_dof = Int64[]
    neqs = 0
    for idof = 1:ndims
      for inode = 1:nnodes
          if (EBC[inode, idof] == 0)
              neqs += 1
              ID[inode, idof] = neqs
              push!(eq_to_dof,inode + (idof-1)*nnodes)
          end
        end
    end

    self.ID, self.neqs, self.eq_to_dof = ID, neqs, eq_to_dof


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

    :param EBC[n, d] is the boundary condition of of node n's dth freedom,
        -1 means fixed Dirichlet boundary nodes
        -2 means time dependent Dirichlet boundary nodes
    :param g[n, d] is the fixed Dirichlet boundary value

    :param nbc:
    :return:
""" -> 
function setNeumannBoundary!(self::Domain, NBC::Array{Int64}, f::Array{Float64})

    fext = zeros(Float64, self.neqs)
    # ID(n,d) is the global equation number of node n's dth freedom, -1 means no freedom

    nnodes, ndims, ID = self.nnodes, self.ndims, self.ID
    for idof = 1:ndims
      for inode = 1:nnodes
          if (NBC[inode, idof] == -1)
              @assert ID[inode, idof] > 0
              fext[ID[inode, idof]] += f[inode, idof]
          end
        end
    end
    self.fext = fext
end


@doc """
    :param disp: neqs array
    :param vel : neqs array

    update Dirichlet boundary
    :return:
""" ->
function updateStates(self::Domain, globaldat::GlobalData)
    self.state[self.eq_to_dof] = globaldat.state
    self.Dstate[self.eq_to_dof] = globaldat.Dstate
    self.time = globaldat.time
    #todo also update time-dependent Dirichlet boundary/force load boundary
end

function getCoords(self::Domain, el_nodes::Array{Int64})
    return self.nodes[el_nodes, :]
end

@doc """
    :param el_nodes: 1d array
    :return: the corresponding dofs ids, u0,u1, .., v0, v1, ..
""" ->
function getDofs(self::Domain, iele::Int64)    
    return self.DOF[iele]
end

function getEqns(self::Domain, iele::Int64)
    return self.LM[iele]
end

function getState(self::Domain, el_dofs::Array{Int64})
    return self.state[el_dofs]
end

function getDstate(self::Domain, el_dofs::Array{Int64})
    return self.Dstate[el_dofs]
end
