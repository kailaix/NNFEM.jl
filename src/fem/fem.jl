export Domain,GlobalData
mutable struct Domain
    nnodes::Int64
    nodes::Array{Float64}
    neles::Int64
    elem::Array
    ndims::Int64
    state::Array{Float64}
    Dstate::Array{Float64}
    LM::Array{Array{Int64}}
    DOF::Array{Array{Int64}}
    ID::Array{Int64}
    neqs::Int64
    EBC::Array{Int64}  # Dirichlet boundary condition
    g::Array{Float64}  # Value for EBC
    eq_to_dof::Array{Int64}
end

@doc """
    Creating a finite element domain 
"""->
function Domain(nodes::Array{Float64}, elements::Array, ndims::Int64, EBC::Array{Int64}, g::Float64)
    nnodes = size(nodes,1)
    neles = size(elements,1)
    state = zeros(nnodes, ndims)
    Dstate = zeros(nnodes, ndims)
    LM = Array{Int64}[]
    DOF = Array{Int64}[]
    ID = Int64[]
    neqs = 0
    EBC = Int64[]
    g = Float64[]
    eq_to_dof = Int64[]
    Domain(nnodes, nodes, neles, elements, ndims, state, Dstate, LM, DOF, ID, neqs, EBC, g, eq_to_dof)
end

@doc """

    :param EBC[n, d] is the boundary condition of of node n's dth freedom,
        -1 means fixed Dirichlet boundary nodes
        -2 means time dependent Dirichlet boundary nodes
    :param g[n, d] is the fixed Dirichlet boundary value

    :param nbc:
    :return:
""" -> 
function setBoundary(self::Domain, EBC::Array{Int64}, g::Array{Float64})

    self.EBC, self.g = EBC, g
    # ID(n,d) is the global equation number of node n's dth freedom, -1 means no freedom
    nnodes, ndims = self.nnodes, self.ndims
    neles, elements = self.neles, self.elements
    ID = zeros(Int64, nnodes, ndims) .- 1

    eq_to_dof = Int64[]
    neqs = 0
    for idof = 1:ndims
      for inode = 1:nnodes
          if (EBC[inode, idof] == 0):
              ID[inode, idof] = neqs
              neqs += 1
              push!(eq_to_dof,inode + (idof-1)*nnodes)
          end
        end
    end

    self.ID, self.neqs, self.eq_to_dof = ID, neqs, eq_to_dof

    # LM(e,d) is the global equation number of element e's d th freedom
    LM = Array{Array{Int64}}(undef, neles)
    for iele = 1:neles
      el_nodes = getNodes(elements,iele)
      ieqns = ID[el_nodes, :][:]
      LM[iele] = ieqns
    end
    self.LM = LM

    # DOF(e,d) is the global dof number of element e's d th freedom

    DOF = Array{Array{Int64}}(undef, neles)
    for iele = 1:neles
      el_nodes = getNodes(elements,iele)
      DOF[iele] = [el_nodes;[idof + nnodes for idof = 1:length(el_nodes)]]
    end
    self.DOF = DOF
    
end

@doc """
    :param disp: neqs array
    :param vel : neqs array

    update Dirichlet boundary
    :return:
""" ->
function updateStates(self::Domain, state::Array{Float64}, Dstate::Array{Float64}, time::Float64)
    self.state[self.eq_to_dof] = state
    self.Dstate[self.eq_to_dof] = Dstate

    #todo also update time-dependent Dirichlet boundary
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

function getEqns(self::Domain, iele::Array{Int64})
    return self.LM[iele]
end

function getState(self::Domain, el_dofs::Array{Int64})
    return self.state[el_dofs]
end

function getDstate(self::Domain, el_dofs::Array{Int64})
    return self.Dstate[el_dofs]
end

mutable struct GlobalData
    state::Array{Float64}
    Dstate::Array{Float64}
    velo::Array{Float64}
    acce::Array{Float64}
    finit::Array{Float64}
    fext::Array{Float64}
    time::Float64
end

function GlobalData()
    state = Float64[]
    Dstate = Float64[]
    velo = Float64[]
    acce = Float64[]
    finit = Float64[]
    fext = Float64[]
    time = 0.0
    GlobalData(state, Dstate, velo, acce, finit, fext, time)
end