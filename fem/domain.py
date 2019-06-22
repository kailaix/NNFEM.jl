############################################################################
#  This Python file is part of PyFEM-1.0, released on Aug. 29, 2012.       #
#  The PyFEM code accompanies the book:                                    #
#                                                                          #
#    'Non-Linear Finite Element Analysis of Solids and Structures'         #
#    R. de Borst, M.A. Crisfield, J.J.C. Remmers and C.V. Verhoosel        #
#    John Wiley and Sons, 2012, ISBN 978-0470666449                        #
#                                                                          #
#  The code is written by J.J.C. Remmers, C.V. Verhoosel and R. de Borst.  #
#  Comments and suggestions can be sent to:                                #
#     PyFEM-support@tue.nl                                                 #
#                                                                          #
#  The latest version can be downloaded from the web-site:                 #                                                                          
#     http://www.wiley.com/go/deborst                                      #
#                                                                          #
#  The code is open source and intended for educational and scientific     #
#  purposes only. If you use PyFEM in your research, the developers would  #
#  be grateful if you could cite the book.                                 #  
#                                                                          #
#  Disclaimer:                                                             #
#  The authors reserve all rights but do not guarantee that the code is    #
#  free from errors. Furthermore, the authors shall not be liable in any   #
#  event caused by the use of the program.                                 #
############################################################################

from numpy import zeros


class Domain:
  '''
  Class represents a structure and contains all auxiliary data-structure
  '''
  
  def __init__ (self, nodes, elements, ndofs, EBC, g):
    '''

    :param nodes: n by nDim array, node coordinates
    :param elements: elements array
    :param dofs: total number of freedoms
    '''

    self.nnodes = len(nodes)
    self.nodes = nodes  #coordinates
    self.neles = len(elements)
    self.elements = elements  #element set

    self.ndofs = ndofs  #nodal degrees of freedoms



    self.state = zeros(self.nnodes * ndofs)

    self.Dstate = zeros(self.nnodes * ndofs)

    self.setBoundary(EBC, g)

  def setBoundary(self, EBC, g):
    '''

    :param EBC[n, d] is the boundary condition of of node n's dth freedom,
           -1 means fixed Dirichlet boundary nodes
           -2 means time dependent Dirichlet boundary nodes
    :param g[n, d] is the fixed Dirichlet boundary value

    :param nbc:
    :return:
    '''
    #todo also set NBC

    self.EBC, self.g = EBC, g
    # ID(n,d) is the global equation number of node n's dth freedom, -1 means no freedom
    nnodes, ndofs = self.nnodes, self.ndofs
    neles, elements = self.neles, self.elements
    ID = zeros([nnodes, ndofs], dtype='int') - 1

    eq_to_dof = []
    neqs = 0
    for idof in range(ndofs):
      for inode in range(nnodes):
          if (EBC[inode, idof] == 0):
              ID[inode, idof] = neqs
              neqs += 1
              eq_to_dof.append(inode + idof*nnodes)

    self.ID, self.neqs, self.eq_to_dof = ID, neqs, eq_to_dof

    # LM(e,d) is the global equation number of element e's d th freedom
    LM = []
    for iele in range(neles):
      el_nodes = elements[iele].getNodes()
      ieqns = ID[el_nodes, :].flatten('F')
      LM.append(ieqns)
    self.LM = LM

    # DOF(e,d) is the global dof number of element e's d th freedom

    DOF = []
    for iele in range(neles):
      el_nodes = elements[iele].getNodes()
      el_dofs = el_nodes[:]
      #todo assume each node has two dofs
      el_dofs.extend([idof + nnodes for idof in range(len(el_nodes))])
      DOF.append(el_dofs)
    self.DOF = DOF







  def updateSates(self, state, Dstate, time):
    '''
    :param disp: neqs array
    :param vel : neqs array

    update Dirichlet boundary
    :return:
    '''

    self.state[self.eq_to_dof] = state
    self.Dstate[self.eq_to_dof] = Dstate

    #todo also update time-dependent Dirichlet boundary

  def getCoords(self, el_nodes):
    return self.nodes[el_nodes, :]

  def getDofs(self, iele):
    '''

    :param el_nodes: 1d array
    :return: the corresponding dofs ids, u0,u1, .., v0, v1, ..
    '''

    return self.DOF[iele]

  def getEqns(self, iele):
    '''

    :param el_nodes: 1d array
    :return: the corresponding equation ids, u0,u1, .., v0, v1, ..
    '''
    return self.LM[iele]

  def getState(self, el_dofs):
    '''

    :param el_nodes: 1d array
    :return: the displacement at these nodes ux0 ux1 .. uy0 uy1 ..
    '''

    return self.state[el_dofs]

  def getDstate(self, el_dofs):
    '''

    :param el_nodes: 1d array
    :return: the displacement at these nodes vx0 vx1 .. vx0 vx1 ..
    '''
    return self.Dstate[el_dofs]
     



