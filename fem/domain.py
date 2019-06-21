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
  
  def __init__ (self, nodes, elements, dofs):
    '''

    :param nodes: n by nDim array, node coordinates
    :param elements: elements array
    :param dofs: total number of freedoms
    '''

    self.nnodes = len(nodes)
    self.nodes = nodes  #coordinates
    self.neles = len(elements)
    self.elements = elements  #element set

    self.dofs = dofs  #number of total freedoms

    self.disp = zeros(dofs)

    self.vel = zeros(dofs)

  def setBoundary(self, ebc, g, nbc, ):
    '''

    :param ebc:
    :param g:
    :param nbc:
    :return:
    '''

    node_to_eqn
    eqn_to_node

  def updateSates(self, disp, vel, time):
    '''
    :param disp: neqs array
    :param vel : neqs array

    update Dirichlet boundary
    :return:
    '''

  def getDofs(self, el_nodes):
    '''

    :param el_nodes: 1d array
    :return: the corresponding equation ids, u0,u1, .., v0, v1, ..
    '''


  def getDisp(self, el_nodes):
    '''

    :param el_nodes: 1d array
    :return: the displacement at these nodes ux0 ux1 .. uy0 uy1 ..
    '''

    return self.disp[el_nodes]

  def getVel(self, el_nodes):
    '''

    :param el_nodes: 1d array
    :return: the displacement at these nodes vx0 vx1 .. vx0 vx1 ..
    '''
    return self.vel[el_nodes]
     



