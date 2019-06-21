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


from numpy import linspace, meshgrid, empty
import sys
sys.path.insert(0, '../fem')
sys.path.insert(0, '../elements')

from FiniteStrainContinuum import FiniteStrainContinuum
from Domain import Domain
from DataStructures import GlobalData


nx, ny =  10, 5
nnodes, neles = (nx + 1)*(ny + 1), nx*ny
x = linspace(0.0, 1.0, nx + 1)
y = linspace(0.0, 0.2, ny + 1)
X, Y = meshgrid(x, y)
nodes = empty((nnodes,2))
nodes[:,0], nodes[:,1] = X.flatten(), Y.flatten()

prop = {'type': 'PlaneStrain', 'E': 1000.0, 'nu': 0.4}

elements = []
for i in range(nx):
  for j in range(ny):
    n = (nx+1)*j + i
    elnodes = [n, n + 1, n + 1 + (nx + 1), n + (nx + 1)]

    FiniteStrainContinuum(elnodes, prop)

    elements.append(FiniteStrainContinuum(elnodes, prop))



dofs = 2*nnodes

plate = Domain(nodes, elements, dofs)
globdat = GlobalData(plate.neqs)

#solver = Solver        ( props , globdat )
#solver.run( props , globdat )