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

from numpy import zeros, ones, ix_
from pyfem.util.dataStructures import Properties
from pyfem.util.dataStructures import elementData


#######################################
# General array assembly routine for: # 
# * assembleInternalForce             #
# * assembleTangentStiffness          #
#######################################




##########################################
# Internal force vector assembly routine # 
##########################################

def assembleInternalForce(globdat, domain):
  # Initialize the global array A with rank 2


  Fint = zeros(len(globdat.dofs) * ones(1, dtype=int))

  neles = domain.neles

  # Loop over the elements in the elementGroup
  for iele in range(neles):
    element = domain.elements[iele]

    # Get the element nodes

    el_dofs = element.get_dofs()

    el_disp  = globdat.state[el_dofs]

    el_Ddisp = globdat.Dstate[el_dofs]

    # Get the element contribution by calling the specified action
    fint = element.getInternalForce(el_disp, el_Ddisp)

    # Assemble in the global array

    Fint[el_dofs] += fint

    return Fint


#############################################
# Tangent stiffness matrix assembly routine # 
#############################################

def assembleStiffAndForce(globdat, domain):
  # Initialize the global array A with rank 2

  Fint = zeros(len(globdat.dofs) * ones(1, dtype=int))
  K = zeros(len(globdat.dofs) * ones(1, dtype=int))

  neles = domain.neles

  # Loop over the elements in the elementGroup
  for iele in range(neles):
    element = domain.elements[iele]

    # Get the element nodes

    el_dofs = element.get_dofs()

    el_disp = globdat.state[el_dofs]

    el_Ddisp = globdat.Dstate[el_dofs]

    # Get the element contribution by calling the specified action
    fint, stiff  = element.getInternalForce(el_disp, el_Ddisp)

    # Assemble in the global array

    Fint[el_dofs] += fint

    K[ix_(el_dofs, el_dofs)] += stiff
    Fint[el_dofs] += fint

    return Fint, K

#############################################
# Mass matrix assembly routine              # 
#############################################

def assembleMassMatrix ( globdat, domain ):
  # Initialize the global array A with rank 2

  Mlumped = zeros(len(globdat.dofs) * ones(1, dtype=int))
  M = zeros(len(globdat.dofs) * ones(1, dtype=int))

  neles = domain.neles

  # Loop over the elements in the elementGroup
  for iele in range(neles):
    element = domain.elements[iele]

    # Get the element nodes

    el_dofs = element.get_dofs()

    el_disp = globdat.state[el_dofs]

    el_Ddisp = globdat.Dstate[el_dofs]

    # Get the element contribution by calling the specified action
    lM, lMlumped = element.getMassMatrix()

    # Assemble in the global array

    M[ix_(el_dofs, el_dofs)] += lM
    Mlumped[el_dofs] += lMlumped

    return M, Mlumped
