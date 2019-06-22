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

from numpy import zeros, array, dot
from Assembly import assembleInternalForce, assembleMassMatrix

import sys

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

class ExplicitSolver:

  def __init__( self , globdat, domain , maxint = 10000):
    
    self.maxCycle = maxint
    
    M,self.Mlumped = assembleMassMatrix( globdat, domain )


  def run( self , globdat, domain ):

    globdat.cycle += 1
    globdat.time  += self.dtime

    lam  = self.loadfunc( globdat.time )
    
    disp = globdat.state
    velo = globdat.velo
    acce = globdat.acce

    fint = globdat.fint
    fhat = globdat.fhat
    
    velo += 0.5*self.dtime * acce
    disp += self.dtime * velo
    
    fint  = assembleInternalForce( globdat, domain )

    acce = globdat.dofs.solve( self.Mlumped , fhat-fint )
       
    velo += 0.5 * self.dtime * acce

    globdat.acce[:] = acce[:]
  
    globdat.elements.commitHistory()

    self.printStep( globdat )
    
    if globdat.cycle == self.maxCycle:
      globdat.active = False

#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------

  def printStep( self , globdat ):
 
    if globdat.cycle%20 == 0 or globdat.cycle == 1:
      print("  Cycle     Time         Kin.Energy")
      print("  ---------------------------------------")
  
    print(' %5i ' % globdat.cycle,)
    print(' %10.3e ' % globdat.time,)
    print(' %10.3e ' % float(0.5*dot(globdat.velo,(self.Mlumped*globdat.velo))))
