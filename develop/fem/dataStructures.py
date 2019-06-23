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

class GlobalData:
  
  def __init__ ( self, neqs ):

    self.state  = zeros( neqs )
    self.Dstate = zeros( neqs )


    self.velo   = zeros( neqs )
    self.acce   = zeros( neqs )

    self.fint = zeros(neqs)
    self.fext = zeros(neqs)

    self.cycle  = 0
    self.iiter  = 0
    self.time   = 0.0


class Properties:
  def __init__(self, dictionary={}):

    for key in dictionary.iterkeys():
      setattr(self, key, dictionary[key])

  def __str__(self):

    myStr = ''
    for att in dir(self):

      # Ignore private members and standard routines
      if att.startswith('__'):
        continue

      myStr += 'Attribute: ' + att + '\n'
      myStr += str(getattr(self, att)) + '\n'

    return myStr

  def __iter__(self):

    propsList = []
    for att in dir(self):

      # Ignore private members and standard routines
      if att.startswith('__'):
        continue

      propsList.append((att, getattr(self, att)))

    return iter(propsList)

  def store(self, key, val):
    setattr(self, key, val)

