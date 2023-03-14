## ===========================================================================
## CHRONO WORKBENCH:github.com/Concrete-Chrono-Development/chrono-preprocessor
##
## Copyright (c) 2023 
## All rights reserved. 
##
## Use of this source code is governed by a BSD-style license that can be
## found in the LICENSE file at the top level of the distribution and at
## github.com/Concrete-Chrono-Development/chrono-preprocessor/blob/main/LICENSE
##
## ===========================================================================
## Developed by Northwestern University
## Primary Authors: Matthew Troemner
## ===========================================================================
##
## Function to generate VTK file for visualization in Paraview of facets
## and their corresponding material for LDPM models.
##
## ===========================================================================

import numpy as np
from pathlib import Path


def mkVtksingleTetFacets(geoName,tempPath,tetFacets):

    """
    Variables:
    --------------------------------------------------------------------------
    ### Inputs ###
    - geoName:          Name of the geometry file
    - tempPath:         Path to the temporary directory
    - tetFacets:        Facet data from the tetrahedral mesh
    --------------------------------------------------------------------------
    ### Outputs ###
    - A VTK file that can be visualized in Paraview
    --------------------------------------------------------------------------
    """

    # Facets for a single tet (12)
    singleTetFacetPoints = tetFacets.reshape(-1,3)[0:36,:]
    singleTetFacetCells = np.array([[0,1,2],\
                                     [3,4,5],\
                                     [6,7,8],\
                                     [9,10,11],\
                                     [12,13,14],\
                                     [15,16,17],\
                                     [18,19,20],\
                                     [21,22,23],\
                                     [24,25,26],\
                                     [27,28,29],\
                                     [30,31,32],\
                                     [33,34,35]])

    with open(Path(tempPath + geoName + \
        '-para-singleTetFacets.000.vtk'),"w") as f:                                                                          
        f.write('# vtk DataFile Version 2.0\n')
        f.write('Facet Visual File\n') 
        f.write('ASCII\n')    
        f.write('DATASET POLYDATA\n')        

        f.write('POINTS ' + str(len(singleTetFacetPoints)) + ' float \n') 
        f.write("\n".join(" ".join(map(str, x)) for x in singleTetFacetPoints))
        f.write('\n\n')  

        f.write('POLYGONS ' + str(len(singleTetFacetCells)) + ' ' \
            + str(round(len(singleTetFacetCells)*4)) +'\n3 ')
        f.write("\n3 ".join(" ".join(map(str, x)) for x in singleTetFacetCells))

        f.write('\n\n')  
