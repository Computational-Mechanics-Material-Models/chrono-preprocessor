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
## Function to generate a VTK file for visualization in Paraview of particles
## and their corresponding diameter for LDPM and CSL models.
##
## ===========================================================================

from pathlib import Path


def mkVtkSingleTetParticles(allNodes,allTets,allDiameters,geoName,tempPath):

    """
    Variables:
    --------------------------------------------------------------------------
    ### Inputs ###
    - allNodes:     List of all nodes
    - allDiameters: List of all particle diameters (including fictitious)
    - allTets:      List of all tetrahedrons
    - geoName:      Name of the geometry
    - tempPath:     Path to the temporary directory
    --------------------------------------------------------------------------
    ### Outputs ###
    - A VTK file for visualizing particles
    --------------------------------------------------------------------------
    """
    
    # Use the tet in the middle of the list so that it is not on the boundary
    tet = allTets[round(len(allTets)/2),:]-1

    # Extract the nodes for the first tetrahedron
    allNodes = [allNodes[int(tet[0])],\
              allNodes[int(tet[1])],\
              allNodes[int(tet[2])],\
              allNodes[int(tet[3])]]

    # Extract the particle diameters for the first tetrahedron
    allDiameters = [allDiameters[int(tet[0])],\
                   allDiameters[int(tet[1])],\
                   allDiameters[int(tet[2])],\
                   allDiameters[int(tet[3])]]

    # Generate VTK file for visualizing particles
    with open(Path(tempPath + geoName + \
        '-para-singleTetParticles.000.vtk'),"w") as f:                                       
        f.write('# vtk DataFile Version 2.0\n')
        f.write('Unstructured grid legacy vtk file with point scalar data\n')            
        f.write('ASCII\n')    
        f.write('\n')  
        f.write('DATASET UNSTRUCTURED_GRID\n')        
        f.write('POINTS ' + str(len(allNodes)) + ' double \n')  
        f.write("\n".join(" ".join(map(str, x)) for x in allNodes))
        f.write('\n\n')  
        f.write('POINT_DATA ' + str(len(allNodes)) + '\n')
        f.write('SCALARS Diameter double\n')
        f.write('LOOKUP_TABLE default\n')
        for x in allDiameters:
            f.write("%s\n" % x)
        f.write('\n')  
