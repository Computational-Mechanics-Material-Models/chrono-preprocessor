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
## This file contains the function to check if a particle is inside a tet
##
## ===========================================================================

import numpy as np

def checkParticleInside(vertices,tets,center,parDiameter,binMin,binMax,coord1,\
    coord2,coord3,coord4):

    """
    Variables:
    --------------------------------------------------------------------------
    ### Inputs ###
    - vertices:         Nodes of the tets
    - tets:             Tets of the mesh
    - center:           Center of the particle
    - parDiameter:      Diameter of the particle
    - binMin:           Minimum coordinates of the bin
    - binMax:           Maximum coordinates of the bin
    - coord1:           Coordinates of the first vertex of each tet
    - coord2:           Coordinates of the second vertex of each tet
    - coord3:           Coordinates of the third vertex of each tet
    - coord4:           Coordinates of the fourth vertex of each tet
    --------------------------------------------------------------------------
    ### Outputs ###
    - Boolean:          True if the particle is inside tets, False if not
    --------------------------------------------------------------------------
    """  

    # Convert center to a 1D array
    center = center.flatten()

    # Store tet vertices that fall inside the bin
    coord1 = np.all([(coord1[:,0] > binMin[0]) , (coord1[:,0] < binMax[0]),\
                     (coord1[:,1] > binMin[1]) , (coord1[:,1] < binMax[1]),\
                     (coord1[:,2] > binMin[2]) , (coord1[:,2] < binMax[2])],axis=0)      
    coord2 = np.all([(coord2[:,0] > binMin[0]) , (coord2[:,0] < binMax[0]),\
                     (coord2[:,1] > binMin[1]) , (coord2[:,1] < binMax[1]),\
                     (coord2[:,2] > binMin[2]) , (coord2[:,2] < binMax[2])],axis=0)          
    coord3 = np.all([(coord3[:,0] > binMin[0]) , (coord3[:,0] < binMax[0]),\
                     (coord3[:,1] > binMin[1]) , (coord3[:,1] < binMax[1]),\
                     (coord3[:,2] > binMin[2]) , (coord3[:,2] < binMax[2])],axis=0)  
    coord4 = np.all([(coord4[:,0] > binMin[0]) , (coord4[:,0] < binMax[0]),\
                     (coord4[:,1] > binMin[1]) , (coord4[:,1] < binMax[1]),\
                     (coord4[:,2] > binMin[2]) , (coord4[:,2] < binMax[2])],axis=0)  

    # Get tets with a vertex that falls inside the bin
    binTets = np.any([coord1,coord2,coord3,coord4],axis=0)

    # Store vertices of those tets
    coord1 = vertices[tets.astype(int)[binTets,0]-1]
    coord2 = vertices[tets.astype(int)[binTets,1]-1]
    coord3 = vertices[tets.astype(int)[binTets,2]-1]
    coord4 = vertices[tets.astype(int)[binTets,3]-1]

    # Get the 8 vertices which make up a bounding box around the particle
    node = np.empty((8,3))
    node[0,:] = [center[0]+parDiameter/2,center[1]+parDiameter/2,\
                 center[2]+parDiameter/2]
    node[1,:] = [center[0]-parDiameter/2,center[1]+parDiameter/2,\
                 center[2]+parDiameter/2]
    node[2,:] = [center[0]+parDiameter/2,center[1]-parDiameter/2,\
                 center[2]+parDiameter/2]
    node[3,:] = [center[0]+parDiameter/2,center[1]+parDiameter/2,\
                 center[2]-parDiameter/2]
    node[4,:] = [center[0]-parDiameter/2,center[1]-parDiameter/2,\
                 center[2]+parDiameter/2]
    node[5,:] = [center[0]+parDiameter/2,center[1]-parDiameter/2,\
                 center[2]-parDiameter/2]
    node[6,:] = [center[0]-parDiameter/2,center[1]+parDiameter/2,\
                 center[2]-parDiameter/2]
    node[7,:] = [center[0]-parDiameter/2,center[1]-parDiameter/2,\
                 center[2]-parDiameter/2]

    # Check whether all 8 vertices fall inside of tets
    inside = 0
    emptyOnes = np.ones(len(coord1[:,0]))

    D00 = np.rot90(np.dstack((coord1[:,0],coord1[:,1],coord1[:,2],\
        emptyOnes)), 3)
    D01 = np.rot90(np.dstack((coord2[:,0],coord2[:,1],coord2[:,2],\
        emptyOnes)), 3)
    D02 = np.rot90(np.dstack((coord3[:,0],coord3[:,1],coord3[:,2],\
        emptyOnes)), 3)
    D03 = np.rot90(np.dstack((coord4[:,0],coord4[:,1],coord4[:,2],\
        emptyOnes)), 3)

    D0 = np.linalg.det(np.hstack((D00,D01,D02,D03)))

    for i in range(8):

        D10 = np.rot90(np.dstack((emptyOnes*node[i,0],\
            emptyOnes*node[i,1],emptyOnes*node[i,2],emptyOnes)), 3)
        
        D1 = np.linalg.det(np.hstack((D10,D01,D02,D03)))
        D2 = np.linalg.det(np.hstack((D00,D10,D02,D03)))
        D3 = np.linalg.det(np.hstack((D00,D01,D10,D03)))
        D4 = np.linalg.det(np.hstack((D00,D01,D02,D10)))

        if np.logical_and(np.logical_and(np.sign(D0) == np.sign(D1),\
            np.sign(D0) == np.sign(D2)),\
            np.logical_and(np.sign(D0) == np.sign(D3),\
            np.sign(D0) == np.sign(D4))).any():
            inside = inside+1
        else:
            pass

        if inside <= i:
            return False
            break

        if inside == 8:
            return True
            break