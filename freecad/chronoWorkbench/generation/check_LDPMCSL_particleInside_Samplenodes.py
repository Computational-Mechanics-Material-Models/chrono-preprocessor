import numpy as np

def check_LDPMCSL_particleInside_Samplenodes(vertices,tets,center,parDiameter,binMin,binMax,coord1,\
    coord2,coord3,coord4,nSamples=26,tol=1e-12):

    """
    This function checks whether a particle is inside the mesh by testing multiple points on the particle surface.
    If you still have issues with particles being placed outside the mesh, using function check_LDPMCSL_particleInside.py.
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
    - nSamples:         Number of sampling directions on the particle surface
                        (supported: 6, 14, 26)
    - tol:              Tolerance for point-in-tet check
    --------------------------------------------------------------------------
    ### Outputs ###
    - Boolean:          True if the particle is inside tets, False if not
    --------------------------------------------------------------------------
    """

    # Convert center to a 1D array
    center = center.flatten()

    # Store tet vertices that fall inside the bin
    binCoord1 = np.all([(coord1[:,0] > binMin[0]) , (coord1[:,0] < binMax[0]),\
                        (coord1[:,1] > binMin[1]) , (coord1[:,1] < binMax[1]),\
                        (coord1[:,2] > binMin[2]) , (coord1[:,2] < binMax[2])],axis=0)

    binCoord2 = np.all([(coord2[:,0] > binMin[0]) , (coord2[:,0] < binMax[0]),\
                        (coord2[:,1] > binMin[1]) , (coord2[:,1] < binMax[1]),\
                        (coord2[:,2] > binMin[2]) , (coord2[:,2] < binMax[2])],axis=0)

    binCoord3 = np.all([(coord3[:,0] > binMin[0]) , (coord3[:,0] < binMax[0]),\
                        (coord3[:,1] > binMin[1]) , (coord3[:,1] < binMax[1]),\
                        (coord3[:,2] > binMin[2]) , (coord3[:,2] < binMax[2])],axis=0)

    binCoord4 = np.all([(coord4[:,0] > binMin[0]) , (coord4[:,0] < binMax[0]),\
                        (coord4[:,1] > binMin[1]) , (coord4[:,1] < binMax[1]),\
                        (coord4[:,2] > binMin[2]) , (coord4[:,2] < binMax[2])],axis=0)

    # Get tets with a vertex that falls inside the bin
    binTets = np.any([binCoord1,binCoord2,binCoord3,binCoord4],axis=0)

    # If no tets are found in the bin, particle cannot be inside
    if not np.any(binTets):
        return False

    # Store vertices of those tets
    tetIDs = tets.astype(int)[binTets,:] - 1
    coord1 = vertices[tetIDs[:,0]]
    coord2 = vertices[tetIDs[:,1]]
    coord3 = vertices[tetIDs[:,2]]
    coord4 = vertices[tetIDs[:,3]]

    # Generate sampling directions on the particle surface
    directions = []

    # 6 axis directions
    directions.extend([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1]
    ])

    # Add 8 body diagonal directions for 14 or 26 samples
    if nSamples >= 14:
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    directions.append([sx, sy, sz])

    # Add 12 edge-midpoint directions for 26 samples
    if nSamples >= 26:
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                directions.append([sx, sy, 0])
                directions.append([sx, 0, sy])
                directions.append([0, sx, sy])

    directions = np.array(directions, dtype=float)

    # Normalize directions to unit vectors
    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    # Generate test points: center + sampled surface points
    node = np.empty((len(directions) + 1, 3))
    node[0,:] = center

    for i in range(len(directions)):
        node[i+1,:] = center + parDiameter / 2 * directions[i,:]

    # Precompute matrices for point-in-tet check
    A = coord1
    B = coord2
    C = coord3
    D = coord4

    AB = B - A
    AC = C - A
    AD = D - A

    # Build 3x3 matrix for each tet
    M = np.stack((AB, AC, AD), axis=2)

    # Remove degenerate tets
    detM = np.linalg.det(M)
    validTets = np.abs(detM) > tol

    if not np.any(validTets):
        return False

    A = A[validTets]
    M = M[validTets]

    # Invert matrices
    try:
        MInv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return False

    # Check whether all sample points fall inside at least one tet
    inside = 0

    for i in range(node.shape[0]):

        # Compute barycentric coordinates for all candidate tets
        rhs = node[i,:] - A
        bary = np.einsum('nij,nj->ni', MInv, rhs)

        l1 = bary[:,0]
        l2 = bary[:,1]
        l3 = bary[:,2]
        l0 = 1.0 - l1 - l2 - l3

        # Check if point lies inside any tet
        pointInside = np.logical_and.reduce([
            l0 >= -tol, l1 >= -tol, l2 >= -tol, l3 >= -tol,
            l0 <= 1.0 + tol, l1 <= 1.0 + tol,
            l2 <= 1.0 + tol, l3 <= 1.0 + tol
        ]).any()

        if pointInside:
            inside = inside + 1
        else:
            pass

        if inside <= i:
            return False
            break

        if inside == node.shape[0]:
            return True
            break