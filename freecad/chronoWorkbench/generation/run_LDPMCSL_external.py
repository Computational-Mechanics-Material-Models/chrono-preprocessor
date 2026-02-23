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

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from freecad.chronoWorkbench.generation.check_multiMat_matVol import check_multiMat_matVol
from freecad.chronoWorkbench.generation.gen_CSL_facetData import gen_CSL_facetData
from freecad.chronoWorkbench.generation.gen_LDPM_facetData import gen_LDPM_facetData
from freecad.chronoWorkbench.generation.gen_LDPMCSL_facetfiberInt import gen_LDPMCSL_facetfiberInt
from freecad.chronoWorkbench.generation.gen_LDPMCSL_fibers import gen_LDPMCSL_fibers
from freecad.chronoWorkbench.generation.gen_LDPMCSL_flowEdges import gen_LDPMCSL_flowEdges
from freecad.chronoWorkbench.generation.gen_LDPMCSL_multiStep import gen_LDPMCSL_multiStep
from freecad.chronoWorkbench.generation.gen_LDPMCSL_tesselation import gen_LDPMCSL_tesselation
from freecad.chronoWorkbench.generation.gen_LDPMCSL_tetrahedralization import gen_LDPMCSL_tetrahedralization
from freecad.chronoWorkbench.generation.gen_multiMat_assign import gen_multiMat_assign
from freecad.chronoWorkbench.generation.gen_multiMat_refine import gen_multiMat_refine
from freecad.chronoWorkbench.generation.gen_multiMat_reform import gen_multiMat_reform
from freecad.chronoWorkbench.generation.sort_multiMat_mat import sort_multiMat_mat
from freecad.chronoWorkbench.generation.sort_multiMat_voxels import sort_multiMat_voxels

from freecad.chronoWorkbench.input.read_ctScan_file import read_ctScan_file
from freecad.chronoWorkbench.input.read_LDPMCSL_tetgen import read_LDPMCSL_tetgen
from freecad.chronoWorkbench.input.read_multiMat_file import read_multiMat_file

from freecad.chronoWorkbench.output.mkData_LDPMCSL_edges import mkData_LDPMCSL_edges
from freecad.chronoWorkbench.output.mkData_LDPMCSL_faceFacets import mkData_LDPMCSL_faceFacets
from freecad.chronoWorkbench.output.mkData_LDPMCSL_facetfiberInt import mkData_LDPMCSL_facetfiberInt
from freecad.chronoWorkbench.output.mkData_LDPMCSL_facets import mkData_LDPMCSL_facets
from freecad.chronoWorkbench.output.mkData_LDPMCSL_facetsVertices import mkData_LDPMCSL_facetsVertices
from freecad.chronoWorkbench.output.mkData_LDPMCSL_flowEdges import mkData_LDPMCSL_flowEdges
from freecad.chronoWorkbench.output.mkData_LDPMCSL_tets import mkData_LDPMCSL_tets
from freecad.chronoWorkbench.output.mkData_nodes import mkData_nodes
from freecad.chronoWorkbench.output.mkData_particles import mkData_particles
from freecad.chronoWorkbench.output.mkIges_LDPMCSL_flowEdges import mkIges_LDPMCSL_flowEdges
from freecad.chronoWorkbench.output.mkPy_LDPM_singleParaview import mkPy_LDPM_singleParaview
from freecad.chronoWorkbench.output.mkPy_LDPM_singleParaviewLabels import mkPy_LDPM_singleParaviewLabels
from freecad.chronoWorkbench.output.mkVtk_LDPM_singleCell import mkVtk_LDPM_singleCell
from freecad.chronoWorkbench.output.mkVtk_LDPM_singleEdge import mkVtk_LDPM_singleEdge
from freecad.chronoWorkbench.output.mkVtk_LDPM_singleEdgeFacets import mkVtk_LDPM_singleEdgeFacets
from freecad.chronoWorkbench.output.mkVtk_LDPM_singleEdgeParticles import mkVtk_LDPM_singleEdgeParticles
from freecad.chronoWorkbench.output.mkVtk_LDPM_singleTet import mkVtk_LDPM_singleTet
from freecad.chronoWorkbench.output.mkVtk_LDPM_singleTetFacets import mkVtk_LDPM_singleTetFacets
from freecad.chronoWorkbench.output.mkVtk_LDPM_singleTetParticles import mkVtk_LDPM_singleTetParticles
from freecad.chronoWorkbench.output.mkVtk_LDPMCSL_facets import mkVtk_LDPMCSL_facets
from freecad.chronoWorkbench.output.mkVtk_LDPMCSL_fibers import mkVtk_LDPMCSL_fibers
from freecad.chronoWorkbench.output.mkVtk_LDPMCSL_flowEdges import mkVtk_LDPMCSL_flowEdges
from freecad.chronoWorkbench.output.mkVtk_LDPMCSL_nonIntFibers import mkVtk_LDPMCSL_nonIntFibers
from freecad.chronoWorkbench.output.mkVtk_LDPMCSL_projFacets import mkVtk_LDPMCSL_projFacets
from freecad.chronoWorkbench.output.mkVtk_particles import mkVtk_particles


def _toggle_on(value):
    return str(value).strip().lower() in ["on", "y", "yes", "true", "1"]


def _temp_path(bundle_dir):
    bundle = str(bundle_dir).replace("\\", "/")
    if not bundle.endswith("/"):
        bundle = bundle + "/"
    return bundle


def run_external_bundle(bundle_dir):

    bundle_dir = Path(bundle_dir).resolve()
    manifest_path = bundle_dir / "external_manifest_ldpmcsl.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tempPath = _temp_path(bundle_dir)
    geoName = cfg["geoName"]
    elementType = cfg["elementType"]

    print("Starting external generation for:", geoName)

    meshVertices = np.load(tempPath + "meshVertices.npy")
    meshTets = np.load(tempPath + "meshTets.npy")
    surfaceNodes = np.load(tempPath + "surfaceNodes.npy")
    surfaceFaces = np.load(tempPath + "surfaceFaces.npy")
    coord1 = np.load(tempPath + "coord1.npy")
    coord2 = np.load(tempPath + "coord2.npy")
    coord3 = np.load(tempPath + "coord3.npy")
    coord4 = np.load(tempPath + "coord4.npy")

    start_time = time.time()

    gen_LDPMCSL_multiStep(
        tempPath,
        cfg["numCPU"],
        cfg["numIncrements"],
        cfg["maxIter"],
        cfg["parOffset"],
        cfg["maxEdgeLength"],
        cfg["max_dist"],
        cfg["minPar_sim"],
        cfg["maxPar_sim"],
        cfg["minPar_exp"],
        cfg["maxPar_exp"],
        cfg["sieveCurveDiameter"],
        cfg["sieveCurvePassing"],
        cfg["wcRatio"],
        cfg["cementC"],
        cfg["airFrac"],
        cfg["fullerCoef"],
        cfg["flyashC"],
        cfg["silicaC"],
        cfg["scmC"],
        cfg["fillerC"],
        cfg["flyashDensity"],
        cfg["silicaDensity"],
        cfg["scmDensity"],
        cfg["fillerDensity"],
        cfg["cementDensity"],
        cfg["densityWater"],
        cfg["multiMatToggle"],
        cfg["aggFile"],
        cfg["multiMatFile"],
        cfg["grainAggMin"],
        cfg["grainAggMax"],
        cfg["grainAggFuller"],
        cfg["grainAggSieveD"],
        cfg["grainAggSieveP"],
        cfg["grainBinderMin"],
        cfg["grainBinderMax"],
        cfg["grainBinderFuller"],
        cfg["grainBinderSieveD"],
        cfg["grainBinderSieveP"],
        cfg["grainITZMin"],
        cfg["grainITZMax"],
        cfg["grainITZFuller"],
        cfg["grainITZSieveD"],
        cfg["grainITZSieveP"],
        cfg["tetVolume"],
        cfg["minC"],
        cfg["maxC"],
        cfg["verbose"],
    )

    internalNodes = np.load(tempPath + "internalNodes.npy")
    materialList = np.load(tempPath + "materialList.npy")
    parDiameterList = np.load(tempPath + "parDiameterList.npy")
    particleID = np.load(tempPath + "particleID.npy")

    if _toggle_on(cfg["fiberToggle"]):
        if cfg["fiberFile"] not in [0, None, [], ""] and Path(str(cfg["fiberFile"])).is_file():
            CTScanFiberData = read_ctScan_file(cfg["fiberFile"])
            CTScanFiberData = np.array(CTScanFiberData).reshape(-1, 10)
            p1Fibers = CTScanFiberData[:, 0:3]
            p2Fibers = CTScanFiberData[:, 3:6]
            orienFibers = CTScanFiberData[:, 6:9]
            fiberLengths = CTScanFiberData[:, 9:10]
        else:
            nFiber = int(
                round(
                    4
                    * cfg["tetVolume"]
                    * cfg["fiberVol"]
                    / (math.pi * cfg["fiberDiameter"] ** 2 * cfg["fiberLength"])
                )
            )
            maxC = np.asarray(cfg["maxC"])
            p1Fibers = (np.zeros((nFiber, 3)) + 2) * maxC
            p2Fibers = (np.zeros((nFiber, 3)) + 2) * maxC
            orienFibers = (np.zeros((nFiber, 3)) + 2) * maxC
            fiberLengths = np.zeros((nFiber, 1))

            for x in range(0, nFiber):
                if x % 100 == 0:
                    print(str(nFiber - x) + " fibers remaining")

                p1Fiber, p2Fiber, orienFiber, lFiber = gen_LDPMCSL_fibers(
                    meshVertices,
                    meshTets,
                    coord1,
                    coord2,
                    coord3,
                    coord4,
                    cfg["maxIter"],
                    cfg["fiberLength"],
                    np.array(cfg["maxC"]),
                    cfg["maxPar_sim"],
                    np.array(
                        [
                            cfg["fiberOrientation1"],
                            cfg["fiberOrientation2"],
                            cfg["fiberOrientation3"],
                        ]
                    ),
                    cfg["fiberPref"],
                    surfaceFaces,
                    cfg["fiberCutting"],
                )
                p1Fibers[x, :] = p1Fiber
                p2Fibers[x, :] = p2Fiber
                orienFibers[x, :] = orienFiber
                fiberLengths[x, :] = lFiber
    else:
        p1Fibers = np.zeros((0, 3))
        p2Fibers = np.zeros((0, 3))
        orienFibers = np.zeros((0, 3))
        fiberLengths = np.zeros((0, 1))

    print("Forming tetrahedralization.")
    gen_LDPMCSL_tetrahedralization(internalNodes, surfaceNodes, surfaceFaces, geoName, tempPath)
    allNodes, allTets, allEdges = read_LDPMCSL_tetgen(
        Path(tempPath + geoName + ".node"),
        Path(tempPath + geoName + ".ele"),
        Path(tempPath + geoName + ".edge"),
    )

    print("Forming tesselation.")
    (
        tetFacets,
        facetCenters,
        facetAreas,
        facetNormals,
        tetn1,
        tetn2,
        tetPoints,
        allDiameters,
        facetPointData,
        facetCellData,
    ) = gen_LDPMCSL_tesselation(allNodes, allTets, parDiameterList, cfg["minPar_sim"], geoName)

    if _toggle_on(cfg["htcToggle"]):
        edgeData = gen_LDPMCSL_flowEdges(
            cfg["htcLength"],
            allNodes,
            allTets,
            tetPoints,
            cfg["maxPar_sim"],
            meshVertices,
            meshTets,
            coord1,
            coord2,
            coord3,
            coord4,
            np.array(cfg["maxC"]),
        )
    else:
        edgeData = 0

    edgeMaterialList = 0
    cementStructure = "Off"

    if _toggle_on(cfg["multiMatToggle"]):
        multiMatX, multiMatY, multiMatZ, multiMatRes, multiMatVoxels = read_multiMat_file(cfg["multiMatFile"])
        aggVoxels, itzVoxels, binderVoxels, aggVoxelIDs = sort_multiMat_voxels(multiMatVoxels)

    particleID = np.concatenate((0 * np.ones([len(allNodes) - len(particleID),]), particleID))

    if _toggle_on(cfg["multiMatToggle"]):
        materialList = np.concatenate((2 * np.ones([len(allNodes) - len(materialList),]), materialList))
        materialList = gen_multiMat_assign(
            allNodes,
            materialList,
            aggVoxels,
            itzVoxels,
            binderVoxels,
            internalNodes,
            multiMatX,
            multiMatY,
            multiMatZ,
            multiMatRes,
            np.array(cfg["minC"]),
        )
    else:
        materialList = np.concatenate((0 * np.ones([len(allNodes) - len(materialList),]), materialList))

    print("Generating facet data.")
    if elementType == "LDPM":
        facetData, facetMaterial, subtetVol, facetVol1, facetVol2, particleMaterial = gen_LDPM_facetData(
            allNodes,
            allTets,
            tetFacets,
            facetCenters,
            facetAreas,
            facetNormals,
            tetn1,
            tetn2,
            materialList,
            cfg["multiMatRule"],
            cfg["multiMatToggle"],
            cementStructure,
            edgeMaterialList,
            facetCellData,
            particleID,
        )
    else:
        facetData, facetMaterial, subtetVol, facetVol1, facetVol2, particleMaterial = gen_CSL_facetData(
            allNodes,
            allEdges,
            allTets,
            tetFacets,
            facetCenters,
            facetAreas,
            facetNormals,
            tetn1,
            tetn2,
            materialList,
            cfg["multiMatRule"],
            cfg["multiMatToggle"],
            cementStructure,
            edgeMaterialList,
            facetCellData,
            particleID,
        )

    FiberdataList = None
    TotalIntersections = None
    MaxInterPerFacet = None
    IntersectedFiber = None
    projectedFacet = None
    if _toggle_on(cfg["fiberToggle"]) and _toggle_on(cfg["fiberIntersections"]):
        (
            FiberdataList,
            TotalIntersections,
            MaxInterPerFacet,
            TotalTet,
            TotalFiber,
            IntersectedFiber,
            projectedFacet,
        ) = gen_LDPMCSL_facetfiberInt(
            p1Fibers,
            p2Fibers,
            cfg["fiberDiameter"],
            fiberLengths,
            orienFibers,
            geoName,
            allTets,
            allNodes,
            tetFacets,
            facetData,
            tetn1,
            tetn2,
            facetNormals,
            facetCenters,
        )

    if _toggle_on(cfg["multiMatToggle"]):
        try:
            multiMatX, multiMatY, multiMatZ, multiMatRes, aggDistinctVoxels = read_multiMat_file(cfg["aggFile"])
            aggVoxels, _, _, aggVoxelIDs = sort_multiMat_voxels(aggDistinctVoxels)
        except Exception:
            pass

        itzVolFracSim, binderVolFracSim, aggVolFracSim, itzVolFracAct, binderVolFracAct, aggVolFracAct = check_multiMat_matVol(
            subtetVol, facetMaterial, aggVoxels, itzVoxels, binderVoxels
        )

        if cfg["multiMatRule"] > 9:
            sortedData1 = sort_multiMat_mat(facetMaterial, facetVol1, facetVol2, particleMaterial, subtetVol)
            i = 0
            while (
                (abs(itzVolFracSim - itzVolFracAct) > 0.02 or abs(itzVolFracSim - itzVolFracAct) == 0.00)
                and abs(binderVolFracSim - binderVolFracAct) > 0.02
                and abs(aggVolFracSim - aggVolFracAct) > 0.02
                and i < len(sortedData1)
            ):
                if sortedData1[i, 3] != sortedData1[i, 4]:
                    sortedData = gen_multiMat_refine(
                        sortedData1,
                        itzVolFracSim,
                        binderVolFracSim,
                        aggVolFracSim,
                        itzVolFracAct,
                        binderVolFracAct,
                        aggVolFracAct,
                        i,
                    )

                    (
                        itzVolFracSim,
                        binderVolFracSim,
                        aggVolFracSim,
                        itzVolFracAct,
                        binderVolFracAct,
                        aggVolFracAct,
                    ) = check_multiMat_matVol(sortedData[:, 5], sortedData[:, 2], aggVoxels, itzVoxels, binderVoxels)
                    sortedData1 = sortedData
                i = i + 1

            facetData, facetMaterial = gen_multiMat_reform(allTets, facetData, sortedData1)

    allDiameters = np.concatenate((np.array([0.0] * int(len(allNodes) - len(parDiameterList))), parDiameterList))

    if cfg["dataFilesGen"]:
        print("Writing data files.")
        mkData_nodes(geoName, tempPath, allNodes)
        mkData_LDPMCSL_tets(geoName, tempPath, allTets)
        if elementType == "CSL":
            mkData_LDPMCSL_edges(geoName, tempPath, allEdges)
        mkData_LDPMCSL_facets(geoName, tempPath, facetData)
        mkData_LDPMCSL_facetsVertices(geoName, tempPath, tetFacets)
        mkData_LDPMCSL_faceFacets(geoName, tempPath, surfaceNodes, surfaceFaces)
        if FiberdataList is not None:
            mkData_LDPMCSL_facetfiberInt(geoName, FiberdataList, TotalIntersections, MaxInterPerFacet, tempPath)
        mkData_particles(allNodes, allDiameters, geoName, tempPath)
        if _toggle_on(cfg["htcToggle"]):
            mkData_LDPMCSL_flowEdges(geoName, edgeData, tempPath)

    if cfg["visFilesGen"]:
        print("Writing visualization files.")
        mkVtk_particles(
            internalNodes,
            parDiameterList,
            materialList[(len(allNodes) - len(internalNodes)) : len(allNodes)],
            geoName,
            tempPath,
        )
        mkVtk_LDPMCSL_facets(geoName, tempPath, tetFacets, facetMaterial)
        if _toggle_on(cfg["htcToggle"]):
            mkVtk_LDPMCSL_flowEdges(geoName, edgeData, tempPath)
            mkIges_LDPMCSL_flowEdges(geoName, edgeData, tempPath)
        if _toggle_on(cfg["fiberToggle"]):
            mkVtk_LDPMCSL_fibers(
                p1Fibers,
                p2Fibers,
                cfg["fiberDiameter"],
                fiberLengths,
                orienFibers,
                geoName,
                tempPath,
            )
            if projectedFacet is not None:
                mkVtk_LDPMCSL_projFacets(geoName, projectedFacet, tempPath)
        if IntersectedFiber is not None:
            mkVtk_LDPMCSL_nonIntFibers(
                p1Fibers,
                p2Fibers,
                cfg["fiberDiameter"],
                fiberLengths,
                orienFibers,
                geoName,
                IntersectedFiber,
                tempPath,
            )

    if cfg["singleTetGen"]:
        if elementType == "LDPM":
            mkVtk_LDPM_singleTetFacets(geoName, tempPath, tetFacets)
            mkVtk_LDPM_singleTetParticles(allNodes, allTets, allDiameters, geoName, tempPath)
            mkVtk_LDPM_singleTet(allNodes, allTets, geoName, tempPath)
            mkVtk_LDPM_singleCell(allNodes, allTets, parDiameterList, tetFacets, geoName, tempPath)
            mkPy_LDPM_singleParaview(geoName, cfg["outDir"], cfg["outName"], tempPath)
            mkPy_LDPM_singleParaviewLabels(geoName, tempPath)
        elif elementType == "CSL":
            mkVtk_LDPM_singleEdgeFacets(geoName, tempPath, allEdges, facetData, tetFacets)
            mkVtk_LDPM_singleEdgeParticles(allNodes, allEdges, allDiameters, geoName, tempPath)
            mkVtk_LDPM_singleEdge(allNodes, allEdges, geoName, tempPath)

    try:
        os.rename(Path(tempPath + geoName + "-para-mesh.vtk"), Path(tempPath + geoName + "-para-mesh.000.vtk"))
    except Exception:
        pass

    for fname in [geoName + "2D.mesh", geoName + ".node", geoName + ".ele", geoName + ".edge"]:
        try:
            os.remove(Path(tempPath + fname))
        except Exception:
            pass

    total_time = round(time.time() - start_time, 2)
    print("External generation completed in " + str(total_time) + " seconds")
    print("Generated files written to: " + str(bundle_dir))
