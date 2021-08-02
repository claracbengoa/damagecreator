import salome
import  SMESH
from salome.geom import geomBuilder
from salome.smesh import smeshBuilder

import sys
import math
import numpy as np
from numpy.linalg import norm
from numpy.random import uniform
from pathlib import Path
from auxiliaryFunctions import clusteringAlgorithm
from auxiliaryFunctions import getTranslationalRiskAngleRefAxis 

from itertools import product

import os

salome.salome_init()
geompy = geomBuilder.New()
smesh = smeshBuilder.New()

def smallestLineOnFace(face):

    bndVertices_Slm = geompy.ExtractShapes(face, geompy.ShapeType["VERTEX"], True)
    indexList = [(0,1), (0,2), (0,3)]
    distances = [geompy.MinDistance(bndVertices_Slm[i], bndVertices_Slm[j]) for i,j in indexList] 
    index = distances.index(min(distances))
    p1 = bndVertices_Slm[indexList[index][0]]
    p2 = bndVertices_Slm[indexList[index][1]]
    line = geompy.MakeLineTwoPnt(p1,p2)

    return line

class Line:

    def __init__(self, Point1, Point2):
        self.origin = Point1
        self.dest = Point2
        v1 = geompy.MakeVertex(*list(Point1))
        v2 = geompy.MakeVertex(*list(Point2))

        self.geom = geompy.MakeLineTwoPnt(v1, v2)

    def addToStudy(self, name = 'Line'):
        geompy.addToStudy(self.geom, name)

    def extendLine(self, multiplier):
        self.geom = geompy.ExtendEdge(self.geom, 0, multiplier)
        # Obtain the Salome vertexes: New Entity-Explode-SubShapes Selection
        [v1, v2] = geompy.ExtractShapes(self.geom, geompy.ShapeType["VERTEX"], True)
        v1coords = geompy.PointCoordinates(v1)
        v2coords = geompy.PointCoordinates(v2)

        self.dest = v2coords if np.allclose(v1coords, self.origin) else v1coords

    def translateOrigin(self, newOrigin):
        '''
        Dada una linea definida por su origen y destino, la traslada paralelamente
        a un nuevo origen
        '''
        vector = np.array(newOrigin) - np.array(self.origin)
        point1 = newOrigin
        point2 = np.array(self.dest) + vector
        translatedLine = Line(point1, point2)  # objeto linea en la nueva posicion

        return translatedLine

    def translateLineToCoords(self, coordsList):

        translatedLines = [self.translateOrigin(coords) for coords in coordsList]

        return translatedLines

    def intersectsWith(self, femesh):

        size = femesh.getSize()
        center = femesh.getCenter()
        multiplier = norm(center - self.origin) + size

        self.extendLine(multiplier)

        elementSize = femesh.getMinElementSize()
        #tolerance = np.sqrt(2)*elementSize*1.1  # diagonal*factor
        tolerance = elementSize*0.1  # diagonal*factor

        smeshType = SMESH.FACE 
        smeshMethod  = SMESH.FT_LyingOnGeom

        aCriteria = [smesh.GetCriterion(smeshType, smeshMethod,
                                        SMESH.FT_Undefined, self.geom,
                                        SMESH.FT_Undefined, SMESH.FT_LogicalOR,
                                        tolerance),]
        aFilter = smesh.GetFilterFromCriteria(aCriteria)
        aFilter.SetMesh(femesh.mesh.GetMesh())
        holesMesh = femesh.mesh.GroupOnFilter(smeshType, 'tangentGroup', aFilter)

        return not holesMesh.IsEmpty()

    def getAngleWithVector(self, vector):

        lineVector = np.array(self.dest - self.origin)

        angle = np.degrees(np.arccos(lineVector.dot(vector)/(norm(lineVector)*norm(vector))))

        return angle


class ScatteringPrism:

    def __init__(self, prismParameters, translationalRiskAngle):

        self.origin = prismParameters['origin']
        self.fwdAngle = prismParameters['fwdAngle']
        self.aftAngle = prismParameters['aftAngle']
        self.fwdAngleR = np.radians(self.fwdAngle)
        self.aftAngleR = np.radians(self.aftAngle)
        self.orientation3D = prismParameters['orientation3D'] 
        self.rotatingPlane = prismParameters['rotatingPlane'] 

        assert (self.rotatingPlane in ('XY', 'XZ', 'YZ')), 'rotatingPlane must be XY, XZ or YZ' 

        self.cobM = self._COBMatrix() 

        # local axis z is always the reference axis for the translational risk angle
        zlocal =  self.cobM[:,2]
        # Check if local axis z has the same sign compared to refAxisCode
        refAxisCode = getTranslationalRiskAngleRefAxis(self.orientation3D, self.rotatingPlane)

        if refAxisCode == 'X':
            axis = np.array([1.0, 0.0, 0.0])
        elif refAxisCode == 'Y':
            axis = np.array([0.0, 1.0, 0.0])
        elif refAxisCode == 'Z':
            axis = np.array([0.0, 0.0, 1.0])

        # different sign implies 180 degrees
        self.translationalRiskAngle_lb = translationalRiskAngle[0]
        self.translationalRiskAngle_ub = translationalRiskAngle[1]

        #print('axis=',axis)
        #print('zlocal=',zlocal)

        if axis.dot(zlocal) < 0:
            self.translationalRiskAngle_lb *= -1
            self.translationalRiskAngle_ub *= -1

    def generateDebris(self, nImpactLines, shapes, velocities):
        '''
        Dado un prisma definido por unos angulos y orientado de una determinada forma
        Dado el numero posible de impactos que se pueden producir
        Genera las lineas de impacto dentro del prisma
        Genera geometria del debris con sus propiedades de forma, velocidad y su linea asociada
        '''
        assert nImpactLines == len(shapes) == len(velocities), 'arrays lenght must be equal to nImpactLines'

        lines = self.getRandomLines(nImpactLines)

        debrisList = [Debris(line, shape, velocity) for line, shape, velocity in zip(lines, shapes, velocities)]

        return debrisList

    def getRandomLines(self, numLines):

        betas = uniform(self.aftAngle, self.fwdAngle, numLines)
        thetas = uniform(self.translationalRiskAngle_lb, self.translationalRiskAngle_ub, numLines)

        lines = [self._getLineInsidePrism(beta, theta) for beta, theta in zip(betas, thetas)]

        return lines

    def _getLineInsidePrism(self, beta, theta):
        # beta, theta in degrees
        pointPrism = self._getPointInsidePrism_global(beta, theta)
        line = Line(self.origin, pointPrism)

        return line

    def _getPointInsidePrism_global(self, beta, theta):
        '''
        globalVector es el vector interior al prisma en coordenadas globales
        libre y necesito las coordenadas del punto final del vector respecto al
        eje global.
        PTO (vertice del prisma en stma global) +
        VECTOR interior al prisma en coordenadas globales =
        PTO interior al prisma en stma global

        Parameters:
        beta: angle inside spread risk angle (degrees)
        theta: angle inside translational risk angle (degrees)
        '''
        localVector = self._getPointInsidePrism_local(beta, theta)
        globalVector = self.cobM.dot(localVector)
        pointInsidePrismGlobal = self.origin + globalVector

        return pointInsidePrismGlobal

    def _getPointInsidePrism_local(self, beta, theta):

        betaR = np.radians(beta)
        thetaR = np.radians(theta)

        h = 1

        x = h
        y = h*np.tan(betaR)
        z = h*np.tan(thetaR)

        if abs(theta) > 90: # Change sign of x and z in 2nd and 3rd quadrant
            x*= -1
            z*= -1 

        return [x, y, z]

    def _COBMatrix(self):

        x = self.orientation3D

        if self.rotatingPlane == 'XY':
            y = [0.0, 0.0, 1.0]
        elif self.rotatingPlane == 'XZ':
            y = [0.0, 1.0, 0.0]
        elif self.rotatingPlane == 'YZ':
            y = [1.0, 0.0, 0.0]

        z = np.cross(x, y)

        x, y, z = [v/norm(v) for v in (x, y, z)]

        cobM = np.column_stack([x, y, z])

        return cobM


class Debris:
    '''
    Define a piece of debris with shape and velocity properties and the
    associated line inside the cone
    '''
    def __init__(self, line, shape, velocity):

        self.line = line
        self.shape = shape
        self.velocity = velocity

        self.origin = line.origin

        self._getDebrisGeom()

    def _getDebrisGeom(self):

        angleRoll = float(uniform(0.0, 180.0, 1))
        anglePitch = float(uniform(-45.0, 45.0, 1))

        debris0 = geompy.MakeFaceObjHW(self.line.geom, self.shape['a'], self.shape['b'])

        debris1 = geompy.Rotate(debris0, self.line.geom, angleRoll*np.pi/180.0, theCopy=True)

        line = smallestLineOnFace(debris1)
        middlePoints_Slm = geompy.MakeVertexOnCurve(line, 0.5, True)
        origin_Slm = geompy.MakeVertex(*self.origin) 
        axis = geompy.MakeTranslationTwoPoints(line,middlePoints_Slm, origin_Slm)

        debris2 = geompy.Rotate(debris1, axis, anglePitch*np.pi/180.0, theCopy=True)
        #geompy.addToStudy(debris0, 'debris0')
        #geompy.addToStudy(debris1, 'debris1')
        geompy.addToStudy(debris2, 'debris2')

        self.geom = debris2


    def generateDebrisMesh(self, elementSize):

        self.mesh = smesh.Mesh(self.geom)
        Regular_1D = self.mesh.Segment()
        size = Regular_1D.LocalLength(elementSize, None, 1e-07)
        Quadrangle_2D = self.mesh.Quadrangle(algo=smeshBuilder.QUADRANGLE)
        isDone = self.mesh.Compute()

    def getNodeCoordinates(self):

        nodesId = self.mesh.GetNodesId()
        debrisNodesCoords = [self.mesh.GetNodeXYZ(id) for id in nodesId]

        return debrisNodesCoords


class FEMesh:

    def __init__(self, NameFile):

        self.NameFile = NameFile
        medFile = NameFile + '.med'
        path = Path.cwd() / medFile
        assert path.exists(), '%s does not exists' % str(path)

        ([self.mesh], status) = smesh.CreateMeshesFromMED(str(path))
        assert status == SMESH.DRS_OK, 'Invalid Mesh'

    def getnElements(self):

        return self.mesh.NbElements()

    def getElementsId(self):

        return self.mesh.GetElementsId()

    def getElementsCoG(self, elements):

        elementsCoG = [self.mesh.BaryCenter(element) for element in elements]

        return np.array(elementsCoG)

    def _getBoundingBox(self):

        box = np.array(self.mesh.BoundingBox())
        minvalues = box[:3]  # hasta el 3
        maxvalues = box[3:]  # del 3 hacia delante no incluido

        return minvalues, maxvalues

    def getSize(self):

        minvalues, maxvalues = self._getBoundingBox()
        size = norm(maxvalues - minvalues)

        return size

    def getCenter(self):

        minvalues, maxvalues = self._getBoundingBox()
        center = (maxvalues + minvalues)/2

        return center

    def getTranslationalRiskAngle(self, origin, orientation3D, rotatingPlane):

        boundVertices = self._getBoundVertices(origin, orientation3D, rotatingPlane)

        origin = np.array(origin)

        p0 = np.array(boundVertices['bnd_1_near'])
        p1 = np.array(boundVertices['bnd_1_far'])
        tangentLine_1, tangent_point_1 = self._getTangentToMesh(origin,p0,p1)
        angle_1 = tangentLine_1.getAngleWithVector(orientation3D)

        p0 = np.array(boundVertices['bnd_2_near'])
        p1 = np.array(boundVertices['bnd_2_far'])
        tangentLine_2, tangent_point_2 = self._getTangentToMesh(origin,p0,p1)
        angle_2 = tangentLine_2.getAngleWithVector(orientation3D)

        tangentLine_1.addToStudy('tangentLine_1')
        tangentLine_2.addToStudy('tangentLine_2')

        tangent_point_1 = np.array(tangent_point_1)
        tangent_point_2 = np.array(tangent_point_2)

        refAxisCode = getTranslationalRiskAngleRefAxis(orientation3D, rotatingPlane)  

        axisDict = {'X': 0, 'Y': 1, 'Z': 2}

        comp = axisDict[refAxisCode]
        if tangent_point_1[comp] < origin[comp]: angle_1 = - angle_1
        if tangent_point_2[comp] < origin[comp]: angle_2 = - angle_2

        return angle_1, angle_2

    def _getBoundVertices(self, origin, orientation3D, rotatingPlane):

        if rotatingPlane == 'XY':
            nVRotatingPlane = [0.0, 0.0, 1.0]
        elif rotatingPlane == 'XZ':
            nVRotatingPlane = [0.0, 1.0, 0.0]
        elif rotatingPlane == 'YZ':
            nVRotatingPlane = [1.0, 0.0, 0.0]

        nVRotatingPlane_Slm = geompy.MakeVectorDXDYDZ(*nVRotatingPlane)

        # normal vector to bound faces of translational risk angle
        nVBoundFaces = np.cross(orientation3D, nVRotatingPlane)
        nVBoundFaces_Slm = geompy.MakeVectorDXDYDZ(*nVBoundFaces)

        #minimum and maximum values of the bounding box
        minvalues, maxvalues = self._getBoundingBox()
        vertex_1_Slm = geompy.MakeVertex(*minvalues) # each component to each argument
        vertex_2_Slm = geompy.MakeVertex(*maxvalues)

        # planes that contain bound faces
        bndPlane_1_Slm = geompy.MakePlane(vertex_1_Slm, nVBoundFaces_Slm, 2*self.getSize())
        bndPlane_2_Slm = geompy.MakePlane(vertex_2_Slm, nVBoundFaces_Slm, 2*self.getSize())

        box = geompy.MakeBoxTwoPnt(vertex_1_Slm, vertex_2_Slm)
        intersection1 = geompy.MakeSection(box, bndPlane_1_Slm, True) # box planar section 
        intersection2 = geompy.MakeSection(box, bndPlane_2_Slm, True) # box planar section 

        origin_Slm = geompy.MakeVertex(*origin)
        planeInOrientation3D_Slm = geompy.MakePlane(origin_Slm, nVRotatingPlane_Slm, 4*self.getSize())
        
        bndLine_1_Slm = geompy.MakeSection(intersection1, planeInOrientation3D_Slm, True) # box planar section 
        bndLine_2_Slm = geompy.MakeSection(intersection2, planeInOrientation3D_Slm, True) # box planar section 
        
        bndVertices_1_Slm = geompy.ExtractShapes(bndLine_1_Slm, geompy.ShapeType["VERTEX"], True)
        bndVertices_2_Slm = geompy.ExtractShapes(bndLine_2_Slm, geompy.ShapeType["VERTEX"], True)

        bndVertices_1 = [geompy.PointCoordinates(v) for v in bndVertices_1_Slm]
        bndVertices_2 = [geompy.PointCoordinates(v) for v in bndVertices_2_Slm]

        def distToorigin(coords):

            dist = norm(np.array(coords) - np.array(origin))

            return dist

        bndVertices_1.sort(key=distToorigin)
        bndVertices_2.sort(key=distToorigin)

        bndVertices = {'bnd_1_near': bndVertices_1[0],
                       'bnd_1_far' : bndVertices_1[1],
                       'bnd_2_near': bndVertices_2[0],
                       'bnd_2_far' : bndVertices_2[1]
                      }

        return bndVertices 

    def _getTangentToMesh(self, origin, lb, ub):

        dist = 1.0
        tol = 0.01
        while dist > tol:

            line_lb = Line(origin, lb) 
            intersects_lb = line_lb.intersectsWith(self)

            line_ub = Line(origin, ub) 
            intersects_ub = line_ub.intersectsWith(self)

            new_point = (lb+ub)/2
            line = Line(origin, new_point) 
            intersects_new_point = line.intersectsWith(self)
    
            if intersects_new_point & intersects_lb:
                lb = new_point
            elif intersects_new_point & intersects_ub:
                ub = new_point
            elif (not intersects_new_point) & intersects_lb:
                ub = new_point
            elif (not intersects_new_point) & intersects_ub:
                lb = new_point

            dist = norm(ub - lb)

        line = Line(origin, new_point) 

        return line, new_point

    def getMinElementSize(self):

        minArea = self.mesh.GetMinMax(SMESH.FT_Area)[0]
        minSize = np.sqrt(4*minArea/np.pi)

        return minSize

    def getAdjacentElementMesh(self, elementId, coplanarAngle=5):

        aCriteria = smesh.GetCriterion(SMESH.FACE, SMESH.FT_CoplanarFaces,
                                       SMESH.FT_Undefined, elementId,
                                       SMESH.FT_Undefined, SMESH.FT_Undefined,
                                       coplanarAngle)
        aFilter = smesh.GetFilterFromCriteria([aCriteria])
        aFilter.SetMesh(self.mesh.GetMesh())
        sub_hole = self.mesh.GroupOnFilter(SMESH.FACE, 'Hole', aFilter)
        sub_mesh = smesh.CopyMesh(sub_hole, 'meshHole', 0, 1)

        return sub_mesh

    def getHoleMeshFromIds(self, ids):

        ids = str(ids).strip('[]') # '3,4,5' Remove characters

        sub_mesh2D = self.getMeshFromRangeOfIds(ids, 2)
        sub_mesh1D = self.getMeshFromRangeOfIds(ids, 1)
        sub_mesh = self.getHoleMeshKeepingOriginalIds(sub_mesh2D, sub_mesh1D, 'meshHole')       

        hole = Hole(sub_mesh)  #  instancia de Hole que tiene esa malla asociada

        return hole

    def getMeshFromRangeOfIds(self, ids, dim):

        assert dim in (1,2), 'dim must be 1 or 2' 
        smeshType = SMESH.FACE if dim == 2 else SMESH.EDGE 

        aCriteria = smesh.GetCriterion(smeshType,SMESH.FT_RangeOfIds,
                                       SMESH.FT_Undefined,ids)
        aFilter = smesh.GetFilterFromCriteria([aCriteria])
        aFilter.SetMesh(self.mesh.GetMesh())
        sub_hole = self.mesh.GroupOnFilter(smeshType, 'Hole%iD' %dim, aFilter)
        sub_mesh = smesh.CopyMesh(sub_hole, 'meshHole%iD' %dim, 0, 1)

        return sub_mesh

    def getMeshFromGroupOfLines(self, debrisLines, dim, tolerance):

        assert dim in (1,2), 'dim must be 1 or 2' 
        smeshType = SMESH.FACE if dim == 2 else SMESH.EDGE 
        assert hasattr(self, 'selectionMethod'), 'FEMesh needs attribute selectionMethod defined to use getMeshFromGroupOfLines'

        smeshMethod  = SMESH.FT_LyingOnGeom if self.selectionMethod == 'OneNode' else SMESH.FT_BelongToGeom

        aCriteria = [smesh.GetCriterion(smeshType, smeshMethod,
                                        SMESH.FT_Undefined, line.geom,
                                        SMESH.FT_Undefined, SMESH.FT_LogicalOR,
                                        tolerance) for line in debrisLines]
        aFilter = smesh.GetFilterFromCriteria(aCriteria)
        aFilter.SetMesh(self.fuselage.mesh.GetMesh())
        holesMesh = self.fuselage.mesh.GroupOnFilter(smeshType, 'groupedHolesFromDebris%iD_%s' %(dim, HolesFromDebris.Id), aFilter)
        mesh = smesh.CopyMesh(holesMesh, 'meshHolesFromDebris%iD_%s' %(dim, HolesFromDebris.Id), 0, 1)  # malla que tiene info de todos los elementos con los que intersecan las lineas del debris

        return mesh  

    def getHoleMeshKeepingOriginalIds(self, sub_mesh2D, sub_mesh1D, meshName):

        ids2D = sub_mesh2D.GetElementsId()
        ids1D = sub_mesh1D.GetElementsId()
        idsAll = self.fuselage.getElementsId()
        idsToRemove = [i for i in idsAll if i not in ids2D+ids1D]

        sub_mesh = smesh.CopyMesh(self.fuselage.mesh, meshName)
        sub_mesh.RemoveElements(idsToRemove)

        return sub_mesh

    def generateDamagedConfig(self, debrisList, damageCriteria, selectionMethod='AllNodes'):
        '''
        Dada una lista de objetos debris
        Crea grupos de elementos a eliminar: Interseca las lineas de impacto
        con la malla, dada la shape del debris
        Elmina esos grupos de elementos de la malla
        '''

        assert selectionMethod in ('AllNodes', 'OneNode'), 'Selection Method must be AllNodes or OneNode'
        size = self.getSize()
        center = self.getCenter()
        multiplier = [norm(center - debris.origin) + size for debris in debrisList]

        for mult, debris in zip(multiplier, debrisList):
            debris.line.extendLine(mult)
            #debris.line.addToStudy()

        damagedConfiguration = DamagedConfig(self, debrisList, damageCriteria, selectionMethod)  # self es el objeto fuselage y self.mesh la malla de salome de ese objeto

        return damagedConfiguration

    def exportMesh(self, name='damagedConfig.med'):
        try:
            path = Path.cwd() / name
            self.mesh.ExportMED(str(path), auto_groups=0, minor=40, overwrite=1, meshPart=None, autoDimension=1)
            pass
        except:
            print('ExportMED() failed. Invalid file name?')

    def addToStudy(self, name='fuselage'):

        smesh.SetName(self.mesh.GetMesh(), name)


class Hole(FEMesh):

    def __init__(self, mesh):

        self.mesh = mesh


class HolesFromDebris(FEMesh):

    # Class variable to use as counter
    Id = 0

    def __init__(self, fuselage, debris, damageCriteria, selectionMethod='AllNodes'):

        self.fuselage = fuselage
        self.debris = debris
        self.damageCriteria = damageCriteria
        self.selectionMethod = selectionMethod
        self.isempty = False

        # Reference the class variable
        HolesFromDebris.Id += 1

        self.groupedHoles = []  # va a ser una lista de objetos hole con la malla de cada agujero asociada. Hay que usar una recursion, y tengo que acumular sobre este vector
        self._getGroupedHoles()

        # damageCriteria es una instancia de la clase DamageCriteria con info de las curvas de velocidad y threshold. Tiene un metodo para aplicar dicho criterio
        self.damagedHoles = self.damageCriteria.apply(self.groupedHoles, self.debris.velocity)  # compobamos para cada hole de la lista si se atraviesa el fuselaje. Si atraviesa, se almacena en la lista damagedHoles

    def _getMeshFromDebris(self):

        elementSize = self.fuselage.getMinElementSize()
        tolerance = np.sqrt(2)*elementSize*1.1  # diagonal*factor
        self.debris.generateDebrisMesh(elementSize)
        debrisNodesCoords = self.debris.getNodeCoordinates()  # list with coordinates
        debrisLines = self.debris.line.translateLineToCoords(debrisNodesCoords)  # general function. list with line objects

        #for line in debrisLines: line.addToStudy('ExtendedLine-%s' % HolesFromDebris.Id)

        mesh2D = self.getMeshFromGroupOfLines(debrisLines, 2, tolerance)
        mesh1D = self.getMeshFromGroupOfLines(debrisLines, 1, tolerance)
        meshName = 'meshHolesFromDebris_%s' % HolesFromDebris.Id
        self.mesh = self.getHoleMeshKeepingOriginalIds(mesh2D, mesh1D, meshName)

    def _separateHolesOnImpacts(self, elements):  # primera vez, al hacer self.getElementsId() obtengo los ids de la malla asociada a todos los agujeros procedentes de un mismo debris

        if elements == []:
            self.isempty = True #  if elements is empty, there is no intersection or hole
        else:
            elementsCoG = self.getElementsCoG(elements)
            clusteredElements = clusteringAlgorithm(elements, elementsCoG)
            print('clusteredElements lenght',len(clusteredElements))
            self.groupedHoles = [self.getHoleMeshFromIds(cluster) for cluster in clusteredElements] #  list of hole objects

    def _sortGroupedHoles(self):

        def distanceToOrigin(hole): # defino una funcion dentro del metodo
            return norm(hole.getCenter()-self.debris.origin)

        self.groupedHoles.sort(key=distanceToOrigin) # ordena los elementos de la lista groupedHoles segun la funcion distanceToOrigin 

    def _getGroupedHoles(self):

        self._getMeshFromDebris()
        elements = self.getElementsId()
        self._separateHolesOnImpacts(elements)
        if self.groupedHoles == []:
            # If groupedHoles is empty add an empty mesh
            self.groupedHoles.append(Hole(self.mesh))  # son instancias de Hole que tienen esa malla vacia asociada
        self._sortGroupedHoles()


class DamageCriteria:

    def __init__(self, vThreshold, ftransfer):

        self.vThreshold = vThreshold
        self.ftransfer = ftransfer

    def apply(self, groupedHoles, velocity):

        damagedHoles = []
        for hole in groupedHoles:
            if velocity > self.vThreshold:
                damagedHoles.append(hole)
            velocity = self.ftransfer(velocity)
        return damagedHoles


class DamagedConfig(FEMesh):

    def __init__(self, fuselage, debrisList, damageCriteria, selectionMethod='AllNodes'):

        self.fuselage = fuselage  # objeto FEMesh
        self.debrisList = debrisList
        self.damageCriteria = damageCriteria
        self.selectionMethod = selectionMethod

        self._intersectLinesAndCutHoles()

    def _intersectLinesAndCutHoles(self):
        '''
        Aplica la clase HolesFromDebris, donde para una debris dado, agrupa los elementos asciados a un mismo agujero
        groupedHoles es una lista de instancias de Hole, donde cada objeto tiene info de la malla de cada agujero
        para cada linea tengo un groupedHoles
        '''
        self.holesFromDebrisList = [HolesFromDebris(self.fuselage, debris, self.damageCriteria, self.selectionMethod) 
                                    for debris in self.debrisList]

        elementsToRemove = []
        for holesFromDebris in self.holesFromDebrisList:
            for hole in holesFromDebris.damagedHoles:
                elementsToRemove += hole.getElementsId()

        np.savetxt('medIds.txt', elementsToRemove, fmt='%d')

        self.mesh = smesh.CopyMesh(self.fuselage.mesh, 'DamagedMesh')
        self.mesh.RemoveElements(elementsToRemove)

    def addToStudy(self):

        super().addToStudy(name='damagedMesh')  # call the parent method

        for debris in self.debrisList: debris.line.addToStudy()  # los elementos de esta lista ya son objetos de tipo Line
