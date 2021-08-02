import damageCreator
from damageCreator import FEMesh, DamageCriteria
from damageCreator import ScatteringPrism

import random
from numpy.random import normal
import numpy as np

from scipy.interpolate import interp1d
from scipy.stats import rv_discrete

from auxiliaryFunctions import clearAll

#random.seed(1)
#np.random.seed(1)

def createDebrisList(debrisDatabase, prismParameters):

    debrisList = []
    scatteringPrism = {}
    for k in debrisDatabase.keys():
        prismParameters['fwdAngle'] = debrisDatabase[k]['fwdAngle']
        prismParameters['aftAngle'] = debrisDatabase[k]['aftAngle']
    
        scatteringPrism[k] = ScatteringPrism(prismParameters, translationalRiskAngle)

        a = debrisDatabase[k]['a'] # model has in units
        b = debrisDatabase[k]['b']

        nImpactLines_k = debrisDatabase[k]['nImpactLines']
        shapes = [{'a': a, 'b': b}] * debrisDatabase[k]['nImpactLines']
        velocities = [debrisDatabase[k]['velocity']] * debrisDatabase[k]['nImpactLines']
        debrisList +=  scatteringPrism[k].generateDebris(nImpactLines_k, shapes, velocities) #accumulate all the debris from diferent normSizes
        
    del scatteringPrism

    return debrisList

def changeSpreadAnglesSign(debrisDatabase):

    for k,value in debrisDatabase.items():
        value['fwdAngle'] *= -1
        value['aftAngle'] *= -1
        debrisDatabase[k] = value

    return debrisDatabase

NameFile = 'meshes/fuselage'
fuselage = FEMesh(NameFile)


origin = [17.0*39.37, -5.75*39.37, -1.75*39.37]
orientation3D = [0, 1, 0] # must be parallel to one coordinate axis
rotatingPlane = 'YZ'
fwdDirection = [-1, 0, 0]

translationalRiskAngle = (1, 45)

vThreshold = 200 # m/s
ftransfer = interp1d([246, 300], [201, 250], fill_value='extrapolate')
damageCriteria = DamageCriteria(vThreshold, ftransfer)

prismParameters = {'origin': origin, 'orientation3D': orientation3D, 'rotatingPlane': rotatingPlane}

nDamagedConfigs = 5
for i in range(nDamagedConfigs):

    fuselage = FEMesh(NameFile)

    # --------- Blades ----------

    bladeLength = 32 # in
    bladeWidth  =  8 # in

    mu = 8
    sig = 0.05*mu
    nImpactLines = int(round(normal(mu, sig, 1)[0]))
    print('Blade impacts =', nImpactLines)

    # Dicrete Random distribution of normalized sizes
    xk = (1, 2, 3, 5, 7, 10) # xk must be an integer number to apply rv_discrete
    pk = (10/27, 9/27, 2/27, 4/27, 1/27, 1/27)

    custm = rv_discrete(values=(xk,pk))
    sampleNormSizes = custm.rvs(size=nImpactLines)

    sampleNormSizes = sampleNormSizes/10
    print('Blade sizes =', sampleNormSizes)

    debrisDatabase = {0.1: {'nImpactLines':list(sampleNormSizes).count(0.1), 'a':bladeLength*0.1, 'b':bladeLength*0.1, 'velocity':935 , 'fwdAngle':10, 'aftAngle':-30}, 
                      0.2: {'nImpactLines':list(sampleNormSizes).count(0.2), 'a':bladeLength*0.2, 'b':bladeLength*0.2, 'velocity':928 , 'fwdAngle':15, 'aftAngle':-25}, 
                      0.3: {'nImpactLines':list(sampleNormSizes).count(0.3), 'a':bladeLength*0.3, 'b':bladeWidth     , 'velocity':894 , 'fwdAngle':10, 'aftAngle':-25}, 
                      0.5: {'nImpactLines':list(sampleNormSizes).count(0.5), 'a':bladeLength*0.5, 'b':bladeWidth     , 'velocity':822 , 'fwdAngle':10, 'aftAngle':-20}, 
                      0.7: {'nImpactLines':list(sampleNormSizes).count(0.7), 'a':bladeLength*0.7, 'b':bladeWidth     , 'velocity':796 , 'fwdAngle':10, 'aftAngle':-20}, 
                      1.0: {'nImpactLines':list(sampleNormSizes).count(1.0), 'a':bladeLength    , 'b':bladeWidth     , 'velocity':644 , 'fwdAngle':15, 'aftAngle':  5}, 
                      }

    if min(fwdDirection) < 0:
        debrisDatabase = changeSpreadAnglesSign(debrisDatabase)                   

    debrisList = createDebrisList(debrisDatabase, prismParameters)

    # --------- Disks ----------

    diskDiamater = 25.8 # in
    diskThickness  = 6.7 # in

    mu = 3
    sig = 0.2*mu
    nImpactLines = int(round(normal(mu, sig, 1)[0]))
    print('Disk impacts =', nImpactLines)

    sampleNormSizes = nImpactLines*[1.0]
    print('Disk sizes =', sampleNormSizes)

    debrisDatabase = {1.0: {'nImpactLines':list(sampleNormSizes).count(1.0), 'a':diskDiamater, 'b':diskThickness, 'velocity':303 , 'fwdAngle':2, 'aftAngle':-3}
                      }    

    debrisList += createDebrisList(debrisDatabase,prismParameters)

    # --------------------------

    damagedConfiguration = fuselage.generateDamagedConfig(debrisList, damageCriteria, selectionMethod='OneNode')
    damagedConfiguration.exportMesh('damagedConfig-%s.med' % str(i+1))
    damagedConfiguration.addToStudy()

    clearAll()
