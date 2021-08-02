import salome
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
#sns.set()


def clusteringAlgorithm(Xids, Xcoords):

    # Calculate the value of epsilon
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(Xcoords)
    distances, indices = nbrs.kneighbors(Xcoords)
    
    eps = max(distances[:,1])
      
    #  Train the model, selecting eps and setting min_samples
    m = DBSCAN(eps=1.5*eps, min_samples=2, metric='euclidean')
    m.fit(Xcoords)
    
    #  The labels_ property contains the list of cluster ids associated to each Xcoord
    clusters = m.labels_
    
    nclusters = len(set(clusters))
    Xclustered = [ [] for i in range(nclusters)]
    for cid, xid in zip(clusters, Xids):
        Xclustered[cid].append(xid)

    return Xclustered
    
def plotXclustered(Xcoords, clusters):

    #  Map every individual cluster to a color.
    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 
              'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
    
    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    
    fig = plt.figure()
    ## python 2.x
    #ax = Axes3D(fig)
    # python 3.x
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(Xcoords[:,0], Xcoords[:,1], Xcoords[:,2], c=vectorizer(clusters))
    plt.show()
    plt.close()

def getTranslationalRiskAngleRefAxis(orientation3D, rotatingPlane):
    '''
       Identify if the coordRef of the tangent point is bigger than or lower than 
       the coordRef of origin 
    '''

    vector = np.array(orientation3D)
    axisCode = rotatingPlane[0]
    
    if axisCode == 'X':
        axis = np.array([1.0, 0.0, 0.0])
    elif axisCode == 'Y':
        axis = np.array([0.0, 1.0, 0.0])

    # check if orientation3D is parallel to one of the vector that defines the rotating plane
    crossProduct = np.cross(vector,axis) 
    normCrossProduct = np.linalg.norm(crossProduct)

    if normCrossProduct == 0.0:
        refAxisCode = rotatingPlane.replace(axisCode,'') # the refAxisCode is the other vector of the rotating plane
    else:
        refAxisCode = axisCode

    return refAxisCode

def clearAll():

    builder = salome.myStudy.NewBuilder()
    for compName in ["SMESH", "GEOM"]:
        comp = salome.myStudy.FindComponent(compName)
        if comp:
            iterator = salome.myStudy.NewChildIterator( comp )
            while iterator.More():
                sobj = iterator.Value()
                iterator.Next()
                builder.RemoveObjectWithChildren( sobj )

