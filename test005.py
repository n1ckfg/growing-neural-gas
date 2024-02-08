import trimesh
import latk
import neuralgas
import numpy as np

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

mesh = trimesh.load("test.ply")
points = mesh.vertices

# defaults: max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200
gas = neuralgas.GrowingNeuralGas(points, max_neurons=100000, max_iter=500, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=20)
gas.learn()

la = latk.Latk(init=True)

edgeList = []

for edge in gas.gng.es:
    point1 = gas.gng.vs[edge.source]["weight"]
    point2 = gas.gng.vs[edge.target]["weight"]
    
    edgeList.append((point1, point2))

newEdgeList = []
newEdge = []

for i in range(0, len(edgeList)):
    if (i == 0):
        point1 = edgeList[i][0]
        point2 = edgeList[i][1]
        newEdge.append(point1)
        newEdge.append(point2)    
    else:
        point1 = edgeList[i][0]
        point2 = edgeList[i][1]
        lastPoint1 = edgeList[i-1][0]
        lastPoint2 = edgeList[i-1][1]

        matchingPoint1 = np.array_equal(point1, lastPoint1) or np.array_equal(point1, lastPoint2)
        matchingPoint2 = np.array_equal(point2, lastPoint1) or np.array_equal(point2, lastPoint2)

        if (matchingPoint1 == True and matchingPoint2 == False): # point 2 is new
            newEdge.append(point2)
        elif (matchingPoint1 == False and matchingPoint2 == True): # point 1 is new
            newEdge.append(point1)
        elif(matchingPoint1 == False and matchingPoint2 == False): # points 1 and 2 are new
            newEdgeList.append(newEdge)
            newEdge = []
            newEdge.append(point1)
            newEdge.append(point2)
        elif (matchingPoint1 == True and matchingPoint2 == True):  # no new points
            pass

    if (i == len(edgeList) - 1 and len(newEdge) > 0):  # add unused points
        newEdgeList.append(newEdge)

for edge in newEdgeList:
    ls = latk.LatkStroke()

    for point in edge:
        lp = latk.LatkPoint(co=(point[0], point[2], point[1]))
        ls.points.append(lp)

    la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")
