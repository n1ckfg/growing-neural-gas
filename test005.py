import trimesh
import latk
import neuralgas
from scipy.spatial.distance import cdist

mesh = trimesh.load("test.ply")
points = mesh.vertices

# defaults: max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200
gas = neuralgas.GrowingNeuralGas(points, max_neurons=100000, max_iter=100, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=20)
gas.learn()

la = latk.Latk(init=True)

edgeList = []

for edge in gas.gng.es:    
    point1 = gas.gng.vs[edge.source]["weight"]
    point2 = gas.gng.vs[edge.target]["weight"]
    edgeList.append((point1, point2))

newEdgeList = []
newEdge = []

for edge in edgeList:
    newEdge.append(edge[0])
    newEdge.append(edge[1])
    newEdgeList.append(newEdge)
    newEdge = []

for newEdge in newEdgeList:
    ls = latk.LatkStroke()
    
    for point in newEdge:
        lp = latk.LatkPoint(co=(point[0], point[2], point[1]))
        ls.points.append(lp)

    la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")

#radius = 0.001
#cdist([points[i]], [points[strokeGroup[-1]]])[0][0] < radius: