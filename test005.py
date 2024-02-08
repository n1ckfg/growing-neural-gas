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

for edge in edgeList:
    ls = latk.LatkStroke()

    point1 = edge[0]
    point2 = edge[1]

    lp1 = latk.LatkPoint(co=(point1[0], point1[2], point1[1]))
    lp2 = latk.LatkPoint(co=(point2[0], point2[2], point2[1]))
    ls.points.append(lp1)
    ls.points.append(lp2)

    la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")
