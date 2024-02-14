import trimesh
import latk
import neuralgas
import numpy as np

def distance(a, b):
    return np.linalg.norm(a - b)

mesh = trimesh.load("test.ply")
points = mesh.vertices

# 1. Generate GNG
# defaults: max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200
gas = neuralgas.GrowingNeuralGas(points, max_neurons=100000, max_iter=500, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=20)
gas.learn()

# 2. get edge indices
edgeIndices = []
for edge in gas.gng.es:
    edgeIndices.append((edge.source, edge.target))

# 3. merge edges with matching indices
newEdgeIndices = []
while edgeIndices:
    edge = edgeIndices.pop(0)

    for index in edge:
      for i, matchingEdge in enumerate(edgeIndices):
        for matchingIndex in matchingEdge:
          if (index == matchingIndex):
            edge = edge + edgeIndices.pop(i)
            break

    edge = list(dict.fromkeys(edge)) # this removes repeated indices
    edge.sort()

    newEdgeIndices.append(edge)

# 4. Get points from indices
edgeList = []

for edge in newEdgeIndices:
    points = []
    
    for index in edge:
        points.append(gas.gng.vs[index]["weight"])
    
    #points = sorted(points, key=lambda point: distance(point, points[0]))
    points = sorted(points, key=lambda point: distance(point, (0,0,0)))
    edgeList.append(points)

# 5. Convert points to Latk strokes
la = latk.Latk(init=True)

for edge in edgeList: 
    ls = latk.LatkStroke()
    
    for point in edge:
        lp = latk.LatkPoint(co=(point[0], point[2], point[1]))
        ls.points.append(lp)

    la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")
