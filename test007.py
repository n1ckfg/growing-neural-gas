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
gas = neuralgas.GrowingNeuralGas(points, max_neurons=100000, max_iter=1000, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=20)
gas.learn()

# 2. get edge indices
edgeIndices = []
for edge in gas.gng.es:
    edgeIndices.append((edge.source, edge.target))

# 3. merge edges with matching indices
reps = 5

for i in range(0, reps):
    newEdgeIndices = []

    while edgeIndices:
        edge = edgeIndices.pop(0)
        for j, matchingEdge in enumerate(edgeIndices):
            '''
            if (i <= int(reps/2)):
                if (edge[1] == matchingEdge[0]):
                    newEdge = edgeIndices.pop(j)
                    edge = edge + newEdge
                    break
            else:
            '''
            if (edge[0] == matchingEdge[0] or edge[0] == matchingEdge[1] or edge[1] == matchingEdge[0] or edge[1] == matchingEdge[1]):
                newEdge = edgeIndices.pop(j)
                edge = edge + newEdge
                break

        edge = list(dict.fromkeys(edge)) # this removes repeated indices
        print(edge)

        newEdgeIndices.append(edge)

    edgeIndices = newEdgeIndices

newEdgeIndices = []
for edge in edgeIndices:
    if (len(edge) > 2):
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
