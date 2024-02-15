import trimesh
import latk
import neuralgas
import numpy as np
from scipy.spatial.distance import cdist

def group_points_into_strokes(points, radius, minPointsCount):
    strokeGroups = []
    unassigned_points = set(range(len(points)))

    while len(unassigned_points) > 0:
        strokeGroup = [next(iter(unassigned_points))]
        unassigned_points.remove(strokeGroup[0])

        for i in range(len(points)):
            if i in unassigned_points and cdist([points[i]], [points[strokeGroup[-1]]])[0][0] < radius:
                strokeGroup.append(i)
                unassigned_points.remove(i)

        if (len(strokeGroup) >= minPointsCount):
            strokeGroup = list(dict.fromkeys(strokeGroup)) # remove duplicates
            strokeGroups.append(strokeGroup)

        print("Found " + str(len(strokeGroups)) + " strokeGroups, " + str(len(unassigned_points)) + " points remaining.")
    return strokeGroups

mesh = trimesh.load("test.ply")
points = mesh.vertices

# 1. Generate GNG
# defaults: max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200
gas = neuralgas.GrowingNeuralGas(points, max_neurons=100000, max_iter=1000, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=20)
gas.learn()

verts = []
for vert in gas.gng.vs:
    verts.append(vert["weight"])


radius = 0.2
minPointsCount = 5
strokeGroups = group_points_into_strokes(verts, radius, minPointsCount)

# 5. Convert points to Latk strokes
la = latk.Latk(init=True)

for stroke in strokeGroups: 
    ls = latk.LatkStroke()
    
    for index in stroke:
        point = verts[index]
        lp = latk.LatkPoint(co=(point[0], point[2], point[1]))
        ls.points.append(lp)

    la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")
