import trimesh
import latk
import neuralgas
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
            strokeGroups.append(strokeGroup)

        print("Found " + str(len(strokeGroups)) + " strokeGroups, " + str(len(unassigned_points)) + " points remaining.")
    return strokeGroups

mesh = trimesh.load("test.ply")
points = mesh.vertices
newPoints = []

# dataset, max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200
gas = neuralgas.GrowingNeuralGas(points, max_neurons=1000000, max_iter=1000, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.999, l=200)
gas.learn()

radius = 0.5
minPointsCount = 5

for i in range(0, len(gas.gng.vs)):
    point = gas.gng.vs[i]["weight"]
    newPoints.append((point[0], point[2], point[1]))

strokeIndexGroups = group_points_into_strokes(newPoints, radius, minPointsCount)

la = latk.Latk(init=True)

for strokeIndexGroup in strokeIndexGroups:
    ls = latk.LatkStroke()
    for index in strokeIndexGroup:
        lp = latk.LatkPoint(co=newPoints[index])
        ls.points.append(lp)
    la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")
