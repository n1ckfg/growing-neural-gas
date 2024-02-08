import trimesh
import latk
import neuralgas

mesh = trimesh.load("test.ply")
points = mesh.vertices

# defaults: max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200
gas = neuralgas.GrowingNeuralGas(points, max_neurons=100000, max_iter=100, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=20)
gas.learn()

# get edges from indices
edgeList = []

for edge in gas.gng.es:
    point1 = gas.gng.vs[edge.source]["weight"]
    point2 = gas.gng.vs[edge.target]["weight"]
    edgeList.append((point1, point2))

# TODO merge edges that share points
newEdgeList = edgeList

# convert edge list to Latk
la = latk.Latk(init=True)

for edge in newEdgeList: 
    ls = latk.LatkStroke()
    
    for point in edge:
        lp = latk.LatkPoint(co=(point[0], point[2], point[1]))
        ls.points.append(lp)

    la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")
