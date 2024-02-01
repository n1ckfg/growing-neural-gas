import trimesh
import latk
import neuralgas

mesh = trimesh.load("test.ply")
points = mesh.vertices

# dataset, max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200
gas = neuralgas.GrowingNeuralGas(points, max_neurons=10000, max_iter=100, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.999, l=200)
gas.learn()

la = latk.Latk(init=True)
ls = latk.LatkStroke()

limit = 10

for i in range(0, len(gas.gng.vs)):
    point = gas.gng.vs[i]["weight"]
    lp = latk.LatkPoint(co=(point[0], point[2], point[1]))
    ls.points.append(lp)
    if (len(ls.points) >= limit):
        la.layers[0].frames[0].strokes.append(ls)
        ls = latk.LatkStroke()

if (len(ls.points) > 1):
    la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")
