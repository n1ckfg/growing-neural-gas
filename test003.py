import trimesh
import latk
import neuralgas

mesh = trimesh.load("test.ply")
points = mesh.vertices

# dataset, max_neurons=2000, max_iter=8, max_age=10, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=200
gas = neuralgas.GrowingNeuralGas(points, max_neurons=100, max_iter=1000, max_age=100, eb=0.1, en=0.006, alpha=0.5, beta=0.995, l=2)
gas.learn()

la = latk.Latk(init=True)

for edge in gas.gng.es:
    if (edge["age"] < 2):
        ls = latk.LatkStroke()
        point1 = gas.gng.vs[edge.source]["weight"]
        point2 = gas.gng.vs[edge.target]["weight"]
        
        lp1 = latk.LatkPoint(co=(point1[0], point1[2], point1[1]))
        lp2 = latk.LatkPoint(co=(point2[0], point2[2], point2[1]))
        ls.points.append(lp1)
        ls.points.append(lp2)

        la.layers[0].frames[0].strokes.append(ls)

la.write("test.latk")
