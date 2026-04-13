from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import threading
import queue
from water_network import WaterNetwork, create_anytown_network, create_balerma_network
from pso_cuda import CUDA_PSO, ParallelPSO

app = Flask(__name__)
CORS(app)

result_queue = queue.Queue()


def create_fitness_function(network):
    pipe_lengths = network.get_pipe_lengths()
    elevations = network.node_elevations.copy()
    demand_nodes = network.demand_nodes
    pipe_data = network.pipe_data
    num_pipes = network.num_pipes
    num_nodes = network.num_nodes

    def calculate_subtree_demand(node, children, demands):
        total = demands.get(node, 0.0)
        for child in children.get(node, []):
            total += calculate_subtree_demand(child, children, demands)
        return total

    def fitness(diameters):
        children = {i: [] for i in range(num_nodes)}
        parent = [-1] * num_nodes
        pipe_to_node = {}

        for i in range(num_pipes):
            u = pipe_data[i]["from"]
            v = pipe_data[i]["to"]
            parent[v] = u
            children[u].append(v)
            pipe_to_node[v] = i

        demands = {i: network.base_demands[i] for i in range(num_nodes)}

        pipe_flows = {}
        for node in range(1, num_nodes):
            flow = calculate_subtree_demand(node, children, demands)
            pipe_idx = pipe_to_node.get(node, -1)
            if pipe_idx >= 0:
                pipe_flows[pipe_idx] = flow

        head = np.zeros(num_nodes)
        head[0] = elevations[0] + 50.0

        for node in range(1, num_nodes):
            path_head_loss = 0.0
            curr = node
            pipe_count = 0
            while curr != 0 and parent[curr] != -1 and pipe_count < num_nodes:
                pid = pipe_to_node.get(curr, -1)
                if pid == -1:
                    break
                L = pipe_lengths[pid]
                D_m = max(diameters[pid], 1.0) / 1000.0
                Q_m3s = max(pipe_flows.get(pid, 0.001), 0.0001) / 1000.0
                if D_m > 0 and Q_m3s > 0:
                    hf = 10.67 * L * (Q_m3s**1.852) / ((150**1.852) * (D_m**4.8704))
                else:
                    hf = 0.0
                path_head_loss += hf
                curr = parent[curr]
                pipe_count += 1

            head[node] = head[0] - path_head_loss

        pipe_cost = np.sum(diameters * pipe_lengths * 100.0)

        pressure_penalty = 0.0
        min_pressure = 15.0
        for node in demand_nodes:
            if node < len(head):
                pressure = head[node] - elevations[node]
                if pressure < min_pressure:
                    pressure_penalty += (min_pressure - pressure) * 1000.0

        return pipe_cost + pressure_penalty

    return fitness


def run_optimization_async(network_name, particles, iterations, subswarms, use_cuda):
    try:
        if network_name == "anytown":
            network = create_anytown_network()
        elif network_name == "balerma":
            network = create_balerma_network()
        else:
            network = create_anytown_network()

        bounds = network.get_diameter_bounds()
        fitness_func = create_fitness_function(network)

        original_diams = network.get_original_diameters()
        original_fitness, original_details = network.evaluate_fitness(original_diams)

        if subswarms > 1:
            pso = ParallelPSO(
                n_particles=particles,
                n_iterations=iterations,
                n_sub_swarms=subswarms,
                w=0.7,
                c1=1.5,
                c2=1.5,
            )
            if not use_cuda:
                pso.use_cuda = False
        else:
            pso = CUDA_PSO(
                n_particles=particles,
                n_iterations=iterations,
                w=0.7,
                c1=1.5,
                c2=1.5,
            )
            if not use_cuda:
                pso.use_cuda = False

        start_time = time.time()
        best_diameters, best_fitness = pso.optimize(fitness_func, bounds, verbose=False)
        optimization_time = time.time() - start_time

        optimized_fitness, details = network.evaluate_fitness(best_diameters)

        result = {
            "success": True,
            "original_cost": float(original_fitness),
            "optimized_cost": float(optimized_fitness),
            "improvement": float(
                (original_fitness - optimized_fitness) / original_fitness * 100
            ),
            "optimization_time": float(optimization_time),
            "min_pressure": float(
                min(details["head"][1:] - network.node_elevations[1:])
            ),
            "max_pressure": float(
                max(details["head"][1:] - network.node_elevations[1:])
            ),
            "num_nodes": network.num_nodes,
            "num_pipes": network.num_pipes,
            "convergence_history": pso.get_convergence_history().tolist(),
            "best_diameters": best_diameters.tolist(),
            "original_diameters": original_diams.tolist(),
            "head": details["head"].tolist(),
            "elevations": network.node_elevations.tolist(),
            "use_cuda": pso.use_cuda,
        }

        result_queue.put(result)

    except Exception as e:
        result_queue.put({"success": False, "error": str(e)})


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.json

    network_name = data.get("network", "anytown")
    particles = int(data.get("particles", 256))
    iterations = int(data.get("iterations", 100))
    subswarms = int(data.get("subswarms", 4))
    use_cuda = data.get("use_cuda", True)

    thread = threading.Thread(
        target=run_optimization_async,
        args=(network_name, particles, iterations, subswarms, use_cuda),
    )
    thread.start()

    return jsonify({"status": "started"})


@app.route("/api/result", methods=["GET"])
def get_result():
    try:
        result = result_queue.get_nowait()
        return jsonify(result)
    except:
        return jsonify({"status": "processing"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
