from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import threading
import queue
from water_network import WaterNetwork, create_anytown_network, create_balerma_network
from main import map_to_commercial
from pso_cuda import CUDA_PSO, ParallelPSO
import wntr

app = Flask(__name__)
CORS(app)

result_queue = queue.Queue()


def create_fitness_function(network_name):
    """
    Refactored to use WNTR for looped hydraulics.
    """
    # 1. Map names to actual .inp files in your repo
    inp_files = {
        "anytown": "networks/anytown.inp",
        "balerma": "networks/balerma.inp"
    }
    inp_path = inp_files.get(network_name, "networks/anytown.inp")

    # 2. Load the model once (The Template)
    wn = wntr.network.WaterNetworkModel(inp_path)
    pipe_names = wn.pipe_name_list
    junction_names = wn.junction_name_list

    # Pre-calculate lengths for cost (constant)
    lengths = np.array([wn.get_link(p).length for p in pipe_names])

    def fitness(diameters):
        # A. Update pipe diameters (Continuous from PSO -> Discrete/Physical)
        # Note: PSO might give small values, ensure they are > 0

        real_diams = map_to_commercial(diameters)
        for i, d in enumerate(diameters):
            pipe = wn.get_link(pipe_names[i])
            # EPANET expects meters, PSO usually provides mm
            pipe.diameter = max(d, 10.0) / 1000.0

        # B. Run EPANET Simulator
        try:
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()

            # C. Extract Pressures
            # We take the pressure at the first timestep (0 seconds)
            pressures = results.node['pressure'].loc[0, junction_names].values

            # D. Objective 1: Minimize Construction Cost
            # Using your heuristic: Diameter * Length * 100
            cost = np.sum(diameters * lengths * 100.0)

            # E. Objective 2: Penalty for Pressure Violations
            pressure_penalty = 0.0
            min_required_p = 20.0  # Common benchmark for Anytown/Balerma

            for p in pressures:
                if p < min_required_p:
                    # Use squared penalty to give the PSO a "direction" back to feasibility
                    pressure_penalty += (min_required_p - p)**2 * 50000.0

            return float(cost + pressure_penalty)

        except Exception as e:
            # If the network is hydraulically impossible (e.g., disconnected)
            return 1e18
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
        original_fitness, original_details = network.evaluate_fitness(
            original_diams)

        pipe_lengths = network.get_pipe_lengths()
        original_cost = float(np.sum(original_diams * pipe_lengths * 100.0))

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
        best_diameters, best_fitness = pso.optimize(
            fitness_func, bounds, verbose=False)
        optimization_time = time.time() - start_time

        _, details = network.evaluate_fitness(best_diameters)

        pipe_lengths = network.get_pipe_lengths()
        optimized_cost = float(np.sum(best_diameters * pipe_lengths * 100.0))

        result = {
            "success": True,
            "original_cost": original_cost,
            "optimized_cost": optimized_cost,
            "improvement": float(
                (original_cost - optimized_cost) / original_cost * 100
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
