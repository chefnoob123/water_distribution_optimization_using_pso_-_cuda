import queue
import threading
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from main import create_fitness_function, map_to_commercial
from pso_cuda import CUDA_PSO, ParallelPSO
from water_network import create_anytown_network, create_balerma_network

app = Flask(__name__)
CORS(app)

result_queue = queue.Queue()


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

        original_diams = map_to_commercial(network.get_original_diameters())
        _, original_details = network.evaluate_fitness(original_diams)

        if subswarms > 1:
            pso = ParallelPSO(
                n_particles=particles,
                n_iterations=iterations,
                n_sub_swarms=subswarms,
                w=0.7,
                c1=1.5,
                c2=1.5,
            )
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
        best_diameters, _ = pso.optimize(fitness_func, bounds, verbose=False)
        optimization_time = time.time() - start_time
        best_diameters = map_to_commercial(best_diameters)
        _, details = network.evaluate_fitness(best_diameters)

        original_cost = float(original_details["cost"])
        optimized_cost = float(details["cost"])
        improvement = 0.0
        if original_cost > 0.0:
            improvement = float(
                (original_cost - optimized_cost) / original_cost * 100.0
            )

        result_queue.put(
            {
                "success": True,
                "original_cost": original_cost,
                "optimized_cost": optimized_cost,
                "improvement": improvement,
                "optimization_time": float(optimization_time),
                "min_pressure": float(details["min_pressure"]),
                "max_pressure": float(details["max_pressure"]),
                "num_nodes": network.num_nodes,
                "num_pipes": network.num_pipes,
                "convergence_history": pso.get_convergence_history().tolist(),
                "best_diameters": best_diameters.tolist(),
                "original_diameters": original_diams.tolist(),
                "pressures": details["pressures"].tolist(),
                "use_cuda": bool(pso.use_cuda),
            }
        )
    except Exception as exc:
        result_queue.put({"success": False, "error": str(exc)})


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.json or {}
    network_name = data.get("network", "anytown")
    particles = int(data.get("particles", 256))
    iterations = int(data.get("iterations", 100))
    subswarms = int(data.get("subswarms", 4))
    use_cuda = bool(data.get("use_cuda", True))

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
    except queue.Empty:
        return jsonify({"status": "processing"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
