#!/usr/bin/env python3
"""
Smart Water Distribution Network Optimization using CUDA Particle Swarm Optimization
Supports real EPANET network files (.inp format)
"""

import numpy as np
import time
import argparse
from water_network import WaterNetwork, create_anytown_network, create_balerma_network
from pso_cuda import CUDA_PSO, ParallelPSO
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Water Distribution Network Optimization with CUDA PSO"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="anytown",
        choices=["anytown", "balerma", "synthetic", "custom"],
        help="Network to optimize: anytown (22 nodes), balerma (443 nodes), synthetic, or custom",
    )
    parser.add_argument(
        "--inp-file",
        type=str,
        default="networks/anytown.inp",
        help="Path to EPANET .inp file (for custom network)",
    )
    parser.add_argument(
        "--particles", type=int, default=256, help="Number of PSO particles"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--subswarms", type=int, default=4, help="Number of sub-swarms for parallel PSO"
    )
    parser.add_argument(
        "--output", type=str, default="results.npz", help="Output file for results"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    return parser.parse_args()


# Standard commercial sizes in mm
COMMERCIAL_DIAMS = np.array(
    [50.8, 101.6, 152.4, 203.2, 254.0, 304.8, 355.6, 406.4, 457.2, 508.0])


def map_to_commercial(diameters):
    """Snaps continuous PSO values to the nearest commercial pipe size."""
    # Reshape for broadcasting
    idx = (np.abs(COMMERCIAL_DIAMS[:, None] - diameters)).argmin(axis=0)
    return COMMERCIAL_DIAMS[idx]


def create_fitness_function(network):
    """Create optimized fitness function for the network"""
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
                    hf = 10.67 * L * (Q_m3s**1.852) / \
                        ((150**1.852) * (D_m**4.8704))
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


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Smart Water Distribution Network Optimization")
    print("Using CUDA Particle Swarm Optimization (PSO)")
    print("=" * 70)

    print("\n[1/5] Loading water distribution network...")

    if args.network == "anytown":
        network = create_anytown_network()
    elif args.network == "balerma":
        network = create_balerma_network()
    elif args.network == "custom":
        if os.path.exists(args.inp_file):
            network = WaterNetwork()
            network.load_epanet_inp(args.inp_file)
        else:
            print(f"Warning: File {
                  args.inp_file} not found. Using Anytown instead.")
            network = create_anytown_network()
    else:
        network = WaterNetwork()
        network.create_synthetic_network(num_nodes=20)

    network.print_summary()

    print("\n[2/5] Setting up optimization parameters...")
    bounds = network.get_diameter_bounds()
    fitness_func = create_fitness_function(network)

    print(f"Decision variables (pipe diameters): {bounds.shape[0]}")
    print(f"Diameter range: {bounds[0, 0]:.1f} - {bounds[0, 1]:.1f} mm")
    print(
        f"Original diameters: {
            np.mean(network.get_original_diameters()):.1f} mm (avg)"
    )

    print("\n[3/5] Initializing CUDA PSO...")

    if args.subswarms > 1:
        pso = ParallelPSO(
            n_particles=args.particles,
            n_iterations=args.iterations,
            n_sub_swarms=args.subswarms,
            w=0.7,
            c1=1.5,
            c2=1.5,
        )
        if args.no_cuda:
            pso.use_cuda = False
    else:
        pso = CUDA_PSO(
            n_particles=args.particles,
            n_iterations=args.iterations,
            w=0.7,
            c1=1.5,
            c2=1.5,
        )
        if args.no_cuda:
            pso.use_cuda = False

    print("\n[4/5] Running optimization...")
    start_time = time.time()

    best_diameters, best_fitness = pso.optimize(
        fitness_func, bounds, verbose=args.verbose
    )

    optimization_time = time.time() - start_time

    print(f"\nOptimization completed in {optimization_time:.2f} seconds")

    print("\n[5/5] Results and Analysis...")
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Network: {network.network_name}")
    print(f"Best Fitness (Cost): {best_fitness:.4f}")
    print(f"\nOptimal Pipe Diameters:")
    print("-" * 50)
    for i, d in enumerate(best_diameters):
        orig = network.pipe_data[i].get("original_diameter", 100.0)
        change = ((d - orig) / orig) * 100
        direction = "↑" if change > 0 else "↓" if change < 0 else "="
        print(
            f"  Pipe {
                i + 1:2d}: {d:7.1f}mm | Original: {orig:6.1f}mm | {direction}{abs(change):5.1f}%"
        )

    fitness_details, details = network.evaluate_fitness(best_diameters)
    original_fitness, _ = network.evaluate_fitness(
        network.get_original_diameters())

    print(f"\nCost Comparison:")
    print("-" * 50)
    print(f"  Original Design Cost: {original_fitness:.2f}")
    print(f"  Optimized Design Cost: {best_fitness:.2f}")
    improvement = ((original_fitness - best_fitness) / original_fitness) * 100
    print(f"  Cost Reduction: {improvement:.1f}%")

    print(f"\nHydraulic Analysis:")
    print("-" * 50)
    print(f"  Infrastructure Cost: ${details['cost']:.2f}")
    print(f"  Pressure Penalty: {details['pressure_penalty']:.2f}")

    print(f"\nNode Hydraulic Head:")
    for i, h in enumerate(details["head"][:10]):
        pressure = h - network.node_elevations[i]
        status = "OK" if pressure >= 10 else "LOW"
        print(f"  Node {i:2d}: Head={h:7.2f}m | Pressure={
              pressure:6.2f}m | {status}")
    if len(details["head"]) > 10:
        print(f"  ... and {len(details['head']) - 10} more nodes")

    convergence = pso.get_convergence_history()
    np.savez(
        args.output,
        best_diameters=best_diameters,
        best_fitness=best_fitness,
        convergence_history=convergence,
        optimization_time=optimization_time,
        network_config={
            "num_nodes": network.num_nodes,
            "num_pipes": network.num_pipes,
            "node_elevations": network.node_elevations,
            "base_demands": network.base_demands,
            "network_name": network.network_name,
            "original_diameters": network.get_original_diameters(),
        },
    )

    print(f"\nResults saved to: {args.output}")
    print("\nTo visualize results, run: python visualize.py --input results.npz")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
