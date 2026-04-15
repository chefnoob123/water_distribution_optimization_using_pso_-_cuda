#!/usr/bin/env python3
"""
Smart Water Distribution Network Optimization using CUDA Particle Swarm Optimization.
"""

import argparse
import os
import time

import numpy as np

from pso_cuda import CUDA_PSO, ParallelPSO
from water_network import (
    COMMERCIAL_DIAMETERS_MM,
    WaterNetwork,
    create_anytown_network,
    create_balerma_network,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Water Distribution Network Optimization with CUDA PSO"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="anytown",
        choices=["anytown", "balerma", "synthetic", "custom"],
        help="Network to optimize: anytown, balerma, synthetic, or custom",
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
    parser.add_argument("--subswarms", type=int, default=4, help="Number of sub-swarms")
    parser.add_argument(
        "--output", type=str, default="results.npz", help="Output file for results"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def map_to_commercial(diameters):
    """Snap continuous diameters to the nearest commercial pipe size."""
    idx = np.abs(COMMERCIAL_DIAMETERS_MM[:, None] - diameters).argmin(axis=0)
    return COMMERCIAL_DIAMETERS_MM[idx]


def create_fitness_function(network):
    """PSO objective: minimize feasible design cost on commercial diameters."""

    def fitness(diameters):
        discrete = map_to_commercial(np.asarray(diameters, dtype=float))
        value, _ = network.evaluate_fitness(discrete)
        return float(value)

    return fitness


def load_network(args):
    if args.network == "anytown":
        return create_anytown_network()
    if args.network == "balerma":
        return create_balerma_network()
    if args.network == "custom":
        if os.path.exists(args.inp_file):
            return WaterNetwork().load_epanet_inp(args.inp_file)
        print(f"Warning: file {args.inp_file} not found. Using Anytown instead.")
        return create_anytown_network()

    return WaterNetwork().create_synthetic_network(num_nodes=20)


def create_optimizer(args):
    if args.subswarms > 1:
        pso = ParallelPSO(
            n_particles=args.particles,
            n_iterations=args.iterations,
            n_sub_swarms=args.subswarms,
            w=0.7,
            c1=1.5,
            c2=1.5,
        )
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

    return pso


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Smart Water Distribution Network Optimization")
    print("Using CUDA Particle Swarm Optimization (PSO)")
    print("=" * 70)

    print("\n[1/5] Loading water distribution network...")
    network = load_network(args)
    network.print_summary()

    original_diameters = map_to_commercial(network.get_original_diameters())
    original_fitness, original_details = network.evaluate_fitness(original_diameters)

    print("\n[2/5] Setting up optimization parameters...")
    bounds = network.get_diameter_bounds()
    fitness_func = create_fitness_function(network)
    print(f"Decision variables (pipe diameters): {bounds.shape[0]}")
    print(f"Diameter range: {bounds[0, 0]:.1f} - {bounds[0, 1]:.1f} mm")
    print(f"Original diameters: {np.mean(original_diameters):.1f} mm (avg)")

    print("\n[3/5] Initializing CUDA PSO...")
    pso = create_optimizer(args)

    print("\n[4/5] Running optimization...")
    start_time = time.time()
    best_diameters, _ = pso.optimize(fitness_func, bounds, verbose=args.verbose)
    optimization_time = time.time() - start_time
    best_diameters = map_to_commercial(best_diameters)
    best_fitness, details = network.evaluate_fitness(best_diameters)

    print(f"\nOptimization completed in {optimization_time:.2f} seconds")

    print("\n[5/5] Results and Analysis...")
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Network: {network.network_name}")
    print(f"Best Feasible Objective: {best_fitness:,.2f}")
    print("\nOptimal Pipe Diameters:")
    print("-" * 60)
    for i, diameter in enumerate(best_diameters):
        original = original_diameters[i]
        change = ((diameter - original) / original) * 100.0 if original else 0.0
        direction = "↑" if change > 0 else "↓" if change < 0 else "="
        print(
            f"  Pipe {i + 1:2d}: {diameter:7.1f} mm | Original: {original:7.1f} mm | {direction}{abs(change):5.1f}%"
        )

    original_cost = original_details["cost"]
    optimized_cost = details["cost"]
    improvement = 0.0
    if original_cost > 0.0:
        improvement = ((original_cost - optimized_cost) / original_cost) * 100.0

    print("\nCost Comparison:")
    print("-" * 60)
    print(f"  Original Design Cost: ${original_cost:,.2f}")
    print(f"  Optimized Design Cost: ${optimized_cost:,.2f}")
    print(f"  Cost Reduction: {improvement:.2f}%")

    print("\nHydraulic Analysis:")
    print("-" * 60)
    print(f"  Pressure Penalty: {details['pressure_penalty']:,.2f}")
    print(f"  Minimum Pressure: {details['min_pressure']:.2f} m")
    print(f"  Maximum Pressure: {details['max_pressure']:.2f} m")

    print("\nNode Pressures:")
    pressures = details["pressures"]
    node_names = details["node_names"]
    for node_name, pressure in list(zip(node_names, pressures))[:10]:
        status = "OK" if pressure >= network.min_pressure else "LOW"
        print(f"  Node {node_name:>4}: Pressure={pressure:7.2f} m | {status}")
    if len(pressures) > 10:
        print(f"  ... and {len(pressures) - 10} more nodes")

    convergence = pso.get_convergence_history()
    np.savez(
        args.output,
        best_diameters=best_diameters,
        best_fitness=best_fitness,
        convergence_history=convergence,
        optimization_time=optimization_time,
        original_cost=original_cost,
        optimized_cost=optimized_cost,
        min_pressure=details["min_pressure"],
        max_pressure=details["max_pressure"],
        network_config={
            "num_nodes": network.num_nodes,
            "num_pipes": network.num_pipes,
            "network_name": network.network_name,
            "original_diameters": original_diameters,
        },
    )

    print(f"\nResults saved to: {args.output}")
    print("\nTo visualize results, run: python3 visualize.py --input results.npz")
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
