#!/usr/bin/env python3
"""
Visualization and Analysis of Water Distribution Network Optimization Results
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PSO Optimization Results")
    parser.add_argument(
        "--input", type=str, default="results.npz", help="Input results file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots", help="Output directory for plots"
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    return parser.parse_args()


def plot_convergence(history, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(history.shape) > 1:
        for i, h in enumerate(history):
            ax.plot(h, alpha=0.5, label=f"Swarm {i + 1}")
        ax.plot(np.min(history, axis=0), "k-", linewidth=2, label="Global Best")
    else:
        ax.plot(history, "b-", linewidth=2)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Fitness (Cost)", fontsize=12)
    ax.set_title("PSO Convergence History", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


def plot_network(network_config, best_diameters, output_path):
    num_nodes = network_config["num_nodes"]
    elevations = network_config["node_elevations"]
    demands = network_config["base_demands"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    pos = {}
    levels = int(np.ceil(np.sqrt(num_nodes)))
    for i in range(num_nodes):
        row = i // levels
        col = i % levels
        pos[i] = (col * 2, -row * 2)

    ax = axes[0]
    for i in range(len(pos)):
        node_pos = pos[i]
        color = "green" if i == 0 else ("red" if demands[i] > 0 else "blue")
        size = 200 if i == 0 else (100 if demands[i] > 0 else 50)
        ax.scatter(*node_pos, c=color, s=size, zorder=5)
        ax.annotate(
            f"N{i}\n{elevations[i]:.1f}m",
            node_pos,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    pipe_idx = 0
    for i in range(1, num_nodes):
        parent = max(0, i // 3)
        if parent < num_nodes and i < num_nodes:
            width = 2 + best_diameters[pipe_idx] * 10
            ax.plot(
                [pos[parent][0], pos[i][0]],
                [pos[parent][1], pos[i][1]],
                "b-",
                linewidth=width,
                alpha=0.7,
            )
            pipe_idx += 1

    ax.set_title("Water Distribution Network Topology", fontsize=14)
    ax.set_xlabel("Distance (units)", fontsize=12)
    ax.set_ylabel("Distance (units)", fontsize=12)
    ax.grid(True, alpha=0.3)

    legend_elements = [
        plt.scatter([], [], c="green", s=100, label="Reservoir"),
        plt.scatter([], [], c="red", s=80, label="Demand Node"),
        plt.scatter([], [], c="blue", s=50, label="Intermediate"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax = axes[1]
    pipe_idx = 0
    for i in range(1, num_nodes):
        parent = max(0, i // 3)
        if parent < num_nodes and pipe_idx < len(best_diameters):
            ax.bar(
                pipe_idx + 1,
                best_diameters[pipe_idx] * 1000,
                color="steelblue",
                alpha=0.7,
            )
            pipe_idx += 1

    ax.set_xlabel("Pipe Index", fontsize=12)
    ax.set_ylabel("Optimal Diameter (mm)", fontsize=12)
    ax.set_title("Optimized Pipe Diameters", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


def plot_hydraulic_results(network_config, results, output_path):
    num_nodes = network_config["num_nodes"]
    elevations = network_config["node_elevations"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    node_ids = np.arange(num_nodes)
    ax.bar(node_ids - 0.2, elevations, 0.4, label="Elevation", color="brown", alpha=0.7)
    ax.bar(
        node_ids + 0.2,
        results["head"],
        0.4,
        label="Hydraulic Head",
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel("Node ID", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    ax.set_title("Node Elevations and Hydraulic Head", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    pressures = results["head"] - elevations
    colors = ["green" if 10 <= p <= 50 else "red" for p in pressures]
    ax.bar(node_ids, pressures, color=colors, alpha=0.7)
    ax.axhline(y=10, color="orange", linestyle="--", label="Min Pressure")
    ax.axhline(y=50, color="purple", linestyle="--", label="Max Pressure")
    ax.set_xlabel("Node ID", fontsize=12)
    ax.set_ylabel("Pressure (m)", fontsize=12)
    ax.set_title("Node Pressures (Green = Valid Range)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(results["flows"]) > 0:
        pipe_ids = np.arange(len(results["flows"]))
        ax.bar(pipe_ids, results["flows"], color="cyan", alpha=0.7)
        ax.set_xlabel("Pipe Index", fontsize=12)
        ax.set_ylabel("Flow Rate (m³/h)", fontsize=12)
        ax.set_title("Pipe Flow Rates", fontsize=14)
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    metrics = ["Cost", "Pressure\nPenalty", "Velocity\nPenalty"]
    values = [results["cost"], results["pressure_penalty"], results["velocity_penalty"]]
    colors = ["steelblue", "orange", "purple"]
    bars = ax.bar(metrics, values, color=colors, alpha=0.7)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Optimization Fitness Components", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


def plot_comparison(initial_fitness, optimized_fitness, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Initial", "Optimized"]
    values = [initial_fitness, optimized_fitness]
    colors = ["coral", "seagreen"]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.5)

    improvement = (initial_fitness - optimized_fitness) / initial_fitness * 100

    ax.set_ylabel("Fitness (Cost)", fontsize=12)
    ax.set_title(f"Optimization Improvement: {improvement:.1f}%", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


def main():
    global args
    args = parse_args()

    print("\n" + "=" * 60)
    print("Water Distribution Network Optimization - Visualization")
    print("=" * 60)

    if not os.path.exists(args.input):
        print(f"Error: Results file '{args.input}' not found!")
        print("Please run main.py first to generate optimization results.")
        return

    data = np.load(args.input, allow_pickle=True)

    best_diameters = data["best_diameters"]
    best_fitness = data["best_fitness"]
    convergence = data["convergence_history"]
    opt_time = data["optimization_time"]
    network_config = data["network_config"].item()

    print(f"\nLoaded results from: {args.input}")
    print(f"Optimization time: {opt_time:.2f} seconds")
    print(f"Best fitness: {best_fitness:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating plots...")

    plot_convergence(convergence, os.path.join(args.output_dir, "convergence.png"))

    plot_network(
        network_config,
        best_diameters,
        os.path.join(args.output_dir, "network_topology.png"),
    )

    from water_network import WaterNetwork

    network = WaterNetwork()
    network.create_synthetic_network(num_nodes=network_config["num_nodes"])
    _, details = network.evaluate_fitness(best_diameters)

    plot_hydraulic_results(
        network_config, details, os.path.join(args.output_dir, "hydraulic_results.png")
    )

    random_diameters = np.random.uniform(0.05, 0.5, len(best_diameters))
    initial_fitness, _ = network.evaluate_fitness(random_diameters)

    plot_comparison(
        initial_fitness, best_fitness, os.path.join(args.output_dir, "comparison.png")
    )

    print("\n" + "=" * 60)
    print(f"All plots saved to: {args.output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
