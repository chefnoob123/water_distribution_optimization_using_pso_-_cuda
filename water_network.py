import os

import networkx as nx
import numpy as np

try:
    import wntr
except ImportError:  # pragma: no cover - handled at runtime
    wntr = None


COMMERCIAL_DIAMETERS_MM = np.array(
    [50.8, 76.2, 101.6, 152.4, 203.2, 254.0, 304.8, 355.6, 406.4, 457.2, 508.0],
    dtype=float,
)
DEFAULT_COST_FACTOR = 0.4


class WaterNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.num_nodes = 0
        self.num_pipes = 0
        self.reservoir_nodes = []
        self.demand_nodes = []
        self.node_elevations = None
        self.base_demands = None
        self.pipe_data = {}
        self.network_name = "Unknown"
        self.inp_file = None
        self.reservoir_head = None
        self.roughness_default = 100.0
        self.cost_factor = DEFAULT_COST_FACTOR
        self.min_pressure = 20.0
        self.max_pressure = 80.0
        self.node_labels = []
        self.pipe_labels = []

    def create_synthetic_network(self, num_nodes=20):
        self.num_nodes = num_nodes
        self.network_name = f"Synthetic-{num_nodes}nodes"
        self.graph.clear()
        self.pipe_data = {}
        self.inp_file = None
        self.reservoir_head = 80.0

        self.graph.add_nodes_from(range(num_nodes))
        self.reservoir_nodes = [0]
        self.demand_nodes = list(range(1, num_nodes))
        self.node_labels = [str(i) for i in range(num_nodes)]
        self.node_elevations = np.zeros(num_nodes)
        self.base_demands = np.zeros(num_nodes)

        edges = []
        for i in range(1, num_nodes):
            parent = max(0, (i - 1) // 3)
            edges.append((parent, i))

        self.num_pipes = len(edges)
        self.pipe_labels = [f"P{i + 1}" for i in range(self.num_pipes)]

        for i, (u, v) in enumerate(edges):
            self.graph.add_edge(u, v)
            self.pipe_data[i] = {
                "from": u,
                "to": v,
                "length": 150.0 + np.random.rand() * 250.0,
                "diameter": 152.4,
                "roughness": 120.0,
                "original_diameter": 152.4,
            }

        for node in self.demand_nodes:
            self.base_demands[node] = 1.5 + np.random.rand() * 4.0
            self.node_elevations[node] = np.random.rand() * 30.0

        self.node_elevations[0] = 60.0
        return self

    def load_epanet_inp(self, filepath):
        self.network_name = os.path.basename(filepath).replace(".inp", "")
        self.graph.clear()
        self.pipe_data = {}
        self.inp_file = filepath

        nodes = {}
        pipes = []
        current_section = None

        with open(filepath, "r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()

                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].upper()
                    continue

                if not line or line.startswith(";"):
                    continue

                parts = line.split()
                if current_section == "JUNCTIONS" and len(parts) >= 3:
                    node_id = parts[0]
                    nodes[node_id] = {
                        "elevation": float(parts[1]),
                        "demand": float(parts[2]),
                        "is_reservoir": False,
                    }
                elif current_section == "RESERVOIRS" and len(parts) >= 2:
                    node_id = parts[0]
                    head = float(parts[1])
                    nodes[node_id] = {
                        "elevation": head,
                        "demand": 0.0,
                        "is_reservoir": True,
                    }
                    if self.reservoir_head is None:
                        self.reservoir_head = head
                elif current_section == "PIPES" and len(parts) >= 6:
                    pipes.append(
                        {
                            "id": parts[0],
                            "from": parts[1],
                            "to": parts[2],
                            "length": float(parts[3]),
                            "diameter": float(parts[4]),
                            "roughness": float(parts[5]),
                        }
                    )

        self.node_id_to_idx = {}
        self.reservoir_nodes = []
        self.demand_nodes = []
        self.node_labels = []

        for idx, (node_id, data) in enumerate(nodes.items()):
            self.node_id_to_idx[node_id] = idx
            self.node_labels.append(node_id)
            if data.get("is_reservoir", False):
                self.reservoir_nodes.append(idx)
            else:
                self.demand_nodes.append(idx)

        self.num_nodes = len(nodes)
        self.node_elevations = np.zeros(self.num_nodes)
        self.base_demands = np.zeros(self.num_nodes)
        for node_id, data in nodes.items():
            idx = self.node_id_to_idx[node_id]
            self.node_elevations[idx] = data["elevation"]
            self.base_demands[idx] = data.get("demand", 0.0)

        self.num_pipes = len(pipes)
        self.pipe_labels = []
        for i, pipe in enumerate(pipes):
            from_idx = self.node_id_to_idx.get(pipe["from"])
            to_idx = self.node_id_to_idx.get(pipe["to"])
            if from_idx is None or to_idx is None:
                continue

            self.pipe_labels.append(pipe["id"])
            self.pipe_data[i] = {
                "from": from_idx,
                "to": to_idx,
                "length": pipe["length"],
                "diameter": pipe["diameter"],
                "roughness": pipe["roughness"],
                "original_diameter": pipe["diameter"],
            }
            self.graph.add_edge(from_idx, to_idx)

        if self.reservoir_head is None and self.reservoir_nodes:
            self.reservoir_head = float(self.node_elevations[self.reservoir_nodes[0]])

        if "anytown" in self.network_name.lower():
            self.cost_factor = 0.4
            self.min_pressure = 20.0
            self.max_pressure = 80.0

        print(f"Loaded EPANET network: {self.network_name}")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Pipes: {self.num_pipes}")
        print(f"  Reservoirs: {len(self.reservoir_nodes)}")
        print(f"  Total demand: {np.sum(self.base_demands):.2f} L/s")
        return self

    def save_network(self, filename):
        np.save(
            filename,
            {
                "num_nodes": self.num_nodes,
                "num_pipes": self.num_pipes,
                "reservoir_nodes": self.reservoir_nodes,
                "demand_nodes": self.demand_nodes,
                "node_elevations": self.node_elevations,
                "base_demands": self.base_demands,
                "pipe_data": self.pipe_data,
                "network_name": self.network_name,
                "inp_file": self.inp_file,
            },
        )

    def get_pipe_lengths(self):
        return np.array(
            [self.pipe_data[i]["length"] for i in range(self.num_pipes)], dtype=float
        )

    def get_pipe_diameters(self):
        return np.array(
            [self.pipe_data[i]["diameter"] for i in range(self.num_pipes)], dtype=float
        )

    def get_original_diameters(self):
        return np.array(
            [
                self.pipe_data[i].get("original_diameter", 152.4)
                for i in range(self.num_pipes)
            ],
            dtype=float,
        )

    def map_to_commercial(self, diameters):
        idx = np.abs(COMMERCIAL_DIAMETERS_MM[:, None] - diameters).argmin(axis=0)
        return COMMERCIAL_DIAMETERS_MM[idx]

    def calculate_pipe_cost(self, diameters):
        return float(
            np.sum(
                np.asarray(diameters, dtype=float)
                * self.get_pipe_lengths()
                * self.cost_factor
            )
        )

    def _tree_hydraulic_solve(self, diameters):
        pipe_lengths = self.get_pipe_lengths()
        n = self.num_nodes
        head = np.zeros(n, dtype=float)
        head[0] = float(
            self.reservoir_head
            if self.reservoir_head is not None
            else self.node_elevations[0]
        )

        children = {i: [] for i in range(n)}
        parent = [-1] * n
        pipe_idx = [-1] * n
        for i in range(self.num_pipes):
            u = self.pipe_data[i]["from"]
            v = self.pipe_data[i]["to"]
            parent[v] = u
            children[u].append(v)
            pipe_idx[v] = i

        def subtree_demand(node):
            total = self.base_demands[node]
            for child in children[node]:
                total += subtree_demand(child)
            return total

        flows = np.zeros(self.num_pipes, dtype=float)
        for i in range(self.num_pipes):
            flows[i] = max(subtree_demand(self.pipe_data[i]["to"]), 0.001)

        for node in range(1, n):
            if parent[node] == -1:
                continue

            path_head_loss = 0.0
            curr = node
            while curr != 0 and parent[curr] != -1:
                pid = pipe_idx[curr]
                if pid == -1:
                    break
                length = pipe_lengths[pid]
                diameter_m = max(diameters[pid], 50.8) / 1000.0
                flow_m3s = max(flows[pid], 0.001) / 1000.0
                roughness = max(
                    self.pipe_data[pid].get("roughness", self.roughness_default), 1.0
                )
                head_loss = (
                    10.67
                    * length
                    * (flow_m3s**1.852)
                    / ((roughness**1.852) * (diameter_m**4.8704))
                )
                path_head_loss += head_loss
                curr = parent[curr]
            head[node] = head[0] - path_head_loss

        pressures = np.array(
            [head[node] - self.node_elevations[node] for node in self.demand_nodes],
            dtype=float,
        )
        return {
            "head": head,
            "flows": flows,
            "pressures": pressures,
            "node_names": [self.node_labels[node] for node in self.demand_nodes],
        }

    def hydraulic_solve(self, diameters):
        diameters = self.map_to_commercial(np.asarray(diameters, dtype=float))
        if wntr is None or not self.inp_file:
            return self._tree_hydraulic_solve(diameters)

        wn = wntr.network.WaterNetworkModel(self.inp_file)
        pipe_names = wn.pipe_name_list
        for i, pipe_name in enumerate(pipe_names):
            if i >= len(diameters):
                break
            wn.get_link(pipe_name).diameter = max(diameters[i], 50.8) / 1000.0

        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        pressure_frame = results.node["pressure"].loc[0, wn.junction_name_list]
        head_frame = results.node["head"].loc[0, wn.node_name_list]
        flow_frame = results.link["flowrate"].loc[0, pipe_names]

        head = np.array(
            [head_frame[node_name] for node_name in wn.node_name_list], dtype=float
        )
        pressures = pressure_frame.to_numpy(dtype=float)
        flows = np.abs(flow_frame.to_numpy(dtype=float)) * 1000.0
        return {
            "head": head,
            "flows": flows,
            "pressures": pressures,
            "node_names": list(wn.junction_name_list),
        }

    def evaluate_fitness(self, diameters):
        diameters = self.map_to_commercial(np.asarray(diameters, dtype=float))

        try:
            hydraulic = self.hydraulic_solve(diameters)
        except Exception:
            infeasible = 1e18
            return infeasible, {
                "cost": infeasible,
                "pressure_penalty": infeasible,
                "head": np.zeros(self.num_nodes, dtype=float),
                "flows": np.zeros(self.num_pipes, dtype=float),
                "pressures": np.full(len(self.demand_nodes), -1e9, dtype=float),
                "node_names": [self.node_labels[node] for node in self.demand_nodes],
                "min_pressure": -1e9,
                "max_pressure": -1e9,
            }

        cost = self.calculate_pipe_cost(diameters)
        pressures = hydraulic["pressures"]

        low_pressure = np.maximum(self.min_pressure - pressures, 0.0)
        high_pressure = np.maximum(pressures - self.max_pressure, 0.0)
        pressure_penalty = float(
            np.sum((low_pressure**2) * 50000.0) + np.sum((high_pressure**2) * 5000.0)
        )

        fitness = cost + pressure_penalty
        return fitness, {
            "cost": cost,
            "pressure_penalty": pressure_penalty,
            "head": hydraulic["head"],
            "flows": hydraulic["flows"],
            "pressures": pressures,
            "node_names": hydraulic["node_names"],
            "min_pressure": float(np.min(pressures)) if len(pressures) else 0.0,
            "max_pressure": float(np.max(pressures)) if len(pressures) else 0.0,
        }

    def get_diameter_bounds(self):
        low = COMMERCIAL_DIAMETERS_MM.min()
        high = COMMERCIAL_DIAMETERS_MM.max()
        return np.array([[low, high] for _ in range(self.num_pipes)], dtype=float)

    def print_summary(self):
        print(f"Water Distribution Network: {self.network_name}")
        print("=" * 50)
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of pipes: {self.num_pipes}")
        print(f"Reservoir nodes: {self.reservoir_nodes}")
        print(f"Demand nodes: {len(self.demand_nodes)}")
        print(f"Total demand: {np.sum(self.base_demands):.2f} L/s")
        print(f"Total pipe length: {np.sum(self.get_pipe_lengths()):.2f} m")
        print(f"Cost factor: {self.cost_factor:.3f}")


def create_anytown_network():
    network_path = os.path.join(os.path.dirname(__file__), "networks", "anytown.inp")
    if os.path.exists(network_path):
        return WaterNetwork().load_epanet_inp(network_path)

    network = WaterNetwork()
    return network.create_synthetic_network(num_nodes=22)


def create_balerma_network():
    network = WaterNetwork()
    network.network_name = "Balerma-Irrigation"
    network.num_nodes = 50
    network.num_pipes = 49
    network.reservoir_nodes = [0]
    network.demand_nodes = list(range(1, 50))
    network.reservoir_head = 90.0
    network.node_labels = [str(i) for i in range(50)]
    network.pipe_labels = [f"P{i + 1}" for i in range(49)]

    np.random.seed(42)
    network.node_elevations = np.random.uniform(0, 30, 50)
    network.node_elevations[0] = 60.0
    network.base_demands = np.zeros(50)
    network.base_demands[1:] = np.random.uniform(1.0, 4.0, 49)

    network.pipe_data = {}
    for i in range(49):
        network.pipe_data[i] = {
            "from": i,
            "to": i + 1,
            "length": 200.0 + np.random.rand() * 300.0,
            "diameter": 152.4,
            "roughness": 120.0,
            "original_diameter": 152.4,
        }
        network.graph.add_edge(i, i + 1)

    network.cost_factor = 0.35
    network.min_pressure = 15.0
    network.max_pressure = 80.0
    return network
