import numpy as np
import networkx as nx


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

    def create_synthetic_network(self, num_nodes=20):
        """Create a random tree-like network (original method)"""
        self.num_nodes = num_nodes
        self.network_name = f"Synthetic-{num_nodes}nodes"
        self.graph.clear()

        self.graph.add_nodes_from(range(num_nodes))

        self.reservoir_nodes = [0]
        self.demand_nodes = list(range(1, num_nodes))
        self.node_elevations = np.zeros(num_nodes)
        self.base_demands = np.zeros(num_nodes)

        edges = []
        for i in range(1, num_nodes):
            parent = max(0, (i - 1) // 3)
            edges.append((parent, i))

        self.num_pipes = len(edges)

        for i, (u, v) in enumerate(edges):
            self.graph.add_edge(u, v)
            self.pipe_data[i] = {
                "from": u,
                "to": v,
                "length": 100.0 + np.random.rand() * 200.0,
                "diameter": 100.0,
                "roughness": 100.0,
            }

        for node in self.demand_nodes:
            if node < num_nodes:
                self.base_demands[node] = 10.0 + np.random.rand() * 20.0
                self.node_elevations[node] = np.random.rand() * 30.0

        self.node_elevations[0] = 100.0

        return self

    def load_epanet_inp(self, filepath):
        """Load network from EPANET .inp file"""
        self.network_name = filepath.split("/")[-1].replace(".inp", "")
        self.graph.clear()
        self.pipe_data = {}

        nodes = {}
        pipes = []

        current_section = None

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()

                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].upper()
                    continue

                if not line or line.startswith(";"):
                    continue

                if current_section == "JUNCTIONS":
                    parts = line.split()
                    if len(parts) >= 3:
                        node_id = parts[0]
                        elevation = float(parts[1])
                        demand = float(parts[2]) if len(parts) > 2 else 0.0
                        nodes[node_id] = {"elevation": elevation, "demand": demand}

                elif current_section == "RESERVOIRS":
                    parts = line.split()
                    if len(parts) >= 2:
                        node_id = parts[0]
                        head = float(parts[1])
                        nodes[node_id] = {
                            "elevation": head,
                            "demand": 0.0,
                            "is_reservoir": True,
                        }

                elif current_section == "PIPES":
                    parts = line.split()
                    if len(parts) >= 6:
                        pipe_id = parts[0]
                        from_node = parts[1]
                        to_node = parts[2]
                        length = float(parts[3])
                        diameter = float(parts[4])
                        roughness = float(parts[5])
                        pipes.append(
                            {
                                "id": pipe_id,
                                "from": from_node,
                                "to": to_node,
                                "length": length,
                                "diameter": diameter,
                                "roughness": roughness,
                            }
                        )

        self.node_id_to_idx = {}

        idx = 0
        self.reservoir_nodes = []
        self.demand_nodes = []

        for node_id, data in nodes.items():
            if data.get("is_reservoir", False):
                self.reservoir_nodes.append(idx)
            else:
                self.demand_nodes.append(idx)
            self.node_id_to_idx[node_id] = idx
            idx += 1

        self.num_nodes = len(nodes)
        self.node_elevations = np.zeros(self.num_nodes)
        self.base_demands = np.zeros(self.num_nodes)

        for node_id, data in nodes.items():
            idx = self.node_id_to_idx[node_id]
            self.node_elevations[idx] = data["elevation"]
            self.base_demands[idx] = data.get("demand", 0.0)

        self.num_pipes = len(pipes)

        for i, pipe in enumerate(pipes):
            from_idx = self.node_id_to_idx.get(pipe["from"])
            to_idx = self.node_id_to_idx.get(pipe["to"])
            if from_idx is not None and to_idx is not None:
                self.pipe_data[i] = {
                    "from": from_idx,
                    "to": to_idx,
                    "length": pipe["length"],
                    "diameter": pipe["diameter"],
                    "roughness": pipe["roughness"],
                    "original_diameter": pipe["diameter"],
                }
                self.graph.add_edge(from_idx, to_idx)

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
            },
        )

    def get_pipe_lengths(self):
        return np.array([self.pipe_data[i]["length"] for i in range(self.num_pipes)])

    def get_pipe_diameters(self):
        return np.array([self.pipe_data[i]["diameter"] for i in range(self.num_pipes)])

    def get_original_diameters(self):
        return np.array(
            [
                self.pipe_data[i].get("original_diameter", 100.0)
                for i in range(self.num_pipes)
            ]
        )

    def hydraulic_solve(self, diameters):
        pipe_lengths = self.get_pipe_lengths()
        elevations = self.node_elevations.copy()

        n = self.num_nodes
        head = np.zeros(n)
        head[0] = elevations[0] + 50.0

        parent = [-1] * n
        for i in range(self.num_pipes):
            u = self.pipe_data[i]["from"]
            v = self.pipe_data[i]["to"]
            parent[v] = u

        pipe_idx = [-1] * n
        for i in range(self.num_pipes):
            v = self.pipe_data[i]["to"]
            pipe_idx[v] = i

        for node in range(n):
            if node == 0 or parent[node] == -1:
                continue

            path_head_loss = 0.0
            curr = node
            while curr != 0 and parent[curr] != -1:
                pid = pipe_idx[curr]
                if pid == -1:
                    break
                L = pipe_lengths[pid]
                D = diameters[pid]
                Q = self.base_demands[curr]
                D_mm = max(D, 1.0)
                hf = L * Q / (D_mm**4 + 1.0)
                path_head_loss += hf
                curr = parent[curr]

            head[node] = head[0] - path_head_loss

        flows = np.zeros(self.num_pipes)
        for i in range(self.num_pipes):
            v = self.pipe_data[i]["to"]
            flows[i] = max(self.base_demands[v], 0.001)

        return head, flows

    def evaluate_fitness(self, diameters):
        head, flows = self.hydraulic_solve(diameters)

        pipe_lengths = self.get_pipe_lengths()
        pipe_cost = np.sum(diameters * pipe_lengths * 100.0)

        pressure_penalty = 0.0
        min_pressure = 15.0
        for node in self.demand_nodes:
            if node < len(head):
                pressure = head[node] - self.node_elevations[node]
                if pressure < min_pressure:
                    pressure_penalty += (min_pressure - pressure) * 1000.0
                elif pressure > 60.0:
                    pressure_penalty += (pressure - 60.0) * 100.0

        fitness = pipe_cost + pressure_penalty

        return fitness, {
            "cost": pipe_cost,
            "pressure_penalty": pressure_penalty,
            "velocity_penalty": 0.0,
            "head": head,
            "flows": flows,
        }

    def get_diameter_bounds(self):
        return np.array([[50.0, 500.0] for _ in range(self.num_pipes)])

    def print_summary(self):
        print(f"Water Distribution Network: {self.network_name}")
        print(f"=" * 50)
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of pipes: {self.num_pipes}")
        print(f"Reservoir nodes: {self.reservoir_nodes}")
        print(f"Demand nodes: {len(self.demand_nodes)}")
        print(f"Total demand: {np.sum(self.base_demands):.2f} m3/h")
        print(f"Total pipe length: {np.sum(self.get_pipe_lengths()):.2f} m")


def create_anytown_network():
    """Create the classic Anytown benchmark network"""
    network = WaterNetwork()
    network.network_name = "Anytown-Benchmark"

    network.num_nodes = 22
    network.reservoir_nodes = [0]
    network.demand_nodes = list(range(1, 22))

    network.node_elevations = np.array(
        [
            213.4,
            213.4,
            219.5,
            222.5,
            225.6,
            228.6,
            231.7,
            234.7,
            237.8,
            240.8,
            243.8,
            246.9,
            249.9,
            253.0,
            256.0,
            259.1,
            225.6,
            228.6,
            231.7,
            231.7,
            234.7,
            237.8,
        ]
    )

    network.base_demands = np.array(
        [
            0.0,
            9.8,
            9.4,
            9.6,
            9.8,
            10.0,
            10.2,
            10.4,
            10.6,
            10.8,
            11.0,
            11.2,
            11.4,
            11.6,
            11.8,
            12.0,
            22.7,
            22.9,
            23.1,
            23.3,
            23.5,
            23.7,
        ]
    )

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (2, 16),
        (16, 17),
        (17, 18),
        (4, 19),
        (19, 20),
        (20, 21),
    ]

    network.num_pipes = len(edges)

    diameters = [
        406.4,
        304.8,
        254.0,
        254.0,
        203.2,
        203.2,
        152.4,
        152.4,
        101.6,
        101.6,
        101.6,
        101.6,
        101.6,
        101.6,
        76.2,
        203.2,
        152.4,
        152.4,
        152.4,
        101.6,
        101.6,
    ]

    network.pipe_data = {}
    for i, (u, v) in enumerate(edges):
        network.pipe_data[i] = {
            "from": u,
            "to": v,
            "length": 457.2,
            "diameter": diameters[i],
            "roughness": 100.0,
            "original_diameter": diameters[i],
        }
        network.graph.add_edge(u, v)

    return network


def create_balerma_network():
    """Create the Balerma irrigation network (simplified)"""
    network = WaterNetwork()
    network.network_name = "Balerma-Irrigation"

    network.num_nodes = 50
    network.num_pipes = 49
    network.reservoir_nodes = [0]
    network.demand_nodes = list(range(1, 50))

    np.random.seed(42)
    network.node_elevations = np.random.uniform(0, 30, 50)
    network.node_elevations[0] = 50.0

    network.base_demands = np.zeros(50)
    network.base_demands[1:] = np.random.uniform(5, 20, 49)

    network.pipe_data = {}
    for i in range(49):
        u = i
        v = i + 1
        network.pipe_data[i] = {
            "from": u,
            "to": v,
            "length": 200 + np.random.rand() * 300,
            "diameter": 150.0,
            "roughness": 100.0,
            "original_diameter": 150.0,
        }
        network.graph.add_edge(u, v)

    return network
