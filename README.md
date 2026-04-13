# Smart Water Distribution Network Optimization using CUDA PSO

GPU-accelerated Particle Swarm Optimization for optimizing water distribution network pipe diameters.

## Datasets Supported

### 1. Synthetic Network (Default)
Randomly generated tree-like network for testing.
```bash
python main.py --network synthetic
```

### 2. Anytown Benchmark Network
Classic 22-node benchmark network from water distribution literature.
- **Nodes**: 22
- **Pipes**: 21
- **Total Demand**: ~299 L/s
- **Location**: `networks/anytown.inp`

### 3. Balerma Irrigation Network
Larger 50-node irrigation network.
```bash
python main.py --network balerma
```

### 4. Custom EPANET Network
Load any EPANET `.inp` file:
```bash
python main.py --network custom --inp-file path/to/network.inp
```

## Quick Start

```bash
# Install dependencies
pip install numpy numba matplotlib networkx

# Run optimization
python main.py

# Visualize results
python visualize.py
```

## Usage Examples

```bash
# Synthetic network (default)
python main.py

# Anytown benchmark
python main.py --network anytown

# Balerma irrigation network
python main.py --network balerma

# Custom parameters
python main.py --nodes 25 --particles 512 --iterations 150

# Force CPU mode
python main.py --no-cuda

# Verbose output
python main.py --verbose
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--network` | Network type (synthetic/anytown/balerma/custom) | synthetic |
| `--inp-file` | Path to EPANET .inp file | networks/anytown.inp |
| `--particles` | Number of PSO particles | 256 |
| `--iterations` | Number of iterations | 100 |
| `--subswarms` | Number of parallel sub-swarms | 4 |
| `--output` | Output file | results.npz |
| `--no-cuda` | Disable CUDA | False |
| `--verbose` | Show progress | False |

## Output Files

- `results.npz` - Optimization results
- `plots/convergence.png` - PSO convergence plot
- `plots/network_topology.png` - Network layout
- `plots/hydraulic_results.png` - Pressure/flow analysis
- `plots/comparison.png` - Before/after comparison

## EPANET Network Format

The `.inp` file should contain:
```
[JUNCTIONS]
;ID    Elev    Demand
1      213.4   9.8
...

[RESERVOIRS]
;ID    Head
0      260.0

[PIPES]
;ID    Node1    Node2    Length    Diameter    Roughness
1      0        1        914.4     406.4      100
...
```

## Algorithm

**Particle Swarm Optimization (PSO)** minimizes:
- **Pipe Cost**: Based on diameter and length
- **Pressure Penalty**: Ensures minimum 10m pressure at demand nodes

**CUDA Acceleration**: Uses Numba JIT compilation for GPU acceleration when available.

## Example Output

```
Network: Synthetic-20nodes
Best Fitness: 28,607,196
Cost Reduction: 21.1%

Optimal Pipe Diameters:
  Pipe 1: 167.6mm (↑67.6% from original)
  Pipe 2: 104.9mm (↑4.9%)
  ...

All node pressures: OK (>10m)
```
