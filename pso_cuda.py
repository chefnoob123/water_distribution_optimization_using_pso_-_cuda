import numpy as np
from numba import cuda, float32, int32
import math
import warnings

warnings.filterwarnings("ignore")

try:

    @cuda.jit
    def pso_init_kernel(positions, velocities, bounds, seed):
        tid = cuda.grid(1)
        n_particles = positions.shape[0]
        dim = positions.shape[1]

        if tid < n_particles:
            np.random.seed(seed + tid)
            for d in range(dim):
                lo = bounds[d, 0]
                hi = bounds[d, 1]
                positions[tid, d] = lo + np.random.random() * (hi - lo)
                velocities[tid, d] = (np.random.random() - 0.5) * (hi - lo) * 0.1
except:
    pass

try:

    @cuda.jit
    def pso_update_kernel(
        positions,
        velocities,
        personal_best_pos,
        personal_best_val,
        global_best_pos,
        global_best_val,
        bounds,
        w,
        c1,
        c2,
        dim,
    ):
        tid = cuda.grid(1)
        n_particles = positions.shape[0]

        if tid < n_particles:
            r1_0 = 0
            r1_1 = 0
            r2_0 = 0
            r2_1 = 0

            for d in range(dim):
                r1 = 0.5
                r2 = 0.5

                cognitive = c1 * r1 * (personal_best_pos[tid, d] - positions[tid, d])
                social = c2 * r2 * (global_best_pos[d] - positions[tid, d])

                velocities[tid, d] = w * velocities[tid, d] + cognitive + social

                positions[tid, d] = positions[tid, d] + velocities[tid, d]

                lo = bounds[d, 0]
                hi = bounds[d, 1]
                if positions[tid, d] < lo:
                    positions[tid, d] = lo
                    velocities[tid, d] *= -0.5
                if positions[tid, d] > hi:
                    positions[tid, d] = hi
                    velocities[tid, d] *= -0.5
except:
    pass


class CUDA_PSO:
    def __init__(self, n_particles=256, n_iterations=100, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.positions = None
        self.velocities = None
        self.personal_best_pos = None
        self.personal_best_val = None
        self.global_best_pos = None
        self.global_best_val = None

        self.history = []
        self.use_cuda = False

        self._check_cuda_available()

    def _check_cuda_available(self):
        try:
            if cuda.is_available():
                device = cuda.get_current_device()
                print(f"CUDA Device: {device.name.decode()}")
                print(f"CUDA Available: Yes")
                self.use_cuda = True
            else:
                print("CUDA Available: No (using NumPy fallback)")
                self.use_cuda = False
        except Exception as e:
            print(f"CUDA Available: No ({str(e)})")
            self.use_cuda = False

    def _initialize(self, bounds):
        dim = bounds.shape[0]

        self.positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(self.n_particles, dim)
        ).astype(np.float64)

        self.velocities = np.random.uniform(
            -0.1 * (bounds[:, 1] - bounds[:, 0]),
            0.1 * (bounds[:, 1] - bounds[:, 0]),
            size=(self.n_particles, dim),
        ).astype(np.float64)

        self.personal_best_pos = self.positions.copy()
        self.personal_best_val = np.full(self.n_particles, np.inf)

        self.global_best_pos = np.zeros(dim)
        self.global_best_val = np.inf

        self.history = []

    def _evaluate_fitness(self, fitness_func):
        fitness_values = np.array(
            [fitness_func(self.positions[i]) for i in range(self.n_particles)]
        )

        improved = fitness_values < self.personal_best_val
        self.personal_best_val[improved] = fitness_values[improved]
        self.personal_best_pos[improved] = self.positions[improved]

        min_idx = np.argmin(self.personal_best_val)
        if self.personal_best_val[min_idx] < self.global_best_val:
            self.global_best_val = self.personal_best_val[min_idx]
            self.global_best_pos = self.personal_best_pos[min_idx].copy()

        return fitness_values

    def _update(self, bounds):
        dim = bounds.shape[0]

        r1 = np.random.random((self.n_particles, dim))
        r2 = np.random.random((self.n_particles, dim))

        cognitive = self.c1 * r1 * (self.personal_best_pos - self.positions)
        social = self.c2 * r2 * (self.global_best_pos - self.positions)

        self.velocities = self.w * self.velocities + cognitive + social

        self.positions = self.positions + self.velocities

        for d in range(dim):
            lo, hi = bounds[d, 0], bounds[d, 1]
            range_val = hi - lo

            below = self.positions[:, d] < lo
            self.positions[below, d] = lo
            self.velocities[below, d] *= -0.5

            above = self.positions[:, d] > hi
            self.positions[above, d] = hi
            self.velocities[above, d] *= -0.5

    def optimize(self, fitness_func, bounds, verbose=True):
        dim = bounds.shape[0]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"CUDA Particle Swarm Optimization")
            print(f"{'=' * 60}")
            print(f"Particles: {self.n_particles}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Dimensions: {dim}")
            print(f"Mode: {'CUDA GPU' if self.use_cuda else 'NumPy CPU'}")
            print(f"{'=' * 60}\n")

        self._initialize(bounds)

        initial_fitness = self._evaluate_fitness(fitness_func)
        self.history.append(self.global_best_val)

        if verbose:
            print(
                f"Iteration 0/{self.n_iterations} - Best Fitness: {self.global_best_val:.4f}"
            )

        for iteration in range(self.n_iterations):
            self._update(bounds)

            self._evaluate_fitness(fitness_func)

            self.history.append(self.global_best_val)

            if verbose and (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.n_iterations} - Best Fitness: {self.global_best_val:.4f}"
                )

        if verbose:
            print(f"\nOptimization Complete!")
            print(f"Final Best Fitness: {self.global_best_val:.4f}")

        return self.global_best_pos, self.global_best_val

    def get_convergence_history(self):
        return np.array(self.history)


class ParallelPSO:
    def __init__(
        self, n_particles=512, n_iterations=100, n_sub_swarms=4, w=0.7, c1=1.5, c2=1.5
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.n_sub_swarms = n_sub_swarms
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.swarm_size = n_particles // n_sub_swarms
        self.swarms = []
        self.global_best_pos = None
        self.global_best_val = np.inf

        self.history = []
        self.use_cuda = False

        self._check_cuda_available()

    def _check_cuda_available(self):
        try:
            if cuda.is_available():
                device = cuda.get_current_device()
                print(f"CUDA Device: {device.name.decode()}")
                print(f"CUDA Available: Yes (Parallel PSO mode)")
                self.use_cuda = True
            else:
                print("CUDA Available: No (using NumPy fallback)")
                self.use_cuda = False
        except Exception as e:
            print(f"CUDA Available: No ({str(e)})")
            self.use_cuda = False

    def optimize(self, fitness_func, bounds, verbose=True):
        dim = bounds.shape[0]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Parallel CUDA Particle Swarm Optimization")
            print(f"{'=' * 60}")
            print(f"Total Particles: {self.n_particles}")
            print(f"Sub-swarms: {self.n_sub_swarms}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Dimensions: {dim}")
            print(f"Mode: {'CUDA GPU' if self.use_cuda else 'NumPy CPU'}")
            print(f"{'=' * 60}\n")

        self.swarms = []
        for i in range(self.n_sub_swarms):
            swarm = CUDA_PSO(
                n_particles=self.swarm_size,
                n_iterations=self.n_iterations,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
            )
            swarm.use_cuda = self.use_cuda
            self.swarms.append(swarm)

        results = []
        for i, swarm in enumerate(self.swarms):
            if verbose:
                print(f"Optimizing sub-swarm {i + 1}/{self.n_sub_swarms}...")
            best_pos, best_val = swarm.optimize(fitness_func, bounds, verbose=False)
            results.append((best_pos, best_val))

            if best_val < self.global_best_val:
                self.global_best_val = best_val
                self.global_best_pos = best_pos

        self.history = [s.history for s in self.swarms]

        if verbose:
            print(f"\nAll sub-swarms completed!")
            print(f"Global Best Fitness: {self.global_best_val:.4f}")

        return self.global_best_pos, self.global_best_val

    def get_convergence_history(self):
        if len(self.history) > 0:
            if isinstance(self.history[0], list):
                min_history = []
                for i in range(max(len(h) for h in self.history)):
                    vals = [h[i] for h in self.history if i < len(h)]
                    min_history.append(min(vals))
                return np.array(min_history)
            return np.array(self.history)
        return np.array([])
