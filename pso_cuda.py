import warnings

import numpy as np

warnings.filterwarnings("ignore")

try:
    from numba import cuda
except ImportError:  # pragma: no cover - runtime fallback
    cuda = None


if cuda is not None:

    @cuda.jit
    def pso_update_kernel(
        positions,
        velocities,
        personal_best_pos,
        global_best_pos,
        bounds,
        r1,
        r2,
        w,
        c1,
        c2,
    ):
        tid = cuda.grid(1)
        n_particles = positions.shape[0]
        dim = positions.shape[1]

        if tid < n_particles:
            for d in range(dim):
                cognitive = (
                    c1 * r1[tid, d] * (personal_best_pos[tid, d] - positions[tid, d])
                )
                social = c2 * r2[tid, d] * (global_best_pos[d] - positions[tid, d])
                velocities[tid, d] = w * velocities[tid, d] + cognitive + social
                positions[tid, d] = positions[tid, d] + velocities[tid, d]

                lo = bounds[d, 0]
                hi = bounds[d, 1]
                if positions[tid, d] < lo:
                    positions[tid, d] = lo
                    velocities[tid, d] *= -0.5
                elif positions[tid, d] > hi:
                    positions[tid, d] = hi
                    velocities[tid, d] *= -0.5


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
        if cuda is None:
            print("CUDA Available: No (numba not installed, using NumPy fallback)")
            self.use_cuda = False
            return

        try:
            if cuda.is_available():
                device = cuda.get_current_device()
                device_name = (
                    device.name.decode()
                    if isinstance(device.name, bytes)
                    else str(device.name)
                )
                print(f"CUDA Device: {device_name}")
                print("CUDA Available: Yes")
                self.use_cuda = True
            else:
                print("CUDA Available: No (using NumPy fallback)")
                self.use_cuda = False
        except Exception as exc:
            print(f"CUDA Available: No ({exc})")
            self.use_cuda = False

    def _initialize(self, bounds):
        dim = bounds.shape[0]
        span = bounds[:, 1] - bounds[:, 0]

        self.positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(self.n_particles, dim)
        ).astype(np.float64)
        self.velocities = np.random.uniform(
            -0.1 * span,
            0.1 * span,
            size=(self.n_particles, dim),
        ).astype(np.float64)

        self.personal_best_pos = self.positions.copy()
        self.personal_best_val = np.full(self.n_particles, np.inf, dtype=np.float64)
        self.global_best_pos = np.zeros(dim, dtype=np.float64)
        self.global_best_val = np.inf
        self.history = []

    def _evaluate_fitness(self, fitness_func):
        fitness_values = np.array(
            [fitness_func(self.positions[i]) for i in range(self.n_particles)],
            dtype=np.float64,
        )

        improved = fitness_values < self.personal_best_val
        self.personal_best_val[improved] = fitness_values[improved]
        self.personal_best_pos[improved] = self.positions[improved]

        min_idx = int(np.argmin(self.personal_best_val))
        if self.personal_best_val[min_idx] < self.global_best_val:
            self.global_best_val = float(self.personal_best_val[min_idx])
            self.global_best_pos = self.personal_best_pos[min_idx].copy()

    def _update_numpy(self, bounds):
        dim = bounds.shape[0]
        r1 = np.random.random((self.n_particles, dim))
        r2 = np.random.random((self.n_particles, dim))

        cognitive = self.c1 * r1 * (self.personal_best_pos - self.positions)
        social = self.c2 * r2 * (self.global_best_pos - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social
        self.positions = self.positions + self.velocities

        for d in range(dim):
            lo, hi = bounds[d, 0], bounds[d, 1]

            below = self.positions[:, d] < lo
            self.positions[below, d] = lo
            self.velocities[below, d] *= -0.5

            above = self.positions[:, d] > hi
            self.positions[above, d] = hi
            self.velocities[above, d] *= -0.5

    def _update_cuda(self, bounds):
        threads_per_block = 128
        blocks = (self.n_particles + threads_per_block - 1) // threads_per_block
        dim = bounds.shape[0]
        r1 = np.random.random((self.n_particles, dim)).astype(np.float64)
        r2 = np.random.random((self.n_particles, dim)).astype(np.float64)

        d_positions = cuda.to_device(self.positions)
        d_velocities = cuda.to_device(self.velocities)
        d_personal_best_pos = cuda.to_device(self.personal_best_pos)
        d_global_best_pos = cuda.to_device(self.global_best_pos)
        d_bounds = cuda.to_device(bounds.astype(np.float64))
        d_r1 = cuda.to_device(r1)
        d_r2 = cuda.to_device(r2)

        pso_update_kernel[blocks, threads_per_block](
            d_positions,
            d_velocities,
            d_personal_best_pos,
            d_global_best_pos,
            d_bounds,
            d_r1,
            d_r2,
            self.w,
            self.c1,
            self.c2,
        )
        cuda.synchronize()
        self.positions = d_positions.copy_to_host()
        self.velocities = d_velocities.copy_to_host()

    def _update(self, bounds):
        if self.use_cuda and cuda is not None:
            try:
                self._update_cuda(bounds)
                return
            except Exception:
                self.use_cuda = False
        self._update_numpy(bounds)

    def optimize(self, fitness_func, bounds, verbose=True):
        dim = bounds.shape[0]

        if verbose:
            print(f"\n{'=' * 60}")
            print("CUDA Particle Swarm Optimization")
            print(f"{'=' * 60}")
            print(f"Particles: {self.n_particles}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Dimensions: {dim}")
            print(f"Mode: {'CUDA GPU' if self.use_cuda else 'NumPy CPU'}")
            print(f"{'=' * 60}\n")

        self._initialize(bounds)
        self._evaluate_fitness(fitness_func)
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
            print("\nOptimization Complete!")
            print(f"Final Best Fitness: {self.global_best_val:.4f}")

        return self.global_best_pos, self.global_best_val

    def get_convergence_history(self):
        return np.array(self.history, dtype=np.float64)


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

        self.swarm_size = max(1, n_particles // n_sub_swarms)
        self.swarms = []
        self.global_best_pos = None
        self.global_best_val = np.inf
        self.history = []
        self.use_cuda = False
        self._check_cuda_available()

    def _check_cuda_available(self):
        base = CUDA_PSO(n_particles=1, n_iterations=1, w=self.w, c1=self.c1, c2=self.c2)
        self.use_cuda = base.use_cuda

    def optimize(self, fitness_func, bounds, verbose=True):
        dim = bounds.shape[0]
        self.global_best_val = np.inf
        self.global_best_pos = np.zeros(dim, dtype=np.float64)

        if verbose:
            print(f"\n{'=' * 60}")
            print("Parallel CUDA Particle Swarm Optimization")
            print(f"{'=' * 60}")
            print(f"Total Particles: {self.n_particles}")
            print(f"Sub-swarms: {self.n_sub_swarms}")
            print(f"Iterations: {self.n_iterations}")
            print(f"Dimensions: {dim}")
            print(f"Mode: {'CUDA GPU' if self.use_cuda else 'NumPy CPU'}")
            print(f"{'=' * 60}\n")

        self.swarms = []
        results = []
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

            if verbose:
                print(f"Optimizing sub-swarm {i + 1}/{self.n_sub_swarms}...")

            best_pos, best_val = swarm.optimize(fitness_func, bounds, verbose=False)
            results.append((best_pos, best_val, swarm.history))
            if best_val < self.global_best_val:
                self.global_best_val = float(best_val)
                self.global_best_pos = best_pos.copy()

        if results:
            max_len = max(len(history) for _, _, history in results)
            merged = []
            for i in range(max_len):
                merged.append(
                    min(history[i] for _, _, history in results if i < len(history))
                )
            self.history = merged
        else:
            self.history = []

        if verbose:
            print("\nAll sub-swarms completed!")
            print(f"Global Best Fitness: {self.global_best_val:.4f}")

        return self.global_best_pos, self.global_best_val

    def get_convergence_history(self):
        return np.array(self.history, dtype=np.float64)
