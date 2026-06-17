"""
Tests for LieGroupVaritationalIntegrator in src/rbnn_flow/utils/integrator.py.

Accuracy tests verify:
  - cayley_transx  : output is a valid SO(3) rotation matrix
  - calc_fc_init   : output is finite and correctly shaped
  - calc_Ac        : residual at the converged fc is ~0
  - calc_grad_Ac   : analytical Jacobian matches finite-difference approximation
  - optimize_fc    : Newton solver converges (last two iterates are close)
  - step           : output orientation stays on SO(3)
  - integrate      : SO(3) constraint and kinetic energy are preserved

Speed tests time each public method and assert generous wall-time limits.

Run `python tests/test_utils/test_integrator.py` to generate the accuracy plot.
"""

import os
import sys
import time

import numpy as np
import pytest
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.rbnn_flow.utils.integrator import LieGroupVaritationalIntegrator

# ── constants ─────────────────────────────────────────────────────────────────

DTYPE = torch.float64
BATCH = 8
TIMESTEP = 1e-2
TRAJ_LEN = 200
TOL_SO3 = 1e-8      # ||R^T R - I||_F tolerance
TOL_ENERGY = 1e-3   # relative kinetic energy drift (LGVI conserves a modified H)

# ── shared helpers ────────────────────────────────────────────────────────────

def _make_integrator() -> LieGroupVaritationalIntegrator:
    return LieGroupVaritationalIntegrator(precision=DTYPE)


def _make_moi(batch: int = 1) -> torch.Tensor:
    """Diagonal inertia tensor I = diag(2, 5, 10), shape [batch, 3, 3]."""
    I = torch.diag(torch.tensor([2.0, 5.0, 10.0], dtype=DTYPE))
    return I.unsqueeze(0).expand(batch, -1, -1).clone()


def _random_rotation(batch: int) -> torch.Tensor:
    """Uniformly random SO(3) matrices via QR decomposition."""
    Q, _ = torch.linalg.qr(torch.randn(batch, 3, 3, dtype=DTYPE))
    signs = torch.linalg.det(Q).sign().unsqueeze(-1).unsqueeze(-1)
    return Q * signs


def _random_pi(batch: int) -> torch.Tensor:
    return torch.randn(batch, 3, dtype=DTYPE)


def _kinetic_energy(pi: torch.Tensor, moi: torch.Tensor) -> torch.Tensor:
    """T = 0.5 * pi^T @ I^{-1} @ pi, returns shape [B]."""
    moi_inv = torch.linalg.inv(moi)
    return 0.5 * (pi.unsqueeze(-2) @ moi_inv @ pi.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def _so3_error(R: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of R^T @ R - I per matrix, shape [B]."""
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    return torch.linalg.matrix_norm(R.mT @ R - eye, ord="fro")


def _simple_V(q: torch.Tensor) -> torch.Tensor:
    """V(q) = 0.5 * ||q||^2. Accepts float32, returns [B, 1]."""
    return 0.5 * (q ** 2).sum(-1, keepdim=True)


# ── accuracy tests ─────────────────────────────────────────────────────────────

class TestCayleyTransx:

    def test_output_shape(self):
        integ = _make_integrator()
        fc = torch.randn(BATCH, 3, dtype=DTYPE)
        assert integ.cayley_transx(fc).shape == (BATCH, 3, 3)

    def test_orthogonality(self):
        """F^T @ F should equal I to double precision."""
        integ = _make_integrator()
        fc = torch.randn(BATCH, 3, dtype=DTYPE) * 0.5
        F = integ.cayley_transx(fc)
        err = _so3_error(F)
        assert torch.all(err < TOL_SO3), f"max ||F^T F - I||_F = {err.max():.2e}"

    def test_determinant_one(self):
        """det(F) should be +1 for every item in the batch."""
        integ = _make_integrator()
        fc = torch.randn(BATCH, 3, dtype=DTYPE) * 0.5
        F = integ.cayley_transx(fc)
        dets = torch.linalg.det(F)
        assert torch.allclose(dets, torch.ones(BATCH, dtype=DTYPE), atol=1e-9), \
            f"det values: {dets}"

    def test_zero_fc_gives_identity(self):
        """Cayley map of the zero vector is the identity matrix."""
        integ = _make_integrator()
        fc = torch.zeros(BATCH, 3, dtype=DTYPE)
        F = integ.cayley_transx(fc)
        eye = torch.eye(3, dtype=DTYPE).expand(BATCH, -1, -1)
        assert torch.allclose(F, eye, atol=1e-12)

    def test_antisymmetry(self):
        """cayley_transx(-fc) should equal cayley_transx(fc)^T (inverse)."""
        integ = _make_integrator()
        fc = torch.randn(BATCH, 3, dtype=DTYPE) * 0.3
        F = integ.cayley_transx(fc)
        F_inv = integ.cayley_transx(-fc)
        err = torch.linalg.matrix_norm(F @ F_inv - torch.eye(3, dtype=DTYPE), ord="fro")
        assert torch.all(err < TOL_SO3), f"max |F(-fc) - F(fc)^T|: {err.max():.2e}"


class TestCalcFcInit:

    def test_output_shape(self):
        integ = _make_integrator()
        a_vec = torch.randn(BATCH, 3, dtype=DTYPE) * 0.1
        fc = integ.calc_fc_init(a_vec, _make_moi(BATCH))
        assert fc.shape == (BATCH, 3)

    def test_finite(self):
        integ = _make_integrator()
        a_vec = torch.randn(BATCH, 3, dtype=DTYPE) * 0.1
        fc = integ.calc_fc_init(a_vec, _make_moi(BATCH))
        assert not fc.isnan().any() and not fc.isinf().any()

    def test_zero_a_gives_zero_fc(self):
        """With a_vec = 0, the initial guess fc should also be 0."""
        integ = _make_integrator()
        a_vec = torch.zeros(BATCH, 3, dtype=DTYPE)
        fc = integ.calc_fc_init(a_vec, _make_moi(BATCH))
        assert torch.allclose(fc, torch.zeros_like(fc), atol=1e-12)


class TestCalcAc:

    def test_output_shape(self):
        integ = _make_integrator()
        a_vec = torch.randn(BATCH, 3, dtype=DTYPE) * 0.1
        moi = _make_moi(BATCH)
        fc = integ.calc_fc_init(a_vec, moi)
        assert integ.calc_Ac(a_vec, moi, fc).shape == (BATCH, 3)

    def test_converged_fc_gives_near_zero_residual(self):
        """
        After optimize_fc, the residual Ac(a, fc*) should be ~0.
        This verifies that calc_Ac and optimize_fc are mutually consistent.
        """
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)

        fc_list = integ.optimize_fc(R_vec=R, pi_vec=pi, moi=moi, V=None,
                                    timestep=TIMESTEP, tol=1e-10)
        fc_opt = fc_list[-1]
        a_vec = TIMESTEP * pi  # M_vec = 0 when V = None
        Ac = integ.calc_Ac(a_vec, moi, fc_opt)

        residual_norms = Ac.norm(dim=-1)
        assert torch.all(residual_norms < 1e-7), \
            f"max |Ac(fc*)| = {residual_norms.max():.2e}"

    def test_batched_vs_single(self):
        """calc_Ac on a batch must equal stacking individual results."""
        integ = _make_integrator()
        a_vec = torch.randn(BATCH, 3, dtype=DTYPE) * 0.1
        moi = _make_moi(BATCH)
        fc = integ.calc_fc_init(a_vec, moi) * 0.5

        Ac_batch = integ.calc_Ac(a_vec, moi, fc)
        for i in range(BATCH):
            Ac_i = integ.calc_Ac(a_vec[i:i+1], moi[i:i+1], fc[i:i+1])
            assert torch.allclose(Ac_batch[i], Ac_i[0], atol=1e-12)


class TestCalcGradAc:

    def test_output_shape(self):
        integ = _make_integrator()
        a_vec = torch.randn(BATCH, 3, dtype=DTYPE) * 0.1
        moi = _make_moi(BATCH)
        fc = integ.calc_fc_init(a_vec, moi)
        assert integ.calc_grad_Ac(a_vec, moi, fc).shape == (BATCH, 3, 3)

    def test_matches_finite_difference(self):
        """
        Analytical Jacobian dAc/dfc should match a central-difference approximation.
        This is the key correctness check for the Newton step in optimize_fc.
        """
        integ = _make_integrator()
        B = 3
        a_vec = torch.randn(B, 3, dtype=DTYPE) * 0.1
        moi = _make_moi(B)
        fc = integ.calc_fc_init(a_vec, moi) * 0.3

        eps = 1e-5
        J_analytical = integ.calc_grad_Ac(a_vec, moi, fc)

        J_fd = torch.zeros(B, 3, 3, dtype=DTYPE)
        for j in range(3):
            fc_p = fc.clone(); fc_p[:, j] += eps
            fc_m = fc.clone(); fc_m[:, j] -= eps
            J_fd[:, :, j] = (integ.calc_Ac(a_vec, moi, fc_p) -
                             integ.calc_Ac(a_vec, moi, fc_m)) / (2 * eps)

        max_err = (J_analytical - J_fd).abs().max().item()
        assert max_err < 1e-5, f"max FD error: {max_err:.2e}"


class TestOptimizeFc:

    def test_returns_nonempty_list(self):
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH)
        moi = _make_moi(BATCH)
        fc_list = integ.optimize_fc(R_vec=R, pi_vec=pi, moi=moi, V=None, timestep=TIMESTEP)
        assert isinstance(fc_list, list) and len(fc_list) >= 2

    def test_convergence_gap_small(self):
        """Last two Newton iterates must be within tol of each other."""
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        fc_list = integ.optimize_fc(R_vec=R, pi_vec=pi, moi=moi, V=None,
                                    timestep=TIMESTEP, tol=1e-10)
        diff = torch.linalg.norm(fc_list[-1] - fc_list[-2], dim=-1)
        assert torch.all(diff < 1e-8), f"max convergence gap: {diff.max():.2e}"

    def test_warm_start_fewer_iterations(self):
        """Passing the previous fc_list as a warm start should converge in fewer steps."""
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)

        cold = integ.optimize_fc(R_vec=R, pi_vec=pi, moi=moi, V=None, timestep=TIMESTEP)
        warm = integ.optimize_fc(R_vec=R, pi_vec=pi, moi=moi, V=None,
                                 timestep=TIMESTEP, fc_list=[cold[-1]])
        assert len(warm) <= len(cold)

    def test_with_potential(self):
        """optimize_fc should still converge when V is provided."""
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.3
        moi = _make_moi(BATCH)
        fc_list = integ.optimize_fc(R_vec=R, pi_vec=pi, moi=moi, V=_simple_V,
                                    timestep=TIMESTEP)
        assert isinstance(fc_list, list) and len(fc_list) >= 2


class TestStep:

    def test_output_shapes(self):
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        R_ii, pi_ii, fc_list = integ.step(R_i=R, pi_i=pi, moi=moi, V=None, timestep=TIMESTEP)
        assert R_ii.shape == (BATCH, 3, 3)
        assert pi_ii.shape == (BATCH, 3)
        assert isinstance(fc_list, list)

    def test_output_is_so3(self):
        """R_ii should satisfy R^T R = I and det(R) = +1."""
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        R_ii, _, _ = integ.step(R_i=R, pi_i=pi, moi=moi, V=None, timestep=TIMESTEP)

        err = _so3_error(R_ii)
        assert torch.all(err < TOL_SO3), f"max SO(3) error: {err.max():.2e}"

        dets = torch.linalg.det(R_ii)
        assert torch.allclose(dets, torch.ones(BATCH, dtype=DTYPE), atol=1e-9)

    def test_finite_output(self):
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        R_ii, pi_ii, _ = integ.step(R_i=R, pi_i=pi, moi=moi, V=None, timestep=TIMESTEP)
        assert not R_ii.isnan().any() and not R_ii.isinf().any()
        assert not pi_ii.isnan().any() and not pi_ii.isinf().any()

    def test_small_timestep_small_change(self):
        """With a tiny timestep, the orientation should barely move."""
        integ = _make_integrator()
        R = torch.eye(3, dtype=DTYPE).unsqueeze(0)
        pi = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
        moi = _make_moi(1)
        R_ii, _, _ = integ.step(R_i=R, pi_i=pi, moi=moi, V=None, timestep=1e-4)
        err = torch.linalg.matrix_norm(R_ii - R, ord="fro").max().item()
        assert err < 1e-3, f"orientation changed too much for tiny dt: {err:.2e}"


class TestIntegrate:

    def test_output_shapes(self):
        integ = _make_integrator()
        R0 = _random_rotation(BATCH)
        pi0 = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        T = 10
        R_traj, pi_traj = integ.integrate(pi_init=pi0, R_init=R0, moi=moi, V=None,
                                          timestep=TIMESTEP, traj_len=T)
        assert R_traj.shape == (BATCH, T, 3, 3)
        assert pi_traj.shape == (BATCH, T, 3)

    def test_first_frame_matches_init(self):
        """The first frame of the trajectory must equal the initial conditions."""
        integ = _make_integrator()
        R0 = _random_rotation(BATCH)
        pi0 = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        R_traj, pi_traj = integ.integrate(pi_init=pi0, R_init=R0, moi=moi, V=None,
                                          timestep=TIMESTEP, traj_len=5)
        assert torch.allclose(R_traj[:, 0], R0.double(), atol=1e-12)
        assert torch.allclose(pi_traj[:, 0], pi0.double(), atol=1e-12)

    def test_so3_preservation(self):
        """Every R in the trajectory must remain on SO(3)."""
        integ = _make_integrator()
        R0 = _random_rotation(BATCH)
        pi0 = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        R_traj, _ = integ.integrate(pi_init=pi0, R_init=R0, moi=moi, V=None,
                                    timestep=TIMESTEP, traj_len=TRAJ_LEN)
        err = _so3_error(R_traj.reshape(-1, 3, 3))
        assert torch.all(err < TOL_SO3), \
            f"max SO(3) error over trajectory: {err.max():.2e}"

    def test_energy_conservation_free_body(self):
        """
        Kinetic energy T = 0.5 * pi^T I^{-1} pi should be near-constant.
        LGVI is symplectic so energy drift is bounded (not growing), but not
        exactly zero — we allow TOL_ENERGY relative drift.
        """
        integ = _make_integrator()
        R0 = torch.eye(3, dtype=DTYPE).unsqueeze(0)
        pi0 = torch.tensor([[1.0, 0.5, 0.25]], dtype=DTYPE)
        moi = _make_moi(1)
        _, pi_traj = integ.integrate(pi_init=pi0, R_init=R0, moi=moi, V=None,
                                     timestep=TIMESTEP, traj_len=TRAJ_LEN)
        E0 = _kinetic_energy(pi_traj[:, 0], moi)
        max_rel_err = max(
            ((_kinetic_energy(pi_traj[:, t], moi) - E0).abs() / E0.abs()).item()
            for t in range(1, TRAJ_LEN)
        )
        assert max_rel_err < TOL_ENERGY, \
            f"max relative energy drift: {max_rel_err:.2e}"

    def test_traj_len_one_is_just_init(self):
        """traj_len=1 should return a trajectory containing only the initial state."""
        integ = _make_integrator()
        R0 = _random_rotation(BATCH)
        pi0 = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        R_traj, pi_traj = integ.integrate(pi_init=pi0, R_init=R0, moi=moi, V=None,
                                          timestep=TIMESTEP, traj_len=1)
        assert R_traj.shape == (BATCH, 1, 3, 3)
        assert torch.allclose(R_traj[:, 0], R0.double(), atol=1e-12)


# ── speed tests ────────────────────────────────────────────────────────────────

class TestSpeed:
    """
    Wall-time benchmarks. Limits are generous to avoid flaky CI failures.
    Print statements show actual timings for manual inspection.
    """

    REPEATS = 5

    def _avg_ms(self, fn) -> float:
        fn()  # warm up
        t0 = time.perf_counter()
        for _ in range(self.REPEATS):
            fn()
        return (time.perf_counter() - t0) / self.REPEATS * 1e3

    def test_cayley_transx_small_batch(self):
        integ = _make_integrator()
        fc = torch.randn(8, 3, dtype=DTYPE)
        ms = self._avg_ms(lambda: integ.cayley_transx(fc))
        print(f"\n  cayley_transx B=8:   {ms:.2f} ms")
        assert ms < 100

    def test_cayley_transx_large_batch(self):
        integ = _make_integrator()
        fc = torch.randn(256, 3, dtype=DTYPE)
        ms = self._avg_ms(lambda: integ.cayley_transx(fc))
        print(f"\n  cayley_transx B=256: {ms:.2f} ms")
        assert ms < 500

    def test_optimize_fc_speed(self):
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        ms = self._avg_ms(lambda: integ.optimize_fc(
            R_vec=R, pi_vec=pi, moi=moi, V=None, timestep=TIMESTEP))
        print(f"\n  optimize_fc B={BATCH}:    {ms:.2f} ms")
        assert ms < 2000

    def test_step_speed(self):
        integ = _make_integrator()
        R = _random_rotation(BATCH)
        pi = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        ms = self._avg_ms(lambda: integ.step(
            R_i=R, pi_i=pi, moi=moi, V=None, timestep=TIMESTEP))
        print(f"\n  step B={BATCH}:           {ms:.2f} ms")
        assert ms < 5000

    def test_integrate_speed(self):
        integ = _make_integrator()
        R0 = _random_rotation(BATCH)
        pi0 = _random_pi(BATCH) * 0.5
        moi = _make_moi(BATCH)
        ms = self._avg_ms(lambda: integ.integrate(
            pi_init=pi0, R_init=R0, moi=moi, V=None, timestep=TIMESTEP, traj_len=50))
        print(f"\n  integrate T=50 B={BATCH}: {ms:.2f} ms")
        assert ms < 60_000


# ── accuracy plot ──────────────────────────────────────────────────────────────

def plot_accuracy(save_path: str = "tests/test_utils/integrator_accuracy.png"):
    """
    Integrates a free rigid body and plots:
      - Panel 1: SO(3) orthogonality error ||R^T R - I||_F over time
      - Panel 2: Relative kinetic energy drift |E(t)-E(0)|/|E(0)| over time
      - Panel 3: Wall time (ms) vs batch size for cayley_transx and integrate
    """
    integ = _make_integrator()
    moi = _make_moi(1)
    R0 = torch.eye(3, dtype=DTYPE).unsqueeze(0)
    pi0 = torch.tensor([[1.0, 0.5, 0.25]], dtype=DTYPE)
    T, dt = 500, 5e-3

    R_traj, pi_traj = integ.integrate(pi_init=pi0, R_init=R0, moi=moi,
                                      V=None, timestep=dt, traj_len=T)

    times = np.arange(T) * dt
    so3_err = _so3_error(R_traj[0]).detach().numpy()           # [T]
    E = np.array([_kinetic_energy(pi_traj[:, t], moi).item()
                  for t in range(T)])
    E_rel = np.abs(E - E[0]) / np.abs(E[0])
    E_rel = np.maximum(E_rel, 1e-16)                           # floor for log scale

    # speed sweep
    batch_sizes = [1, 4, 8, 16, 32, 64]
    repeats = 3
    ms_cayley, ms_integrate = [], []
    for B_s in batch_sizes:
        fc_b = torch.randn(B_s, 3, dtype=DTYPE)
        R0_b = torch.eye(3, dtype=DTYPE).unsqueeze(0).expand(B_s, -1, -1).clone()
        pi0_b = torch.ones(B_s, 3, dtype=DTYPE) * 0.5
        moi_b = _make_moi(B_s)

        t0 = time.perf_counter()
        for _ in range(repeats):
            integ.cayley_transx(fc_b)
        ms_cayley.append((time.perf_counter() - t0) / repeats * 1e3)

        t0 = time.perf_counter()
        for _ in range(repeats):
            integ.integrate(pi_init=pi0_b, R_init=R0_b, moi=moi_b,
                            V=None, timestep=dt, traj_len=50)
        ms_integrate.append((time.perf_counter() - t0) / repeats * 1e3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].semilogy(times, np.maximum(so3_err, 1e-16))
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$\|R^\top R - I\|_F$")
    axes[0].set_title("SO(3) Constraint Error")
    axes[0].grid(True, which="both", alpha=0.4)

    axes[1].semilogy(times, E_rel)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$|E(t) - E(0)|\,/\,|E(0)|$")
    axes[1].set_title("Relative Kinetic Energy Drift")
    axes[1].grid(True, which="both", alpha=0.4)

    axes[2].plot(batch_sizes, ms_cayley, "o-", label="cayley_transx")
    axes[2].plot(batch_sizes, ms_integrate, "s-", label="integrate (T=50)")
    axes[2].set_xlabel("Batch size")
    axes[2].set_ylabel("Wall time (ms)")
    axes[2].set_title("Speed vs Batch Size")
    axes[2].legend()
    axes[2].grid(True, alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    return fig


class TestAccuracyPlot:
    def test_plot_saves_file(self, tmp_path):
        path = str(tmp_path / "integrator_accuracy.png")
        fig = plot_accuracy(save_path=path)
        assert os.path.exists(path)
        plt.close(fig)


# ── entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_accuracy()
    plt.show()
