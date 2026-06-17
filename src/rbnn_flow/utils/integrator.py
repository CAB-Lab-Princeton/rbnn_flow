from __future__ import annotations

import torch

from src.rbnn_flow.utils.maths import skew


class LieGroupVaritationalIntegrator:
    """Lie Group Variational Integrator for SO(3) rigid body dynamics."""

    def __init__(self, precision: torch.dtype = torch.float64):
        self.precision = precision

    def calc_M(self, R: torch.Tensor, V) -> torch.Tensor:
        assert not R.isnan().any() and not R.isinf().any()

        bs = R.shape[0]
        q = R.reshape(bs, 9).detach().requires_grad_(True)

        with torch.enable_grad():
            assert not q.isnan().any()
            V_q = V(q.float())
            assert not V_q.isnan().any() and not V_q.isinf().any()
            dV = torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
            assert not dV.isnan().any() and not dV.isinf().any()

        dV = dV.reshape(bs, 3, 3)
        SM = dV.mT @ R - R.mT @ dV
        assert not SM.isnan().any() and not SM.isinf().any()
        return torch.stack((SM[..., 2, 1], SM[..., 0, 2], SM[..., 1, 0]), dim=-1)

    def cayley_transx(self, fc: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(3, dtype=self.precision, device=fc.device)
        S = skew(fc)
        return (eye + S) @ torch.linalg.inv(eye - S)

    def calc_fc_init(self, a_vec: torch.Tensor, moi: torch.Tensor) -> torch.Tensor:
        return (torch.linalg.inv(2 * moi - skew(a_vec)) @ a_vec.unsqueeze(-1)).squeeze(-1)

    def calc_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        dot_af = (a_vec * fc).sum(-1, keepdim=True)  # [B, 1]
        return (a_vec
                + (skew(a_vec) @ fc.unsqueeze(-1)).squeeze(-1)
                + dot_af * fc
                - 2 * (moi @ fc.unsqueeze(-1)).squeeze(-1))

    def calc_grad_Ac(self, a_vec: torch.Tensor, moi: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(3, dtype=self.precision, device=a_vec.device)
        dot_af = (a_vec * fc).sum(-1)[:, None, None]  # [B, 1, 1] — broadcasts over eye
        return skew(a_vec) + dot_af * eye + torch.einsum('bi, bj -> bij', fc, a_vec) - 2 * moi

    def optimize_fc(self, R_vec: torch.Tensor, pi_vec: torch.Tensor, moi: torch.Tensor, V,
                    fc_list: list = [], timestep: float = 1e-3,
                    max_iter: int = 100, tol: float = 1e-8) -> list:
        M_vec = torch.zeros_like(pi_vec) if V is None else self.calc_M(R=R_vec, V=V)
        a_vec = timestep * (pi_vec + 0.5 * timestep * M_vec)

        if not fc_list:
            fc_list.append(self.calc_fc_init(a_vec=a_vec, moi=moi))

        eps = torch.ones(fc_list[-1].shape[0], dtype=self.precision)

        for _ in range(max_iter):
            if not torch.any(eps > tol):
                break
            fc_i = fc_list[-1]
            Ac = self.calc_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            grad_Ac = self.calc_grad_Ac(a_vec=a_vec, moi=moi, fc=fc_i)
            fc_ii = fc_i - (torch.linalg.inv(grad_Ac) @ Ac.unsqueeze(-1)).squeeze(-1)
            eps = torch.linalg.norm(fc_ii - fc_i, dim=-1)
            fc_list.append(fc_ii)

        return fc_list

    def step(self, R_i: torch.Tensor, pi_i: torch.Tensor, moi: torch.Tensor, V,
             fc_list: list = [], timestep: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor, list]:
        fc_list = self.optimize_fc(R_vec=R_i, pi_vec=pi_i, moi=moi,
                                   timestep=timestep, fc_list=fc_list, V=V)
        F_i = self.cayley_transx(fc=fc_list[-1])
        R_ii = R_i @ F_i

        if V is None:
            M_i = torch.zeros_like(pi_i)
            M_ii = torch.zeros_like(pi_i)
        else:
            M_i = self.calc_M(R=R_i, V=V)
            M_ii = self.calc_M(R=R_ii, V=V)

        half_dt = 0.5 * timestep
        Ft = F_i.mT
        pi_ii = (Ft @ (pi_i + half_dt * M_i).unsqueeze(-1)).squeeze(-1) + half_dt * M_ii
        return R_ii, pi_ii, fc_list

    def integrate(self, pi_init: torch.Tensor, R_init: torch.Tensor, moi: torch.Tensor, V,
                  timestep: float = 1e-3, traj_len: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        pi_list = [pi_init.double()]
        R_list = [R_init.double()]

        for _ in range(1, traj_len):
            R_ii, pi_ii, _ = self.step(R_i=R_list[-1], pi_i=pi_list[-1],
                                       moi=moi, V=V, fc_list=[], timestep=timestep)
            R_list.append(R_ii)
            pi_list.append(pi_ii)

        return torch.stack(R_list, dim=1), torch.stack(pi_list, dim=1)
    

def _make_intergrator(precision: torch.dtype) -> LieGroupVaritationalIntegrator:
    return LieGroupVaritationalIntegrator(precision=precision)
