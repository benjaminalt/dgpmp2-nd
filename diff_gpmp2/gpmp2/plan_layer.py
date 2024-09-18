"""
This file is part of DGPMP2-ND    

Copyright (C) 2024 ArtiMinds Robotics GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

-------------------------
Part of this code is based on dGPMP2

Copyright (c) 2020, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of mosquitto nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import logging

import torch
import torch.nn as nn

from diff_gpmp2.gpmp2.factors import Factors
from diff_gpmp2.gpmp2.gp.prior_factor import PriorFactor

logger = logging.getLogger(__name__)


class GPParams:

    def __init__(
        self,
        Q_c_inv: torch.Tensor,
        K_s: float,
        K_g: float,
        K_obs: float,
        K_gp: float,
        K_lims: float,
    ) -> None:
        self.Q_c_inv = Q_c_inv
        self.K_s = torch.tensor(K_s)
        self.K_g = torch.tensor(K_g)
        self.K_obs = torch.tensor(K_obs)
        self.K_gp = torch.tensor(K_gp)
        self.K_lims = torch.tensor(K_lims)


class OptimParams:

    def __init__(
        self,
        method: str,
        reg: 0.1,
        init_lr: float,
        max_iters: int,
        error_patience: int,
    ) -> None:
        self.method = method
        self.reg = reg
        self.init_lr = init_lr
        self.max_iters = max_iters
        self.error_patience = error_patience


class PlanLayer(nn.Module):

    def __init__(
        self,
        factors: Factors,
        state_dim: int,
        optim_params: OptimParams,
        device: torch.device,
        batch_size=1,
        verbose=False,
    ):
        super(PlanLayer, self).__init__()
        self.device = device
        self.reg = optim_params.reg
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.factors = factors
        self.verbose = verbose

    def to(self, device: torch.device):
        self.device = device
        self.factors.to(device)
        return self

    def forward(self, thb):
        """
        Arguments
        ===
        thb: torch.tensor (batch_size, num_traj_states, state_dim)
        """
        last_error, A, b, K = self.construct_linear_system_batch(thb)
        dthetab = self.solve_linear_system_batch(A, b, K, delta=self.reg)
        return dthetab, last_error

    def construct_linear_system_batch(self, thb):
        # reset matrizes with zero, they are class attributes to speedup creation
        M = self.factors.get_num_constraints()
        N = self.factors.get_N()
        # A = torch.zeros(self.batch_size, M, N, device=self.device)
        # b = torch.zeros(self.batch_size, M, 1, device=self.device)
        # K = torch.zeros(self.batch_size, M, M, device=self.device)
        As = []
        Ks = []
        bs = []

        errors = {}

        offset = 0
        for factor, masks in zip(self.factors.get_factors(), self.factors.get_masks()):
            consts = factor.number_of_constraints()

            err, H, inv_cov = factor.get_error(thb)
            errors[factor.name] = err.detach()

            # TODO: nice to exclude factors that are null, but might make matrix singular -> crash
            if torch.any(err != 0) or isinstance(factor, PriorFactor):
                A = torch.zeros(self.batch_size, consts, N, device=self.device)
                K = torch.zeros(self.batch_size, consts, M, device=self.device)
                As.append(A.masked_scatter_(masks["A"], H))
                bs.append(err.view((1, -1, 1)))
                K[:, :, offset : offset + consts].masked_scatter_(masks["K"], inv_cov)
                Ks.append(K)

                offset += consts

        A = torch.concat(As, dim=1)
        K = torch.concat(Ks, dim=1)
        b = torch.concat(bs, dim=1)

        K = K[:, :, :offset]  # some factors may be skipped, remove those from K

        return errors, A, b, K

    def solve_linear_system_batch(self, A: torch.Tensor, b: torch.Tensor, K: torch.Tensor, delta=0.0):
        """
         Solves for δθ:
                (H^{T} * Σ^{-1} * H) δθ = - H^{T} * Σ^{-1} * h(θ^{i})
        Paper references following equation:
                (K^{-1} + H^{T} * Σ^{-1} * H) δθ = -K^{-1} * (θ^i - μ) - H^{T} * Σ^{-1} * h(θ^{i})

                paper splits (K^{-1} + H^{T} * Σ^{-1} * H) into D*D^{T} with cholesky decomposition,
                then solves for δθ with δθ = D^{-1} * (- H^{T} * Σ^{-1} * h(θ^{i})) * D^{-T}.
                Thats slower because two inverse matrix operations are required.

                we combine the gp covariance and likelihood covariance into one. Thats why we don't have
                extra terms vor K^{-1} and θ^i - μ, but define in the gp factor:
                    H^{T} * Σ^{-1} := K^{-1}
                    h(θ^{i}) := θ^i - μ

         with:
            delta * I                 :regularization term
            H        := A             :Jacobian of cost function for θ=θ^i
            Σ^{-1}   := K             :inverse covariance of the likelihood (likelihood function captures planning requirements)
            h(θ^{i}) := b             :vector-valued linearized cost function (errors of the different factors)

        Taylor expansion of cost function: h(θ) = h(θ^i) + H*δθ

        Implementation differs from DGPMP2 paper by omitting -K^{-1} * (θ^i - μ), which is the GP prior.
            It actually does not. It just differs from the formula used in the dGPMP2 paper.
        """
        N = self.factors.get_N()
        I = torch.eye(N, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        A_t_K = A.transpose(1, 2) @ K
        u_t_u = A_t_K @ A + delta * I  # left side of linear system
        A_t_K_b = A_t_K @ b  # right side of equation
        dtheta = torch.linalg.solve(u_t_u, A_t_K_b)

        return dtheta.view(self.batch_size, self.factors.get_num_traj_states(), self.state_dim)

    def error_batch(self, thb, verbose: bool = False):
        """Return the non-linear normalized error for the factor graph at the current trajectory"""
        return self.factors.compute_error_batch(thb, verbose=verbose)
