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

import torch

from diff_gpmp2.gpmp2.factor import Factor
from diff_gpmp2.utils import mat_utils


class GPFactor(Factor):

    def __init__(
        self,
        name: str,
        weight: float,
        dof: int,
        device: torch.device,
        threshold: float,
        batch_size: int = 1,
        error_sum_weight: float = 1.0,
    ):
        super().__init__(name, threshold, error_sum_weight)
        self.device = device
        self.batch_size = batch_size
        self.dof = dof
        self.state_dim = self.dof * 2  # position+velocity
        self.weight = weight

        self.delta_t = None
        self.total_time_sec = None
        self.Q_c_inv = None
        self.Q_inv = None

    def to(self, device: torch.device) -> "GPFactor":
        self.device = device
        if self.delta_t is not None:
            self.delta_t = self.delta_t.to(device)
        if self.Q_c_inv is not None:
            self.Q_c_inv = self.Q_c_inv.to(device)
        if self.Q_inv is not None:
            self.Q_inv = self.Q_inv.to(device)
        return self

    def threshold_reached(self, error: torch.Tensor) -> bool:
        """error of all points on trajectory must be within threshold

        This ensures that the velocity change is within a threshold and the position change does not deviate
        too much from the velocity given
        """
        return torch.all(torch.norm(error, dim=-1) <= self.threshold)

    def admissible_solution(self, error: torch.Tensor) -> bool:
        """whether error is good enough to be a valid solution"""
        return True

    def calc_phi(self):
        I = torch.eye(self.dof, device=self.device)
        Z = torch.zeros(self.dof, self.dof, device=self.device)
        phi_u = torch.cat((I, self.delta_t * I), dim=1)
        # phi_u = torch.cat((I, Z), dim=1)
        phi_l = torch.cat((Z, I), dim=1)
        phi = torch.cat((phi_u, phi_l), dim=0)
        return phi

    def calc_Q_inv_batch(self):
        assert self.delta_t is not None, "make sure to call set_total_time_sec() first"
        m1 = 12.0 * (self.delta_t**-3.0) * self.Q_c_inv
        m2 = -6.0 * (self.delta_t**-2.0) * self.Q_c_inv
        m3 = 4.0 * (self.delta_t**-1.0) * self.Q_c_inv

        Q_inv_u = torch.cat((m1, m2), dim=-1)
        Q_inv_l = torch.cat((m2, m3), dim=-1)
        Q_inv = torch.cat((Q_inv_u, Q_inv_l), dim=-2)
        return Q_inv

    def _get_error(self, thb: torch.Tensor, verbose: bool = False):
        # FIXME: penelizes velocity changes quite hard i think. This sometimes limits the obstacle avoidance capabilities i think
        phi = self.calc_phi()
        state_1 = thb[:, :-1]
        state_2 = thb[:, 1:]
        error = state_2 - state_1 @ phi.T
        # error[..., 6:] = torch.exp(error[..., 6:])

        # H1_full and H2_full form isotropic matrix; H2_full blocks right to H1_full blocks
        phi = self.calc_phi().unsqueeze(0).repeat(thb.shape[0], 1, 1)
        H1_full = phi.unsqueeze(1).repeat(1, self.__get_num_gp_factors(), 1, 1)
        H2_full = -1.0 * torch.eye(self.state_dim, device=self.device).unsqueeze(0).unsqueeze(0).repeat(
            thb.shape[0], self.__get_num_gp_factors(), 1, 1
        )
        H = torch.concat((H1_full, H2_full), dim=3)

        # verbose = True
        if False:  # verbose:
            error_vis = torch.abs(error[0].T.detach().cpu())
            pos_error = torch.sum(error_vis[:6], dim=0)
            vel_error = torch.sum(error_vis[6:], dim=0)
            from matplotlib import pyplot as plt

            plt.figure()
            # plt.plot(pos_error, c='red', label='pos')
            plt.plot(error_vis[6:].T, label=list(range(6)))  # , c='blue', label='vel')
            plt.legend()
            plt.title("Error: Pos %.2f Vel %.2f" % (torch.sum(pos_error), torch.sum(vel_error)))
            # plt.savefig("tmp-error.png")
            plt.show()
            plt.figure()
            # state_diff = torch.norm((state_2 - state_1)[0, :, :6], dim=-1).detach().cpu()
            # plt.plot(state_diff, label='pos-diff')
            # vel = torch.norm(state_1[0, :, 6:], dim=-1).detach().cpu()
            # plt.plot(vel, label='vel')
            # plt.title("Trajectory position change and velocity")
            # plt.legend()
            # # plt.savefig("tmp-diff.png")
            # plt.show()

        return error.unsqueeze(-1), H, self.Q_inv

    def set_total_time_sec(self, time: torch.Tensor):
        self.total_time_sec = time
        self.__update_delta_t()

    def __get_num_gp_factors(self) -> int:
        # gp factors equals segments between traj points
        return self._get_num_traj_states() - 1

    def initialize(self, num_traj_states: int):
        super().initialize(num_traj_states)
        self.__update_delta_t()
        self.update_inv_cov()

    def __update_delta_t(self):
        if self.total_time_sec is not None and self._get_num_traj_states() is not None:
            # how long each segment should take in seconds
            self.delta_t = self.total_time_sec / (self.__get_num_gp_factors() + 1)

    def number_of_constraints(self) -> int:
        return self.__get_num_gp_factors() * self.state_dim

    def get_inv_cov(self):
        return self.Q_inv

    def update_inv_cov(self):
        q_c_inv = (
            mat_utils.isotropic_matrix(1.0 / torch.pow(self.weight, 2.0), self.dof, self.device)
            .unsqueeze(0)
            .repeat(self.batch_size, self.__get_num_gp_factors(), 1, 1)
        )
        self.Q_c_inv = q_c_inv
        self.Q_inv = self.calc_Q_inv_batch()

    def set_inv_cov(self, Q_inv):
        self.Q_inv = Q_inv

    def get_masks(self, columns: int):
        A = torch.zeros((self.number_of_constraints(), columns), dtype=bool)
        K = torch.zeros((self.number_of_constraints(), self.number_of_constraints()), dtype=bool)

        for i in range(self.__get_num_gp_factors()):
            A[
                i * self.state_dim : (i + 1) * self.state_dim,
                i * self.state_dim : (i + 2) * self.state_dim,
            ] = True
            K[
                i * self.state_dim : (i + 1) * self.state_dim,
                i * self.state_dim : (i + 1) * self.state_dim,
            ] = True

        return A, K
