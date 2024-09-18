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
"""

from typing import Optional

import torch

from diff_gpmp2.differentiable_robot_model.diff_robot_model import DiffRobotModel
from diff_gpmp2.gpmp2.factor import Factor
from diff_gpmp2.utils import mat_utils


class PriorTrajectoryFactor(Factor):
    def __init__(
        self,
        name: str,
        state_dim: int,
        wksp_dim: int,
        robot_model: DiffRobotModel,
        K: torch.Tensor,
        device: torch.device,
        threshold: float,
        meanb: Optional[torch.Tensor] = None,
        batch_size=1,
    ):
        """
        Parameters
        ===
            thb_point_idx: which point the prior should account for (0 -> start point, -1 -> goal point)
        """
        super().__init__(name, threshold)
        self.state_dim = state_dim
        self.wksp_dim = wksp_dim
        self.device = device
        self.robot_model = robot_model
        self.batch_size = batch_size
        self.K = K
        self.set_mean(meanb)

    def initialize(self, num_traj_states: int):
        super().initialize(num_traj_states)
        self.inv_cov = mat_utils.isotropic_matrix(1.0 / torch.pow(self.K, 2.0), self.robot_model.n_dofs, self.device)[
            None, None
        ].repeat(self.batch_size, num_traj_states, 1, 1)

    def threshold_reached(self, error: torch.Tensor) -> bool:
        return torch.norm(error) <= self.threshold

    def admissible_solution(self, error: torch.Tensor) -> bool:
        return True

    def _get_error(self, thb: torch.Tensor, verbose: bool = False):
        assert thb.shape[0] == 1
        positions_wkspb = thb[0, :, : self.robot_model.n_dofs]
        Jfksb = torch.eye(self.robot_model.n_dofs)[None, None].repeat((1, thb.shape[1], 1, 1))

        # TODO: currently, it will only respect th intersect of the two trajectories. Interpolation would be better i think
        min_length = min(self.meanb.shape[0], positions_wkspb.shape[0])
        meanb = self.meanb[:min_length]
        positions_wkspb = positions_wkspb[:min_length]

        error = meanb - positions_wkspb
        if len(error) < thb.shape[1]:
            zeros = torch.zeros((thb.shape[1] - len(error), 1))
            error = torch.concat((error, zeros), dim=0)

        # from matplotlib import pyplot as plt
        # plt.ioff()
        # ax = plt.subplot(projection='3d')
        # ax.plot(*self.robot_model.compute_forward_kinematics(thb[0, :, :self.robot_model.n_dofs])[:, :3].T, label="Curr")
        # ax.plot(*self.robot_model.compute_forward_kinematics(meanb)[:, :3].T, label="PdV")
        # plt.legend()
        # plt.show(block=True)

        return (
            error.view(thb.shape[0], thb.shape[1], self.robot_model.n_dofs, 1),
            Jfksb,
            self.inv_cov,
        )

    def number_of_constraints(self):
        return self._get_num_traj_states() * self.robot_model.n_dofs

    def get_masks(self, columns: int):
        A = torch.zeros((self.number_of_constraints(), columns), dtype=bool)
        K = torch.zeros((self.number_of_constraints(), self.number_of_constraints()), dtype=bool)

        ndof = self.robot_model.n_dofs
        for i in range(self._get_num_traj_states()):
            A[
                i * ndof : (i + 1) * ndof,
                i * self.state_dim : i * self.state_dim + ndof,
            ] = True
            K[
                i * ndof : (i + 1) * ndof,
                i * ndof : (i + 1) * ndof,
            ] = True

        return A, K

    def set_mean(self, meanb: Optional[torch.Tensor]):
        assert (
            meanb is None or meanb.shape[1] == self.robot_model.n_dofs
        ), f"mean vector must be of size (-1, {self.robot_model.n_dofs})"

        self.meanb = meanb
