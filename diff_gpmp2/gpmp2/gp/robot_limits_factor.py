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


class RobotLimitsFactor(Factor):
    def __init__(
        self,
        name: str,
        state_dim: int,
        robot_model: DiffRobotModel,
        K: torch.Tensor,
        device: torch.device,
        threshold: float,
        batch_size=1,
    ):
        """
        Parameters
        ===
            thb_point_idx: which point the prior should account for (0 -> start point, -1 -> goal point)
        """
        super().__init__(name, threshold)
        self.state_dim = state_dim
        self.device = device
        self.robot_model = robot_model
        self.batch_size = batch_size
        self.K = K

    def initialize(self, num_traj_states: int):
        super().initialize(num_traj_states)
        self.inv_cov = mat_utils.isotropic_matrix(1.0 / torch.pow(self.K, 2.0), self.robot_model.n_dofs, self.device)[
            None, None
        ].repeat(self.batch_size, num_traj_states, 1, 1)

    def to(self, device: torch.device) -> "RobotLimitsFactor":
        self.K = self.K.to(device)
        if hasattr(self, "inv_cov"):
            self.inv_cov = self.inv_cov.to(device)
        self.device = device

    def threshold_reached(self, error: torch.Tensor) -> bool:
        return torch.norm(error) <= self.threshold

    def admissible_solution(self, error: torch.Tensor) -> bool:
        return True

    def _get_error(self, thb: torch.Tensor, verbose: bool = False):
        assert thb.shape[0] == 1

        thb_confs = thb[..., : self.robot_model.n_dofs]

        limits_low = self.robot_model.joint_limits[:, 0]
        error_low = torch.where(thb_confs < limits_low, limits_low - thb_confs, 0)

        limits_up = self.robot_model.joint_limits[:, 1]
        error_up = torch.where(thb_confs > limits_up, thb_confs - limits_up, 0)

        error = error_low + error_up

        # Jacobian should be diagonal matrix with all ones, or -1 when upper limits are exceeded
        H = torch.eye(self.robot_model.n_dofs, device=self.device)[None, None].repeat((*thb.shape[:2], 1, 1))
        H *= (torch.ones_like(error_up) - 2 * torch.sign(error_up))[..., None]

        return (
            error.view(thb.shape[0], thb.shape[1], self.robot_model.n_dofs, 1),
            H,
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
