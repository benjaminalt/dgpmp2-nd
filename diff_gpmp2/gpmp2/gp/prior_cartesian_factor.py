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


class PriorCartesianFactor(Factor):
    def __init__(
        self,
        name: str,
        state_dim: int,
        wksp_dim: int,
        sig,
        robot_model: DiffRobotModel,
        thb_point_idx: int,
        K: torch.Tensor,
        device: torch.dtype,
        threshold: float,
        meanb: torch.Tensor = None,
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
        self.thb_point_idx = thb_point_idx
        self.robot_model = robot_model
        self.set_mean(meanb)

        self.set_inv_cov(K, batch_size)

    def threshold_reached(self, error: torch.Tensor) -> bool:
        return torch.norm(error) <= self.threshold

    def admissible_solution(self, error: torch.Tensor) -> bool:
        return True

    def _get_error(self, thb: torch.Tensor, verbose: bool = False):
        thb_point = thb[:, self.thb_point_idx, : self.robot_model.n_dofs]
        pose_wkspb = self.robot_model.compute_forward_kinematics(thb_point)
        positions_wkspb = pose_wkspb[:, :3]  # select position

        Jfksb = self.robot_model.compute_endeffector_jacobian(thb_point)
        Jfksb = Jfksb[:, :3]  # only lin jac
        Jfksb = Jfksb.reshape(thb_point.shape[0], Jfksb.shape[1], Jfksb.shape[2])
        # concat zero jacobian for velocities, at it is also part of the state
        H_fkb = torch.cat((Jfksb, torch.zeros_like(Jfksb)), dim=-1).contiguous()

        error = self.meanb - positions_wkspb
        return error.view(thb.shape[0], 1, self.wksp_dim, 1), H_fkb, self.inv_cov

    def number_of_constraints(self):
        return self.wksp_dim

    def get_masks(self, columns: int):
        if self.thb_point_idx == -1:  # if end would be -0 the last element is ment
            end = columns
        else:
            end = self.state_dim * (self.thb_point_idx + 1)

        A = torch.zeros((self.number_of_constraints(), columns), dtype=bool)
        A[:, self.state_dim * self.thb_point_idx : end] = True
        K = torch.ones((self.number_of_constraints(), self.number_of_constraints()), dtype=bool)
        return A, K

    def set_mean(self, meanb: Optional[torch.Tensor]):
        assert meanb is None or meanb.shape[0] == self.wksp_dim, f"mean vector must be of size {self.wksp_dim}"

        self.meanb = meanb

    def set_inv_cov(self, K, batch_size):
        inv_covb = (
            mat_utils.isotropic_matrix(1.0 / torch.pow(K, 2.0), self.wksp_dim, self.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        self.inv_cov = inv_covb.unsqueeze(1)
