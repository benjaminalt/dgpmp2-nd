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
from diff_gpmp2.utils.mat_utils import isotropic_matrix, pose_quaternion_to_euler_zyx


class PriorCartesianTrajectoryFactor(Factor):
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
        error_sum_weight: float = 1.0,
    ):
        """
        Inverse covariance is isotropic matrix with 1/(K^2) on the diagonal
        """
        super().__init__(name, threshold, error_sum_weight)
        self.state_dim = state_dim
        self.wksp_dim = wksp_dim
        self.device = device
        self.robot_model = robot_model
        self.batch_size = batch_size
        self.K = K
        self.set_mean(meanb)

    def initialize(self, num_traj_states: int):
        super().initialize(num_traj_states)
        self.inv_cov = isotropic_matrix(1.0 / torch.pow(self.K, 2.0), self.wksp_dim, self.device)[
            None, None
        ].repeat(self.batch_size, num_traj_states, 1, 1)

    def threshold_reached(self, error: torch.Tensor) -> bool:
        return torch.norm(error) <= self.threshold

    def admissible_solution(self, error: torch.Tensor) -> bool:
        return True

    def _get_error(self, thb: torch.Tensor, verbose: bool = False):
        """
        :return: Pointwise error on trajectory, EE jacobian, inverse covariance
        """
        assert self.meanb is not None
        assert thb.shape[0] == 1
        assert thb.shape[1] == self.meanb.shape[0]
        thb_point = thb[0, :, : self.robot_model.n_dofs]
        pose_wkspb = self.robot_model.compute_forward_kinematics(thb_point)
        pose_wkspb = pose_quaternion_to_euler_zyx(pose_wkspb)

        Jfksb = self.robot_model.compute_endeffector_jacobian(thb_point)

        mean_pose = pose_quaternion_to_euler_zyx(self.meanb)
        error = mean_pose - pose_wkspb
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.plot(error, label=list(range(error.shape[-1])))
        # # plt.plot(positions_rots, label=list(map(lambda s: "Pos %d"% s, range(3))))
        # # plt.plot(mean_rot, label=list(map(lambda s: "PoV %d"% s, range(3))))
        # plt.legend()
        # plt.show()
        # plt.ioff()
        # plt.plot(error.detach().cpu(), label=range(error.shape[1]))
        # plt.legend()
        # plt.show(block=True)
        return (
            error.view(thb.shape[0], thb.shape[1], self.wksp_dim, 1),
            Jfksb,
            self.inv_cov,
        )

    def number_of_constraints(self):
        return self._get_num_traj_states() * self.wksp_dim

    def get_masks(self, columns: int):
        A = torch.zeros((self.number_of_constraints(), columns), dtype=bool)
        K = torch.zeros((self.number_of_constraints(), self.number_of_constraints()), dtype=bool)

        for i in range(self._get_num_traj_states()):
            A[
                i * self.wksp_dim : (i + 1) * self.wksp_dim,
                i * self.state_dim : i * self.state_dim + self.robot_model.n_dofs,
            ] = True
            K[
                i * self.wksp_dim : (i + 1) * self.wksp_dim,
                i * self.wksp_dim : (i + 1) * self.wksp_dim,
            ] = True

        return A, K

    def set_mean(self, meanb: Optional[torch.Tensor]):
        # assert (
        #     meanb is None or meanb.shape[1] == self.wksp_dim
        # ), f"mean vector must be of size (-1, {self.wksp_dim})"

        self.meanb = meanb
