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

import torch

from diff_gpmp2.gpmp2.factors import Factors
from diff_gpmp2.gpmp2.gp.prior_cartesian_trajectory_factor import PriorCartesianTrajectoryFactor
from diff_gpmp2.gpmp2.obstacle.obstacle_factor import ObstacleFactor


def plot_matplotlib(last_errors, A, b, K, current_trajectory, factors: Factors, output_dir):
    from matplotlib import pyplot as plt

    hd = None
    for factor in factors.get_factors():
        if isinstance(factor, ObstacleFactor):
            robot = factor.robot_model
        if isinstance(factor, PriorCartesianTrajectoryFactor):
            hd = factor.meanb.detach().cpu()
            robot = factor.robot_model

    metadata = {}
    for factor in factors.get_factors():
        metadata[factor.name] = torch.as_tensor(factor.number_of_constraints()).item()

    traj_joints = current_trajectory[0, :, :6]  # only joints relevant
    traj_pose = robot.compute_forward_kinematics(traj_joints)
    traj_jacobian = robot.compute_endeffector_jacobian(traj_joints)  # jacobian is [n_links, n_dof]

    traj_pos = traj_pose[:, :3]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=10, azim=-90)

    factors_used = torch.as_tensor([(val != 0).any() for val in last_errors.values()])
    metadata_filtered = {k: v for (k, v), used in zip(metadata.items(), factors_used) if used}
    colors = lambda v: plt.get_cmap("tab10")(v / len(metadata))

    n_traj_points = len(traj_joints)

    ax.scatter(*traj_pos.T.cpu().detach().numpy(), c="k")
    if hd is not None:
        hd_pos = hd[:, :3]
        ax.scatter(
            *hd_pos.T.cpu().detach().numpy(),
            color=colors(list(metadata.keys()).index("TrajPrior")),
        )

    idx = 0
    for c_i, (factor, factor_constraints) in enumerate(metadata.items()):
        if factor not in metadata_filtered.keys():
            continue
        to_idx = idx + factor_constraints
        b_factor = b[0, idx:to_idx]
        A_factor = A[0, idx:to_idx]
        K_factor = K[0, idx:to_idx]
        idx = to_idx

        if factor == "TrajPrior":
            # b_factor consists of error for each component of pose
            b_factor = b_factor.reshape((n_traj_points, 6))
            b_factor_pos = b_factor[:, :3]

            torch.save(b_factor_pos, (output_dir / "traj.pt").as_posix())

            to_traj_pos = traj_pos + b_factor_pos
            for f, t in zip(traj_pos, to_traj_pos):
                ax.plot(
                    *torch.stack((f, t), dim=0).T.cpu().detach().numpy(),
                    color=colors(c_i),
                )

        elif factor == "Obst.":
            # b_factor consists of error for each cylinder of robot
            b_factor = b_factor.reshape((n_traj_points, -1))
            A_factor = A_factor.reshape((n_traj_points, -1, n_traj_points, 12))

            delta_positions = []

            for i_f, (point_pos, point_jac, b_factor_tp, A_factor_tp) in enumerate(
                zip(traj_pos, traj_jacobian, b_factor, A_factor)
            ):
                # b_factor_tp contains error cor each cylinder at traj. point
                # A_factor_tp contains jacobian for each cylinder at traj. point mapping error to how to move joints
                summed_A_factor = (A_factor_tp.transpose(0, 2) @ b_factor_tp).T

                # only the i_f traj point can have a jacobian, as the other trajectory points dont influence the
                # error of this traj point
                summed_A_factor = summed_A_factor[i_f]

                delta_conf = summed_A_factor[:6]  # only delta conf interesting, skip delta vel-conf

                delta_pose = point_jac @ delta_conf  # jacobian maps joint space delta to cartesian space delta

                delta_position = delta_pose[:3]  # cartesian rotational delta not interested in
                delta_positions.append(delta_position)

                ax.plot(
                    *torch.stack((point_pos, point_pos + delta_position), dim=0).T.cpu().detach().numpy(),
                    color=colors(c_i),
                )

            torch.save(torch.stack(delta_positions, dim=0), (output_dir / "obst.pt").as_posix())

        elif factor == "Start":
            delta_conf = b_factor[:6].squeeze()
            delta_pose = traj_jacobian[0] @ delta_conf
            delta_pos = delta_pose[:3]
            ax.plot(
                *torch.stack((traj_pos[0], traj_pos[0] + delta_pos), dim=0).T.cpu().detach().numpy(),
                color=colors(c_i),
            )

        elif factor == "Goal":
            delta_conf = b_factor[:6].squeeze()
            delta_pose = traj_jacobian[0] @ delta_conf
            delta_pos = delta_pose[:3]
            ax.plot(
                *torch.stack((traj_pos[-1], traj_pos[-1] + delta_pos), dim=0).T.cpu().detach().numpy(),
                color=colors(c_i),
            )

        elif factor == "GP":
            pass

    plt.savefig((output_dir / "traj.png").as_posix())

    plt.close(fig)

    ## save obst plot
    if "Obst." in last_errors:
        for f in factors.get_factors():
            if f.name == "Obst.":
                cylinder_names = f.get_cylinder_names()
        b_factor = last_errors["Obst."].squeeze()
        plt.figure()
        for idx in range(b_factor.shape[1]):  # error for each link
            link_error = b_factor.cpu().detach()[:, idx]
            plt.plot(
                link_error,
                label=f"%s (%.2f)" % (cylinder_names[idx], link_error.sum().item()),
            )
        plt.plot(torch.sum(b_factor, dim=1).detach().cpu(), "--", label="sum")
        plt.title(f"Obstacle Error {torch.sum(b_factor).item()}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(output_dir / "obstacle.png", bbox_inches="tight")
        plt.close()

    if "TrajPrior" in last_errors:
        b_factor = last_errors["TrajPrior"].squeeze()
        plt.figure()
        for idx in range(b_factor.shape[1]):  # error for each link
            link_error = b_factor.cpu().detach()[:, idx]
            plt.plot(link_error, label=f"%s (%.2f)" % (idx, link_error.sum().item()))
        plt.plot(torch.sum(b_factor, dim=1).detach().cpu(), "--", label="sum")
        plt.title(f"PriorTraj Error {torch.sum(b_factor).item()}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.savefig(output_dir / "prior-traj.png", bbox_inches="tight")
        plt.close()
