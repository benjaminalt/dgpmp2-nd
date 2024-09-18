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

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
import torch
from diff_gpmp2.differentiable_robot_model.diff_robot_model import DiffRobotModel

from diff_gpmp2.gpmp2.factor import Factor
from diff_gpmp2.gpmp2.obstacle.obstacle_cost import HingeLossObstacleCost
from diff_gpmp2.utils import mat_utils
from diff_gpmp2.utils.geometry import Cylinder
from diff_gpmp2.utils.sdf import SignedDistanceField, caculate_align_mat

logger = logging.getLogger(__name__)


class ObstacleFactor(Factor):

    def __init__(
        self,
        name: str,
        state_dim: int,
        sdf: SignedDistanceField,
        weight: float,
        robot_model: DiffRobotModel,
        device: torch.device,
        threshold: float,
        eps: float = 0.0,
        batch_size: int = 1,
        verbose: bool = False,
        viz_output_dir: Optional[Path] = None,
        viewpoint_config_path: Optional[Path] = None,
    ):
        """
        :param state_dim: number of degrees of freedom of the robot * 2 (velocity also considered)
        :param eps: safety distance from the sdf
        :param plot_output_dir: Directory in which to store generated visualizations. Visualization also requires verbose == True.
        :param viewpoint_config_path: Path to file containing JSON description of open3d viewpoint
        """
        super().__init__(name, threshold)
        self.device = device
        self.robot_model = robot_model
        self.state_dim = state_dim
        self.eps = eps
        self.sdf = sdf
        self.nlinks = self.robot_model.n_links
        self.batch_size = batch_size
        self.verbose = verbose
        self.weight = weight

        logger.debug("ObstacleFactor: Loading collision meshes for collision free planning...")
        self.cylinders = robot_model.cylinders
        self.cylinders_indices = list(map(lambda e: robot_model.get_link_names().index(e[0]), self.cylinders))

        self.obs_cost = HingeLossObstacleCost()

        # Setup visualization
        self.viz_output_dir = viz_output_dir
        if self.viz_output_dir:
            if self.viz_output_dir.exists():
                shutil.rmtree(self.viz_output_dir.as_posix())
            self.viz_output_dir.mkdir()
            if self.verbose:
                logging.info(f"ObstacleFactor: Creating visualiuations at {self.viz_output_dir.as_posix()}")

        self.viewpoint_config_path = (
            viewpoint_config_path
            if viewpoint_config_path
            else Path(__file__).parent / "config" / "viewpoint_default.json"
        )
        self.iter_num = None
        self.seq_num = 0  # Sequence number within iteration

    def to(self, device: torch.device):
        self.device = device
        updated_cylinders = []
        for link_name, cylinder in self.cylinders:
            updated_cylinders.append((link_name, cylinder.to(device)))
        self.cylinders = updated_cylinders
        self.sdf = self.sdf.to(device)
        return self

    def get_cylinder_names(self) -> List[str]:
        return list(map(lambda e: e[0], self.cylinders))

    def initialize(self, num_traj_states: int):
        # if num traj states changes, must recompute inv_cov
        super().initialize(num_traj_states)
        if self.iter_num is None:
            self.iter_num = 0
        else:
            self.iter_num += 1
        self.seq_num = 0
        self.__compute_inv_cov()

    def threshold_reached(self, error: torch.Tensor) -> bool:
        """all obstacle collisions within threshold"""
        return torch.all(torch.abs(error) <= self.threshold)

    def admissible_solution(self, error: torch.Tensor) -> bool:
        """an admissible solution musn't have a collision. Consequently error must be 0"""
        return error.sum().item() == 0

    def __compute_inv_cov(self):
        num_features = self.__num_features()

        obs_inv_cov = mat_utils.isotropic_matrix(1.0 / torch.pow(self.weight, 2.0), num_features, self.device)
        obscov_inv_traj = torch.zeros(self._get_num_traj_states(), num_features, 1, device=self.device)
        obscov_inv_traj = obscov_inv_traj + obs_inv_cov
        self.inv_cov = obscov_inv_traj.expand((self.batch_size, *obscov_inv_traj.shape))  # expand to batch

    def __num_features(self) -> int:
        return len(self.cylinders)

    def number_of_constraints(self):
        return self._get_num_traj_states() * self.__num_features()

    def inverse_transform(self, transform: torch.Tensor):
        assert len(transform.shape) == 3, "batch of pose 4x4 matrizes expected"
        translate = transform[:, :3, 3]
        rotate = transform[:, :3, :3]
        rotate_inv = rotate.transpose(1, 2)
        translate_inv = -torch.bmm(rotate_inv, translate[..., None]).squeeze()
        transform_inv = torch.eye(4, device=transform.device).repeat(len(transform), 1, 1)
        transform_inv[:, :3, :3] = rotate_inv
        transform_inv[:, :3, 3] = translate_inv
        return transform_inv

    def get_cylinder_min_voxel(self, poseb: torch.Tensor, cylinder: Cylinder, verbose: bool = False) -> torch.Tensor:
        """given a cylinder and the pose of the cylinder origin, compute the bks position for each pose that is closet to an environment object

        Closest means it has the lowest entry in the sdf.
        """
        sdf = self.sdf
        cyl_originb = poseb @ cylinder.pose
        cyl_height = cylinder.height
        cyl_radius = cylinder.radius

        # get positions of each sdf element in cartesian space
        # Adds voxel_size / 2 so that the point is exactly between 4 voxels. This ensures the jacobian does not degenerate to be dependant on one voxel only.
        # this does not work really well for some reason
        sdf_bks_affine = self.sdf.bks_positions.reshape((-1, 4)) + self.sdf.voxel_size / 2
        # sdf_bks_affine = self.sdf.bks_positions.reshape((-1, 4))

        # transform sdf voxels in bks into cylinder coordinate system
        cyl_origin_inv = self.inverse_transform(cyl_originb)
        sdf_indices_cyl = (sdf_bks_affine[None] @ cyl_origin_inv.transpose(1, 2))[..., :-1]  # drop affine dimension

        # check which voxels are in cylinder and retrieve voxel with minimal distance
        # FIXME: each voxels origin (sdf_bks_affine) is the bottom left corner of the voxel. The following
        # selects all voxels who's origin is in the cylinder. It does not select the voxels, that intersect with the cylinder.
        in_cylinder_height = torch.abs(sdf_indices_cyl[..., 2]) <= cyl_height / 2
        in_cylinder_radius = torch.norm(sdf_indices_cyl[..., :2], dim=-1) <= cyl_radius
        in_cylinder = in_cylinder_height & in_cylinder_radius

        # old approach: (only 2. of new approach)
        # select voxel that has minimal distance in sdf. Problem is that if no voxel collides with cylinder, voxel is selected at random (because all are have infinite distance)

        # new approach:
        # 1. - from all voxels that intersect with cylinder and have negative distance (collide with object) select the one that is closest to poseb.
        #       The origin of the mesh is not important, more important is the origin of the link the cylinder is attached to
        # 2. - if cylinder only intersects with voxels, that have a positive distance, select the one with the minimal distance,
        # 3. - if cylinder does not intersect with any voxel, use the voxel that is closest to the cylinder

        # 2. base heuristic. Always used
        sdf_cylinder_intersect = torch.where(in_cylinder, sdf.matrix.flatten(), torch.inf)
        min_voxel_bks = sdf_bks_affine[torch.argmin(sdf_cylinder_intersect, dim=1), :3]

        # 1. this helps (often) but adds a lot of computation (memory usage too high)
        sdf_cylinder_collision = sdf_cylinder_intersect < 0
        if torch.any(sdf_cylinder_collision):  # FIXME: cleanup computation
            poseb_inv = self.inverse_transform(poseb)
            sdf_indices_poseb = (sdf_bks_affine[None] @ poseb_inv.transpose(1, 2))[..., :-1]  # drop affine dimension

            in_cylinder_indices = (sdf_cylinder_collision).nonzero()
            norms = torch.norm(sdf_indices_poseb[sdf_cylinder_collision], dim=-1)
            norms = norms.repeat((len(poseb), 1))
            # set norms of cylinder indices that are part of different traj point to inf
            mapping = in_cylinder_indices[:, 0] == torch.arange(len(poseb), device=poseb.device)[:, None]
            norms[~mapping] = torch.inf
            min_indices = norms.argmin(dim=-1)
            min_voxel_indices = in_cylinder_indices[min_indices, 1]
            min_voxel_bks[sdf_cylinder_collision.any(dim=-1)] = sdf_bks_affine[min_voxel_indices, :3][
                sdf_cylinder_collision.any(dim=-1)
            ]

        # 3.  this selection does not help really, but it makes sense
        # no_voxel_intersection = torch.all(~in_cylinder, dim=1)
        # min_voxel_bks[no_voxel_intersection] = sdf_bks_affine[torch.argmin(torch.norm(sdf_indices_cyl[no_voxel_intersection], dim=-1), dim=-1), :3]
        # min_voxel_bks[no_voxel_intersection] = sdf_bks_affine[torch.argmin(torch.norm(sdf_indices_cyl[no_voxel_intersection, :, :2], dim=-1) - cyl_radius + torch.abs(sdf_indices_cyl[no_voxel_intersection, :, 2]) - cyl_radius / 2, dim=-1), :3]
        # min_voxel_bks[no_voxel_intersection] = poseb[no_voxel_intersection, :3, 3]

        if verbose:
            # sdf_indices_in_cyl_bks = sdf_indices_bks_affine[..., :3][in_cylinder]
            # sdf_indices_out_cyl_bks = sdf_indices_bks[~in_cylinder][::10] # every 10th for performance reasons
            cyl = torch.tensor(cylinder.to_o3d().vertices)
            from matplotlib import pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            # ax.scatter(*sdf_indices_out_cyl_bks.T, alpha=0.1, marker='o')
            ax.scatter(
                *sdf_indices_in_cyl_bks.T,
                alpha=0.2,
                marker="x",
                label="voxels in cylinder",
            )
            ax.scatter(*cyl_originb[0, :3, 3][..., None], marker="^", label="cylinder center")
            ax.scatter(*min_voxel_bks[..., None], marker="+", label="min distance")
            ax.scatter(*cyl.T, alpha=0.2, marker="v", label="cylinder shape")
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            ax.set_zlabel("Z Label")
            plt.legend()
            plt.show()

        return min_voxel_bks

    def _get_error(self, thb: torch.Tensor, verbose: bool = False):
        """
        Arguments:
            thb: (1, traj_len, state_dim)

        Details:
            sphere_centersb: torch.tensor (batch_size, num_traj_states, nlinks, wksp_dim)

            H_fkb: torch.tensor (batch_size, num_traj_states, n_links * wksp_dim, state_dim)
                Jacobian from joint configuration to joint position in wksp
            H_eb: torch.tensor (batch_size, num_traj_states, nlinks, n_links * wksp_dim)

            Computes H1b = H_eb @ H_fkb
                this is the derivative of c(d(x(theta, S)))
                    c=> hinge loss
                    d=> distance function (sdf)
                    x=> forwardk kinematics
                    theta=> joint configuration

        Returns:
            H1b: torch.tensor (batch_size, num_traj_states, nlinks, state_dim)
                Jacobian from obstacle error for each link (sphere) to joint configuration
        """
        assert thb.shape[0] == 1, "batch not working yet"

        thb = thb[0, :, : self.robot_model.n_dofs]

        # fast (~ 0.015s)
        wksp_pose_links = self.robot_model.compute_forward_kinematics_all_links(thb)
        wksp_pose_links = wksp_pose_links[:, self.cylinders_indices]
        wksp_affine_links = mat_utils.pose_to_affine(wksp_pose_links)

        # fast (< 0.01s)
        Jfks_pose_links = self.robot_model.compute_endeffector_jacobian_all_links(thb)
        Jfks_pos_links = Jfks_pose_links[:, self.cylinders_indices]
        # Jfks_pos_links has dim (... x xyzrpy x ndof)
        # Jfks_pos_links = pose_euler_zyx_to_affine(Jfks_pos_links.transpose(-2, -1)).transpose(-3, -1)

        # fast (~ 0.03s)
        voxelsbb = []
        idx = 0
        for idx, cylinder in enumerate(self.cylinders):
            positionb = self.get_cylinder_min_voxel(wksp_affine_links[:, idx], cylinder[1], verbose=False)
            voxelsbb.append(positionb)
        voxelsbb = torch.stack(voxelsbb, dim=1)

        voxelsb = voxelsbb.reshape(-1, voxelsbb.shape[-1])
        # not applied: for security add voxel_size / 2 as padding
        # sphere_radiib = torch.full(size=(voxelsb.shape[0], ), fill_value=0.0, device=self.device)
        diagonal = torch.sqrt(torch.as_tensor(2 * (self.sdf.voxel_size / 2) ** 2))  # diagonal from voxel center to edge
        diagonal = torch.sqrt(
            diagonal**2 + torch.as_tensor((self.sdf.voxel_size / 2) ** 2)
        )  # diagonal from voxel center to corner
        sphere_radiib = torch.full(size=(voxelsb.shape[0],), fill_value=diagonal, device=self.device)

        # fast (< 0.01s)
        errorb, _, H_eb = self.obs_cost.hinge_loss_signed_batch(voxelsb, sphere_radiib, self.eps, self.sdf)

        errorb = errorb.reshape(*voxelsbb.shape[:2], 1)
        H_e = H_eb.reshape(*voxelsbb.shape[:2], *H_eb.shape[-2:])

        # only cylinder translation jacobian, rotational not needed
        Jfks_pos_links = Jfks_pos_links[:, :, :3]

        # concat zero jacobian for velocities, at it is also part of the state
        H_fkb = torch.concat((Jfks_pos_links, torch.zeros_like(Jfks_pos_links)), dim=-1)

        H1b = torch.einsum("bsij,bsjk->bsik", H_e, H_fkb)

        # FIXME: hack: dont look at start & end of trajectory as they will always collide due to approximation with environment.
        # five_p_points = thb.shape[0] * 5 // 100
        # errorb[:five_p_points] = 0
        # errorb[-five_p_points:] = 0

        # if torch.any(errorb[[0, -1]] != 0):
        #     logger.warn(
        #         """Obstacle Factor adds error to start or end configuration.
        #         This will collide with start/goal prior and produce inaccurate trajectories.
        #         Make sure that the start/goal pose do not collide with the environment"""
        #     )

        # add removed batch
        errorb = errorb[None]
        H1b = H1b[None]

        if self.verbose and self.viz_output_dir:
            self._plot_trajectory_and_planning_world(errorb, wksp_affine_links, voxelsbb, H_e)

        # if self.verbose:
        #     from matplotlib import pyplot as plt
        #     plt.figure()
        #     for idx in range(errorb.shape[2]):  # error for each link
        #         link_error = errorb.cpu().detach()[0, :, idx, 0]
        #         plt.plot(link_error, label=f"%d (%e)" % (idx, link_error.sum().item()))
        #     plt.plot(torch.sum(errorb[0, :, :, 0], dim=1).detach().cpu(), '--', label="sum")
        #     plt.title(f"Obstacle Error {torch.sum(errorb).item()}")
        #     plt.legend()
        #     # plt.savefig('obstacle.png')
        #     plt.show()  # FIXME

        return errorb, H1b, self.inv_cov

    def get_inv_cov(self):
        return self.inv_cov

    def set_im_sdf(self, im, sdf):
        self.obs_cost.env.initialize_from_image(im, sdf)

    def set_eps(self, eps):
        self.eps = eps

    def eval_prob(self, state):
        error, _ = self._get_error(state)
        f = np.exp(-1.0 / 2.0 * np.transpose(error).dot(self.inv_cov).dot(error))
        norm = np.power(2.0 * np.pi, self.ndims / 2.0) * np.sqrt(np.linalg.det(self.inv_cov))
        f = f / norm
        return f

    def get_masks(self, columns: int):
        num_features = self.__num_features()
        A = torch.zeros((self.number_of_constraints(), columns), dtype=bool)
        K = torch.zeros((self.number_of_constraints(), self.number_of_constraints()), dtype=bool)

        for i in range(self._get_num_traj_states()):
            A[
                i * num_features : (i + 1) * num_features,
                i * self.state_dim : (i + 1) * self.state_dim,
            ] = True
            K[
                i * num_features : (i + 1) * num_features,
                i * num_features : (i + 1) * num_features,
            ] = True

        return A, K

    def _plot_trajectory_and_planning_world(self, errorb: torch.Tensor, wksp_affine_links, voxelsbb, H_e):
        # es = [[0, i, torch.argmin(errorb[0, i]), 0]
        #         for i in range(errorb.shape[1] * 3 // 4, errorb.shape[1])]  # always display center of trajectory
        traj_points_with_collision = (
            (errorb != 0).nonzero().cpu().numpy()
        )  # display all trajectory points that have collision
        # show collision env
        collision_env_pointcloud = self.sdf.get_voxels_o3d_as_point_cloud(
            color=[0, 0, 0]
        )  # Collision objects are black voxels

        # show ee_link path
        traj_spheres = []
        for pose in wksp_affine_links[:, -1]:
            mesh_sphere = (
                o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                .compute_vertex_normals()
                .transform(pose.detach().cpu().numpy())
                .paint_uniform_color([0.1, 0.1, 0.7])
            )
            traj_spheres.append(mesh_sphere)

        traj_points_with_collision_iter = iter(traj_points_with_collision)
        cs = []
        e = traj_points_with_collision[0] if len(traj_points_with_collision) > 0 else None
        cyls = []
        cylinders = False

        def next_e(vis):
            nonlocal e
            try:
                e = next(traj_points_with_collision_iter)
            except:
                return
            traj_idx = e[1]
            link_idx = e[2]
            nonlocal cs
            for c in cs:
                vis.remove_geometry(c)
            cs.clear()

            # show robot
            for idx, (link_name, c) in enumerate(
                self.robot_model.get_oobs_of_links()
            ):  # Also includes attached objects
                wksp_pose_quat_link = wksp_affine_links[traj_idx, idx]
                c = c.compute_triangle_normals().compute_vertex_normals()
                c = c.transform(wksp_pose_quat_link.cpu().detach().numpy())
                cs.append(c)

            # show collision cylinders
            nonlocal cyls
            nonlocal cylinders
            if cylinders:
                toggle_cylinders(vis)  # remove cylinders
            cyls = []
            for idx, cyl in enumerate(self.cylinders):
                cyl_tf = wksp_affine_links[traj_idx, idx].cpu().detach().numpy()
                cyls.append(cyl[1].to_o3d().transform(cyl_tf).compute_vertex_normals().compute_triangle_normals())
            cyls[link_idx].paint_uniform_color([0, 0, 0.8])
            cylinders = False

            # show jacobian direction
            start = voxelsbb[traj_idx, link_idx].detach().cpu()
            jac = (H_e[traj_idx, link_idx, 0] * errorb[0, traj_idx, link_idx]).cpu().detach()
            vec_len = np.linalg.norm(jac)
            vec = caculate_align_mat(jac / vec_len)
            if vec_len != 0:
                cs.append(
                    o3d.geometry.TriangleMesh.create_arrow(
                        cone_height=0.2 * vec_len,
                        cone_radius=0.06 * vec_len,
                        cylinder_height=0.8 * vec_len,
                        cylinder_radius=0.04 * vec_len,
                    )
                    .compute_vertex_normals()
                    .rotate(vec, center=[0, 0, 0])
                    .translate(start)
                    .paint_uniform_color([0.5, 0.5, 0])
                )
            # cs.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.005).compute_vertex_normals().translate(start).paint_uniform_color([0, 1, 0]))
            # cs.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate((start + jac).cpu().detach()).paint_uniform_color([0.5, 0.5, 0]))

            for c in cs:
                vis.add_geometry(c)

        def toggle_cylinders(vis):
            nonlocal cylinders
            nonlocal cyls
            if cylinders:
                for cyl in cyls:
                    vis.remove_geometry(cyl)
            else:
                for cyl in cyls:
                    vis.add_geometry(cyl)
            cylinders = not cylinders

        large_arrow = None

        def toggle_arrow(vis):
            nonlocal large_arrow
            nonlocal e
            if e is None:
                return
            if large_arrow is not None:
                vis.remove_geometry(large_arrow)
                large_arrow = None
            else:
                traj_idx = e[1]
                link_idx = e[2]
                start = voxelsbb[traj_idx, link_idx].cpu().detach().numpy()
                jac = H_e[traj_idx, link_idx, 0].cpu().detach() * errorb[0, traj_idx, link_idx].cpu()
                vec_len = np.linalg.norm(jac)
                vec = caculate_align_mat(jac / vec_len)
                vec_len = vec_len * 10
                large_arrow = (
                    o3d.geometry.TriangleMesh.create_arrow(
                        cone_height=0.2 * vec_len,
                        cone_radius=0.06 * vec_len,
                        cylinder_height=0.8 * vec_len,
                        cylinder_radius=0.04 * vec_len,
                    )
                    .compute_vertex_normals()
                    .rotate(vec, center=[0, 0, 0])
                    .translate(start)
                    .paint_uniform_color([0.5, 0.5, 0])
                )
                vis.add_geometry(large_arrow)

        def print_pinhole_parameters(vis):
            print("Extrinsic parameters:")
            print(vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic)

        if self.verbose and self.viz_output_dir:
            # directly save pngs from same viewpoint
            vis = o3d.visualization.Visualizer()
            with open(self.viewpoint_config_path) as viewpoint_config_file:
                js = json.load(viewpoint_config_file)
            if "intrinsic" in js:
                width, height = js["intrinsic"]["width"], js["intrinsic"]["height"]
            else:
                width, height = 800, 600
            vis.create_window(
                width=width, height=height, visible=False
            )  # works for me with False, on some systems needs to be true
            next_e(vis)

            vis.add_geometry(collision_env_pointcloud)
            for c in traj_spheres:
                vis.add_geometry(c)

            # toggle_cylinders(vis)

            vc = vis.get_view_control()

            if "trajectory" in js:
                vc.change_field_of_view(js["trajectory"][0]["field_of_view"])
                vc.set_front(js["trajectory"][0]["front"])
                vc.set_lookat(js["trajectory"][0]["lookat"])
                vc.set_up(js["trajectory"][0]["up"])
                vc.set_zoom(js["trajectory"][0]["zoom"])
            elif "extrinsic" in js:  # Pinhole camera parameters
                pinhole_params = o3d.io.read_pinhole_camera_parameters(self.viewpoint_config_path.as_posix())
                assert vc.convert_from_pinhole_camera_parameters(pinhole_params)
            else:
                raise ValueError(f"Unknown open3d viewpoint file format in {self.viewpoint_config_path.as_posix()}")

            vis.poll_events()
            vis.update_renderer()
            iter_output_dir = self.viz_output_dir / str(self.iter_num)
            iter_output_dir.mkdir(exist_ok=True)
            vis.capture_screen_image((iter_output_dir / f"obstacle_factor_{self.seq_num:05d}.png").as_posix())
            vis.destroy_window()
            self.seq_num += 1
        else:
            # interactive open3d window
            geoms = [collision_env_pointcloud] + traj_spheres
            o3d.visualization.draw_geometries_with_key_callbacks(
                geoms,
                {
                    ord("C"): toggle_cylinders,
                    ord("E"): next_e,
                    ord("A"): toggle_arrow,
                    ord("P"): print_pinhole_parameters,
                },
            )

    def plot_trajectory(self, thb: torch.Tensor, output_dir: Optional[Path] = None,
                        show_cylinders=False):

        collision_env_pointcloud = self.sdf.get_voxels_o3d_as_point_cloud(
            color=[0, 0, 0]
        )  # Collision objects are black voxels
        wksp_pose_links = self.robot_model.compute_forward_kinematics_all_links(
            thb.squeeze()[:, : self.robot_model.n_dofs]
        )

        wksp_pose_links = wksp_pose_links[:, self.cylinders_indices]
        wksp_affine_links = mat_utils.pose_to_affine(wksp_pose_links)

        # show ee_link path
        traj_spheres = []
        for pose in wksp_affine_links[:, -1]:
            mesh_sphere = (
                o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                .compute_vertex_normals()
                .transform(pose.detach().cpu().numpy())
                .paint_uniform_color([0.1, 0.1, 0.7])
            )
            traj_spheres.append(mesh_sphere)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        for traj_idx, joint_state in enumerate(thb):
            # show robot
            robot_links = []
            for link_idx, (link_name, c) in enumerate(
                self.robot_model.get_oobs_of_links()
            ):  # Also includes attached objects
                wksp_pose_quat_link = wksp_affine_links[traj_idx, link_idx]
                c = c.compute_triangle_normals().compute_vertex_normals()
                c = c.transform(wksp_pose_quat_link.cpu().detach().numpy())
                robot_links.append(c)

            cyls = []
            if show_cylinders:
                for idx, cyl in enumerate(self.cylinders):
                    cyl_tf = wksp_affine_links[traj_idx, idx].cpu().detach().numpy()
                    cyls.append(cyl[1].to_o3d().transform(cyl_tf).compute_vertex_normals().compute_triangle_normals())

            vis = o3d.visualization.Visualizer()
    
            if output_dir:
                # save PNG from fixed viewpoint
                with open(self.viewpoint_config_path) as viewpoint_config_file:
                    js = json.load(viewpoint_config_file)
                if "intrinsic" in js:
                    width, height = js["intrinsic"]["width"], js["intrinsic"]["height"]
            else:
                width, height = 800, 600

            show = output_dir is None
            vis.create_window(
                width=width, height=height, visible=show
            )  # works for me with False, on some systems needs to be true

            vis.add_geometry(collision_env_pointcloud)
            for c in traj_spheres:
                vis.add_geometry(c)
            for c in robot_links:
                vis.add_geometry(c)
            for c in cyls:
                vis.add_geometry(c)

            if output_dir:
                vc = vis.get_view_control()
                if vc is None:
                    return
                
                if "trajectory" in js:
                    vc.change_field_of_view(js["trajectory"][0]["field_of_view"])
                    vc.set_front(js["trajectory"][0]["front"])
                    vc.set_lookat(js["trajectory"][0]["lookat"])
                    vc.set_up(js["trajectory"][0]["up"])
                    vc.set_zoom(js["trajectory"][0]["zoom"])
                elif "extrinsic" in js:  # Pinhole camera parameters
                    pinhole_params = o3d.io.read_pinhole_camera_parameters(self.viewpoint_config_path.as_posix())
                    assert vc.convert_from_pinhole_camera_parameters(pinhole_params)
                else:
                    raise ValueError(f"Unknown open3d viewpoint file format in {self.viewpoint_config_path.as_posix()}")

            vis.run()

            if output_dir:
                vis.poll_events()
                vis.update_renderer()
                filepath = output_dir / f"obstacle_factor_{traj_idx}.png"
                vis.capture_screen_image(filepath.as_posix())
                vis.destroy_window()

