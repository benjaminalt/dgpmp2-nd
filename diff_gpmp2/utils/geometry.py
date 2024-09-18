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

import open3d as o3d
import torch
import numpy as np

class Cylinder:

    def __init__(self, pose: torch.Tensor, height: float, radius: float) -> None:
        """
        :param pose: 4x4 affine matrix
        """
        self.pose = pose
        self.height = height
        self.radius = radius

    def to(self, device: torch.device):
        self.pose = self.pose.to(device)
        return self

    def transform(self, transform: torch.Tensor) -> "Cylinder":
        """
        :param transform: 4x4 affine matrix
        """
        pose = transform @ self.pose
        return Cylinder(pose=pose, height=self.height, radius=self.radius)

    def to_o3d(self, cartesian_space: bool = True) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.radius, height=self.height)
        if cartesian_space:
            mesh.transform(self.pose.cpu().detach().numpy())
        return mesh

def get_enclosing_cylinder(mesh: o3d.geometry.TriangleMesh, device: torch.device) -> Cylinder:
    verbose = False

    points = torch.vstack([torch.from_numpy(v) for v in np.asarray(mesh.vertices, dtype=np.float32)])
    hull_points = points[mesh.compute_convex_hull()[1]]  # get points part ofgg convex hull

    # get obb and use max extent as height of cylinder
    obb = mesh.get_oriented_bounding_box()
    obb_extents = torch.tensor(obb.extent, dtype=torch.get_default_dtype())
    obb_center = torch.tensor(obb.center, dtype=torch.get_default_dtype())
    obb_R = torch.tensor(obb.R, dtype=torch.get_default_dtype())
    max_extent = torch.max(obb_extents)
    max_extent_axis = torch.argmax(obb_extents)

    # compute min and max points along height axis
    max_axis_offset = torch.zeros(3)
    max_axis_offset[max_extent_axis] = max_extent / 2
    p1 = obb_center - obb_R @ max_axis_offset
    p2 = obb_center + obb_R @ max_axis_offset

    if verbose:
        p1_b = o3d.geometry.TriangleMesh.create_sphere(0.01)
        p1_b.translate(p1)
        p1_b.paint_uniform_color([0, 1, 0])
        p2_b = o3d.geometry.TriangleMesh.create_sphere(0.01)
        p2_b.translate(p2)
        p2_b.paint_uniform_color([1, 0, 0])

    # project all points on plane perpendicular to axis p1-p2 to determine radius of cylinder
    perp_disp = torch.linalg.cross((p2 - p1)[None], hull_points - p1) / torch.norm(p2 - p1)
    dist = torch.norm(perp_disp[:, None] - perp_disp[None], dim=-1)
    max_dist = torch.max(dist)

    # if height axis not z -> rotate so height axis always along z
    if max_extent_axis == 0:
        rot = o3d.geometry.get_rotation_matrix_from_xyz([0, torch.pi / 2, 0])
        obb_R = obb_R @ rot
    elif max_extent_axis == 1:
        rot = o3d.geometry.get_rotation_matrix_from_xyz([torch.pi / 2, 0, 0])
        obb_R = obb_R @ rot

    pose = torch.eye(4, device=device)
    pose[:3, :3] = obb_R
    pose[:3, 3] = obb_center

    cyl_height = max_extent
    cyl_radius = max_dist / 2
    cyl_pose = pose
    cylinder = Cylinder(height=cyl_height, radius=cyl_radius, pose=cyl_pose)

    # check
    if verbose:
        geometries = [cylinder.to_o3d(), obb, mesh]
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.run()
        vis.destroy_window()

    return cylinder