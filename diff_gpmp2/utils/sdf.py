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

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy import ndimage
import open3d as o3d
import transformations as tf

logger = logging.getLogger(__name__)

class CollisionObject:
    def __init__(self, object_pose: torch.Tensor, mesh: o3d.geometry.TriangleMesh):
        """
        :param object_pose
        :param mesh: Collision mesh
        """
        if torch.any(torch.tensor(mesh.get_axis_aligned_bounding_box().get_extent())) > 10:
            logger.info("mesh larger than 10 meters. Assuming unit mm and scaling by 0.001")
            mesh = mesh.scale(0.001, center=[0, 0, 0])

        mesh_origin = object_pose[:3]
        mesh_orientation = object_pose[3:]
        mesh_rotation = torch.tensor(tf.euler_matrix(*mesh_orientation)[:3, :3], dtype=torch.get_default_dtype())
        mesh_transform = torch.eye(4)
        mesh_transform[:3, :3] = mesh_rotation
        mesh_transform[:3, 3] = mesh_origin
        self.mesh = mesh.transform(mesh_transform)


class SignedDistanceField:
    def __init__(self, matrix: torch.Tensor, origin: torch.Tensor, voxel_size: float) -> None:
        """
        :param voxel_size: In m
        """
        self.matrix = matrix
        self.origin = origin
        self.voxel_size = voxel_size
        self.bks_positions = self.__compute_bks_positions()

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    def __compute_bks_positions(self) -> torch.Tensor:
        device = self.matrix.device
        sdf_indices = torch.tensor(np.indices(self.matrix.shape), dtype=int, device=device).permute((1, 2, 3, 0))
        sdf_indices_bks = sdf_indices * self.voxel_size + self.origin
        return torch.concat(
            (
                sdf_indices_bks,
                torch.ones((*sdf_indices_bks.shape[:3], 1), device=device),
            ),
            dim=-1,
        )

    def get_cartesian_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        extents = torch.tensor(self.matrix.shape, device=self.origin.device) * self.voxel_size
        to = self.origin + extents
        return self.origin, to

    def to(self, device: torch.device) -> "SignedDistanceField":
        self.matrix = self.matrix.to(device)
        self.origin = self.origin.to(device)
        self.bks_positions = self.bks_positions.to(device)
        return self

    def get_voxels_o3d(self) -> List[o3d.geometry.TriangleMesh]:
        mat = self.matrix
        # substract voxel_size / 2 since bks_position is position of center
        voxels_bks = self.bks_positions[mat <= 0] - self.voxel_size / 2
        return [
            o3d.geometry.TriangleMesh.create_box(
                width=self.voxel_size, height=self.voxel_size, depth=self.voxel_size
            )
            .translate(v.squeeze()[:3])
            .compute_triangle_normals()
            .compute_vertex_normals()
            .paint_uniform_color([1, 0, 0])
            for v in voxels_bks.detach().cpu()
        ]

    def get_voxels_o3d_as_point_cloud(self, color: List[int] = None) -> o3d.geometry.PointCloud:
        mat = self.matrix
        # substract voxel_size / 2 since bks_position is position of center
        voxels_bks = self.bks_positions[mat <= 0] - self.voxel_size / 2
        voxels_bks_pos_np = voxels_bks.cpu().detach().numpy()[..., :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxels_bks_pos_np)
        # pcd.estimate_normals()
        if color:
            pcd.paint_uniform_color(color)
        return pcd

    def visualize(self):
        from diff_gpmp2.gpmp2.obstacle.obstacle_cost import HingeLossObstacleCost

        device = self.matrix.device
        obs = HingeLossObstacleCost()
        voxelsb = self.bks_positions[self.matrix < 0][..., :3]
        voxelsb[..., :] += torch.as_tensor(self.voxel_size) * 0.5
        voxelsb = voxelsb.reshape(-1, voxelsb.shape[-1])
        sphere_radiib = torch.full(size=(voxelsb.shape[0],), fill_value=0, device=device)
        errorb, _, H = obs.hinge_loss_signed_batch(voxelsb, sphere_radiib, self)

        e2 = []
        e2.extend(self.get_voxels_o3d())
        for start, H, e in zip(voxelsb, H[:, 0, :], errorb):
            jac = H.detach() * e
            vec_len = np.linalg.norm(jac)
            vec = caculate_align_mat(jac / vec_len)
            if vec_len == 0:
                continue
            e2.append(
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

        e2.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=self.origin))
        e2.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

        o3d.visualization.draw_geometries(e2)

    @staticmethod
    def from_obstacles(obstacles: List[CollisionObject], collision_padding: float, voxel_size: float,
                       visualize=False) -> 'SignedDistanceField':
        """
        Returns signed distance field for the given obstacles.
        """

        union_mesh = obstacles[0].mesh
        for obstacle in obstacles[1:]:
            union_mesh += obstacle.mesh

        # compute how many voxels to add as padding (round up)
        padlen_v = torch.ceil(torch.tensor(collision_padding / voxel_size)).to(int)
        # re-compute the actual padding applied
        padlen = padlen_v * voxel_size  # update collision_padding by rounding error

        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(union_mesh, voxel_size=voxel_size)
        voxels = torch.vstack([torch.tensor(v.grid_index, dtype=int) for v in voxel_grid.get_voxels()])
        origin = torch.tensor(voxel_grid.origin, dtype=torch.get_default_dtype())

        # translate by collision_padding
        origin = origin - padlen
        voxels = voxels + padlen_v

        # compute how large the sdf matrix must be
        max_idx = torch.max(voxels, dim=0)[0] + 1
        max_idx = max_idx + padlen_v  # add padding
        matrix = torch.zeros(*max_idx, dtype=bool)  # max elements per dim equals max_idx + 1
        matrix[tuple(voxels.T)] = True

        # matrix must be True when NO collision -> bitwise_not
        im = torch.clone(~matrix).to(dtype=torch.get_default_dtype())
        # TODO: add padding im = torch.pad(im, (collision_padding, collision_padding), "constant", constant_values=(1, 1))
        inv_im = 1.0 - im
        dist_func = ndimage.distance_transform_edt
        im_dist = torch.tensor(
            dist_func(im.numpy()), dtype=torch.get_default_dtype()
        )  # TODO: not gradient applicable yet
        inv_im_dist = torch.tensor(
            dist_func(inv_im.numpy()), dtype=torch.get_default_dtype()
        )  # TODO: not gradient applicable yet
        sdf_matrix = (im_dist - inv_im_dist) * voxel_size

        sdf = SignedDistanceField(matrix=sdf_matrix, origin=origin, voxel_size=voxel_size)

        if visualize:
            voxels_bks = sdf.bks_positions[matrix <= 0]
            boxes = [
                o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
                .translate(v.squeeze()[:3])
                .compute_triangle_normals()
                .compute_vertex_normals()
                .paint_uniform_color([1, 0, 0])
                for v in voxels_bks
            ]
            boxes.append(union_mesh)
            o3d.visualization.draw_geometries(boxes)
        
        return sdf

def sdf_2d(image, padlen=1, res=1.0):
    """
    Returns signed distance transform for the input image.
    Remember to convert it to actual metric values when using with planner by multiplying it with
    environment resolution.
    """

    im = np.array(image > 0.75, dtype=np.float64)

    if padlen > 0:
        im = np.pad(im, (padlen, padlen), "constant", constant_values=(1.0, 1.0))
    inv_im = np.array(1.0 - im, dtype=np.float64)
    dist_func = ndimage.distance_transform_edt
    im_dist = dist_func(im)
    inv_im_dist = dist_func(inv_im)
    sedt = (im_dist - inv_im_dist) * res
    return sedt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def costmap_2d(sdf, eps):
    loss = -1.0 * sdf + eps
    hinge = sdf <= eps
    hinge = hinge.double()
    cost_map = hinge * loss
    return cost_map


def safe_sdf(sdf, eps):
    loss = -1.0 * sdf + eps
    return loss


def bilinear_interpolate_3d(
    imb: SignedDistanceField, stateb: torch.Tensor, device: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """bilinear interpolation
    get costs for states in `stateb` by interpolation sdf `imb`.

    Arguments:
      imb: torch.tensor (batch_size, rows, columns, depth)
      stateb: torch.tensor (batch_size, num_traj_states * n_links, wksp_dim)
        batch of positions in work space
    Returns:
      distance_to_obstacles: torch.tensor (batch_size, num_traj_states * n_links, 1)
      Jacobian: torch.tensor (batch_size, num_traj_states * n_links, wksp_dim)
    """
    limits_low, limits_high = imb.get_cartesian_limits()
    res = imb.voxel_size
    sdf = imb.matrix

    # must have padding of at least 1 voxel size. Otherwise, no neighboring points to compute diff error with
    limits_low_padded = limits_low + imb.voxel_size
    limits_high_padded = limits_high - imb.voxel_size

    # only select trajectory points that are within sdf, the other points have distance inf
    in_limits = torch.all((stateb <= limits_high_padded) & (stateb >= limits_low_padded), dim=-1)
    stateb_inlim = stateb[in_limits]
    J_inlim = torch.zeros((stateb_inlim.shape[0], 1, stateb_inlim.shape[1]), device=device)

    # compute where the cartesian origin is in obstacle space
    orig_pix = -limits_low / res  # coordinates of origin

    # position of trajectory in 'obstacle space'
    px = (orig_pix[0] + stateb_inlim[:, 0] / res).contiguous().view(-1)
    py = (orig_pix[1] + stateb_inlim[:, 1] / res).contiguous().view(-1)
    pz = (orig_pix[2] + stateb_inlim[:, 2] / res).contiguous().view(-1)

    # the four interpolation points in 2d  that surround the goal point (px, py)
    px1 = torch.floor(px).long()
    px2 = px1 + 1
    py1 = torch.floor(py).long()
    py2 = py1 + 1
    pz1 = torch.floor(pz).long()
    pz2 = pz1 + 1

    px1 = torch.clamp(px1, 0, sdf.shape[0] - 1)
    px2 = torch.clamp(px2, 0, sdf.shape[0] - 1)
    py1 = torch.clamp(py1, 0, sdf.shape[1] - 1)
    py2 = torch.clamp(py2, 0, sdf.shape[1] - 1)
    pz1 = torch.clamp(pz1, 0, sdf.shape[2] - 1)
    pz2 = torch.clamp(pz2, 0, sdf.shape[2] - 1)

    # values at interpolation points
    Ia = sdf[px1, py1, pz1]
    Ib = sdf[px2, py1, pz1]
    Ic = sdf[px1, py2, pz1]
    Id = sdf[px2, py2, pz1]
    Ie = sdf[px1, py1, pz2]
    If = sdf[px2, py1, pz2]
    Ig = sdf[px1, py2, pz2]
    Ih = sdf[px2, py2, pz2]

    dx1 = px - px1
    dx2 = px2 - px
    dy1 = py - py1
    dy2 = py2 - py
    dz1 = pz - pz1
    dz2 = pz2 - pz

    # weight is always 1 - |px - pxi|
    wa = dx2 * dy2 * dz2
    wb = dx1 * dy2 * dz2
    wc = dx2 * dy1 * dz2
    wd = dx1 * dy1 * dz2
    we = dx2 * dy2 * dz1
    wf = dx1 * dy2 * dz1
    wg = dx2 * dy1 * dz1
    wh = dx1 * dy1 * dz1

    # actual interpolation
    d_obs_inlim = wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih

    # derivation w.r.t. px
    J_inlim[:, 0, 0] = (
        -1.0 * (dz2 * dy2 * (Ib - Ia) + dz2 * dy1 * (Id - Ic) + dz1 * dy2 * (If - Ie) + dz1 * dy1 * (Ih - Ig)) / res
    )
    # derivation w.r.t. py
    J_inlim[:, 0, 1] = (
        -1.0 * (dz2 * dx2 * (Ic - Ia) + dz2 * dx1 * (Id - Ib) + dz1 * dx2 * (Ig - Ie) + dz1 * dx1 * (Ih - If)) / res
    )
    # derivation w.r.t. pz
    J_inlim[:, 0, 2] = (
        -1.0 * (dy2 * dx2 * (Ie - Ia) + dy2 * dx1 * (If - Ib) + dy1 * dx2 * (Ig - Ic) + dy1 * dx1 * (Ih - Id)) / res
    )

    # respect clamp in derivation, points outside sdf are have distance inf
    d_obs = torch.full((stateb.shape[0],), fill_value=torch.inf, device=device)
    d_obs[in_limits] = d_obs_inlim
    J = torch.zeros((stateb.shape[0], 1, stateb.shape[1]), device=device)
    J[in_limits] = J_inlim

    return d_obs, J


def bilinear_interpolate(imb, stateb, res, x_lims, y_lims, use_cuda=False):
    """bilinear interpolation
    get costs for states in `stateb` by interpolation sdf `imb`.

    Arguments:
      imb: torch.tensor (batch_size, rows, columns)
        signed distance field [y, x]
      stateb: torch.tensor (batch_size, num_traj_states * n_links, wksp_dim)
    Returns:
      distance_to_obstacles: torch.tensor (batch_size, num_traj_states * n_links, 1)
      Jacobian: torch.tensor (batch_size, num_traj_states * n_links, wksp_dim)
    """
    imb = imb.squeeze(1)

    if use_cuda:
        dtype_long = torch.cuda.LongTensor
        device = torch.device("cuda")
    else:
        dtype_long = torch.LongTensor
        device = torch.device("cpu")

    J = torch.zeros_like(stateb)
    MAX_D = x_lims[1] - x_lims[0]  # maximal x distance

    orig_pix_x = 0.0 - x_lims[0] / res  # x coordinate of origin in pixel space
    orig_pix_y = 0.0 - y_lims[0] / res  # y coordinate of origin in pixel space
    orig_pix = torch.tensor([orig_pix_x, orig_pix_y], device=device)

    # position in 'obsticle space'
    px = (orig_pix[0] + stateb[:, :, 0] / res).contiguous().view(-1)
    py = (orig_pix[1] + stateb[:, :, 1] / res).contiguous().view(-1)

    # the four interpolation points in 2d  that surround the goal point (px, py)
    px1 = torch.floor(px).type(dtype_long)
    px2 = px1 + 1
    py1 = torch.floor(py).type(dtype_long)
    py2 = py1 + 1
    px1 = torch.clamp(px1, 0, imb.shape[-1] - 1)
    px2 = torch.clamp(px2, 0, imb.shape[-1] - 1)
    py1 = torch.clamp(py1, 0, imb.shape[1] - 1)
    py2 = torch.clamp(py2, 0, imb.shape[1] - 1)

    # pz = torch.arange(imb.shape[0], device=device).repeat(stateb.shape[1], 1)
    # pz = pz.t().contiguous().view(-1).long()

    # values at interpolation points
    Ia = imb[:, py1, px1]
    Ib = imb[:, py1, px2]
    Ic = imb[:, py2, px1]
    Id = imb[:, py2, px2]

    wa = (px2 - px) * (py2 - py)
    wb = (px - px1) * (py2 - py)
    wc = (px2 - px) * (py - py1)
    wd = (px - px1) * (py - py1)

    # actual interpolation
    d_obs = wa * Ia + wb * Ib + wc * Ic + wd * Id
    d_obs = d_obs.reshape(stateb.shape[0], stateb.shape[1], 1)

    # jacobian computation
    # derivation w.r.t. px
    wja = py2 - py
    wjb = py - py1
    # derivative of d_obs w.r.t. px, the -1.0 is actually incorrect?! but without it, the result is worse!
    J[:, :, 0] = (-1.0 * (wja * (Ib - Ia) + wjb * (Id - Ic)) / res).view(stateb.shape[0], stateb.shape[1])
    # derivation w.r.t. py
    wjc = px2 - px
    wjd = px - px1
    J[:, :, 1] = (-1.0 * (wjc * (Ic - Ia) + wjd * (Id - Ib)) / res).view(stateb.shape[0], stateb.shape[1])

    inlimxu = stateb[:, :, 0] <= x_lims[1]
    inlimxl = stateb[:, :, 0] >= x_lims[0]
    inlimx = (inlimxu + inlimxl) == 1
    inlimyu = stateb[:, :, 1] <= y_lims[1]
    inlimyl = stateb[:, :, 1] >= y_lims[0]
    inlimy = (inlimyu + inlimyl) == 1
    inlimcond = (inlimx + inlimy) == 1
    inlimcond = inlimcond.reshape(stateb.shape[0], stateb.shape[1], 1)

    d_obs = torch.where(inlimcond, d_obs, torch.tensor(MAX_D, device=device))
    J = torch.where(inlimcond, J, torch.zeros(1, 2, device=device))

    return d_obs, J


def caculate_align_mat(pVec_Arr):

    def get_cross_prod_mat(vec):
        # vec shape (3)
        qCross_prod_mat = np.array(
            [
                [0, -vec[2], vec[1]],
                [vec[2], 0, -vec[0]],
                [-vec[1], vec[0], 0],
            ]
        )
        return qCross_prod_mat

    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = (
            np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
        )

    qTrans_Mat *= scale
    return qTrans_Mat


if __name__ == "__main__":
    sdf_ma = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 0]]])
    point = torch.tensor([0.5, 0.5, 0.5])[None, None]
    sdf = SignedDistanceField(matrix=sdf_ma, origin=torch.tensor([0, 0, 0]), voxel_size=1)
    dist, jac = bilinear_interpolate_3d(imb=sdf, stateb=point)
