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
from diff_gpmp2.external.pytorch3d_transformations import quaternion_to_matrix, matrix_to_euler_angles


def isotropic_matrix(sig, dim, device=torch.device("cpu")):
    mat = sig * torch.eye(dim, device=device)
    return mat

def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(quaternions)

def pose_to_affine(pose: torch.Tensor) -> torch.Tensor:
    assert pose.shape[-1] == 7
    rot = quaternion_to_rotation_matrix(pose[..., 3:])
    trans = pose[..., :3].unsqueeze(-1)
    # torch.eye doesn't work as it does not retain grads
    affine = torch.as_tensor([0, 0, 0, 1], dtype=pose.dtype, device=pose.device).repeat((*pose.shape[:-1], 1, 1))
    affine = torch.concat((torch.concat((rot, trans), dim=-1), affine), dim=-2)
    return affine

def pose_quaternion_to_euler_zyx(pose: torch.Tensor) -> torch.Tensor:
    """
    Convert a pose of format (x,y,z,qw,qx,qy,qz) into a pose of format (x,y,z,rx,ry,rz)
    where the orientation is expressed in Euler angles (ZYX convention).
    :param pose: Pose of format (x,y,z,qw,qx,qy,qz)
    :return: Pose of format (x,y,z,rx,ry,rz)
    """
    return torch.concat((pose[..., :3], quaternion_to_euler_zyx(pose[..., 3:7])), dim=-1)

def quaternion_to_euler_zyx(q: torch.Tensor) -> torch.Tensor:
    """
    Convert q quaternion (w,x,y,z) to Euler angles (ZYX convention, in radians, normalized to [-pi, pi])
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion
    :param q: Quaternion (w,x,y,z)
    :return Euler angles (ZYX convention, in radians, normalized to [-pi, pi])
    """
    rotations = matrix_to_euler_angles(quaternion_to_matrix(q), convention="ZYX").flip(-1)
    return rotations