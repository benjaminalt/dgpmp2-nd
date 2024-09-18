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

import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import transformations as tf
from differentiable_robot_model.robot_model import DifferentiableRobotModel
from ..utils.geometry import get_enclosing_cylinder
from ..utils.mat_utils import pose_to_affine
from urdf_parser_py.urdf import Mesh

class DiffRobotModel:

    def __init__(
        self,
        urdf_path: str,
        good_initial_conf: torch.Tensor,
        name="",
        ee_link_name: str = "default_point",
        attached_objects: Optional[List[Tuple[str, Path, torch.Tensor]]] = None,
        device: Optional[torch.device] = None
    ):
        """
        good_initial_conf: torch.Tensor
            joint configuration that is a good pose for the manipulator, e.g. not in a singularity.
            This conf will be used as initial conf to compute the DLS IK.
        """
        device = torch.ones(1, device=device).device  # re-evaluate device to also include the index
        self.__robot_model = DifferentiableRobotModel(urdf_path, name, device)
        self.__urdf_path = urdf_path
        self.__good_initial_conf = good_initial_conf.to(device)
        assert ee_link_name in self.get_link_names()
        self.ee_link = ee_link_name
        self.n_dofs = self.__robot_model._n_dofs
        self.n_links = len(self.__robot_model._controlled_joints)

        self.set_attached_objects(attached_objects if attached_objects is not None else [])

    @property
    def device(self) -> torch.device:
        return self.__robot_model._device

    @property
    def joint_limits(self) -> torch.Tensor:
        """
        :return: joint limits of the robot as a tensor of shape [self.n_dofs, 2], in radians
        """
        return torch.tensor(
            [[joint["lower"], joint["upper"]] for joint in self.__robot_model.get_joint_limits()],
            device=self.device,
        )

    def save(self, file_path: Path):
        """save the robot model into a zip file to easily share it with others

        :param file_path: the zip file to save to
        """
        assert file_path.suffix == ".zip"
        file_path.unlink(missing_ok=True)

        tmp_dir = Path(tempfile.mkdtemp())
        urdf_path = Path(self.__urdf_path)
        for file in urdf_path.parent.glob("*"):
            if file.is_dir():
                shutil.copytree(file, tmp_dir / file.name)
            else:
                shutil.copyfile(file, tmp_dir / file.name)
        metadata_dict = {
            "good_initial_conf": self.__good_initial_conf.tolist(),
            "ee_link": self.ee_link,
            "name": self.__robot_model.name,
            "device": self.device.type,
            "attached_objects": [],
        }
        attached_dir = tmp_dir / "attached-objects"
        attached_dir.mkdir(exist_ok=True)
        for idx, (link_name, f_path, rel_pose) in enumerate(self.attached_objects):
            f_path = Path(f_path)
            new_file = attached_dir / ("%d_%s" % (idx, f_path.name))
            shutil.copyfile(f_path, new_file)

            metadata_dict["attached_objects"].append(
                {
                    "link_name": link_name,
                    "file_path": new_file.relative_to(tmp_dir).as_posix(),
                    "rel_pose": rel_pose.tolist(),
                }
            )

        with (tmp_dir / "metadata.json").open("w") as f:
            json.dump(metadata_dict, f)

        shutil.make_archive(
            base_name=(file_path.parent / file_path.stem).as_posix(),
            format="zip",
            root_dir=tmp_dir,
        )

        shutil.rmtree(tmp_dir)

    @staticmethod
    def load(file_path: Path, device: Optional[torch.device] = None) -> "DiffRobotModel":
        """load a zip file as robot

        :param file_path: path to zip file
        :return: robot loaded from zip file
        """
        assert file_path.suffix == ".zip"
        assert file_path.exists()
        temp_dir = Path(tempfile.mkdtemp())

        shutil.unpack_archive(file_path, temp_dir)

        with (temp_dir / "metadata.json").open() as f:
            metadata_dict = json.load(f)
        robot_file = temp_dir / "robot.urdf"

        attached_objects = []

        for attached_object in metadata_dict["attached_objects"]:
            mesh_file = temp_dir / attached_object["file_path"]

            attached_objects.append(
                (
                    attached_object["link_name"],
                    mesh_file,
                    torch.as_tensor(attached_object["rel_pose"]),
                )
            )

        return DiffRobotModel(
            urdf_path=robot_file.as_posix(),
            good_initial_conf=torch.as_tensor(metadata_dict["good_initial_conf"]),
            name=metadata_dict["name"],
            ee_link_name=metadata_dict["ee_link"],
            attached_objects=attached_objects,
            device=device,
        )

    def set_attached_objects(self, attached_objects: List[Tuple[str, Path, torch.Tensor]]):
        self.attached_objects = attached_objects

        # precompute robot collision cylinder
        obbs = self.get_oobs_of_links()
        self.cylinders = [(link_name, get_enclosing_cylinder(mesh, device=self.device)) for link_name, mesh in obbs]

    def to(self, device: torch.device) -> "DiffRobotModel":
        # reinitialize robot model with correct device
        self.__robot_model = DifferentiableRobotModel(self.__urdf_path, self.__robot_model.name, device)
        self.__good_initial_conf = self.__good_initial_conf.to(device)
        return self

    def get_link_names(self) -> List[str]:
        # all links in urdf (also world and stuff, not only robot links)
        return self.__robot_model.get_link_names()

    def compute_endeffector_jacobian(self, q: torch.Tensor, link_name: Optional[str] = None) -> torch.Tensor:
        r"""

        Args:
            link_name: name of link name for the jacobian
            q: joint angles [batch_size x n_dofs] or [n_dofs]

        Returns: linear and angular jacobian

        """
        not_batch = False
        if len(q.shape) == 1:
            not_batch = True
            q = q[None]  # if not batch, expand

        if link_name is None:
            link_name = self.ee_link

        lin_jac, ang_jac = self.__robot_model.compute_endeffector_jacobian(q=q, link_name=link_name)

        jac = torch.concat((lin_jac, ang_jac), dim=1)

        return jac[0] if not_batch else jac

    def compute_endeffector_jacobian_all_links(self, q: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs] or [n_dofs]

        Returns: tensor [batch_size x link x n_dofs x 6] or [batch_size x link x n_dofs x 6]
            linear and angular jacobian

        """
        not_batch = False
        if len(q.shape) == 1:
            not_batch = True
            q = q[None]  # if not batch, expand

        lin_jacs, ang_jacs = self.__robot_model.compute_endeffector_jacobian_all_links(q)
        jacs = torch.concat((lin_jacs, ang_jacs), dim=2)

        return jacs[0] if not_batch else jacs

    def compute_forward_kinematics_all_links(self, q: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs] or [n_dofs]

        Returns:
            translation and rotation of link frame [batch_size x n_links x 7] or [n_links x 7]
            quarternion returned in [w, x, y, z] notation

        """
        not_batch = False
        if len(q.shape) == 1:
            not_batch = True
            q = q[None]  # if not batch, expand

        fks = self.__robot_model.compute_forward_kinematics_all_links(q)
        wksp_posb = []
        wksp_rotb = []
        for link_pos, link_rot in fks.values():
            link_rot = link_rot[:, [3, 0, 1, 2]]  # [x,y,z,w] -> [w,x,y,z]
            wksp_posb.append(link_pos)
            wksp_rotb.append(link_rot)

        wksp_posb = torch.stack(wksp_posb, dim=1)
        wksp_rotb = torch.stack(wksp_rotb, dim=1)

        wksp_pose = torch.concat((wksp_posb, wksp_rotb), dim=-1)

        return wksp_pose[0] if not_batch else wksp_pose

    def get_controlled_joint_names(self) -> List[str]:
        controlled_joints = self.__robot_model._controlled_joints
        controlled_joint_names = []
        for name, id in self.__robot_model._name_to_idx_map.items():
            if id in controlled_joints:
                controlled_joint_names.append(name)
        return controlled_joint_names

    def get_oobs_of_links(self) -> List[Tuple[str, o3d.geometry.TriangleMesh]]:
        urdf_dir_path = Path(self.__urdf_path).parent

        obbs = []
        for joint_name in self.get_controlled_joint_names():
            link = self.__robot_model._urdf_model.robot.link_map[joint_name]
            collision = link.collision

            assert hasattr(collision, "geometry"), f"no collision object found for {joint_name}"

            if hasattr(collision, "origin") and collision.origin is not None:
                collision_origin = collision.origin
                transformation = tf.euler_matrix(*collision_origin.rpy)
                transformation[:3, 3] = collision_origin.position
            else:  # no relative transformation of collision geometry
                transformation = np.eye(4)

            collision_geometry = collision.geometry
            if isinstance(collision_geometry, Mesh):
                mesh_filename = collision_geometry.filename
                mesh_path = urdf_dir_path / mesh_filename
                assert mesh_path.exists(), "mesh does not exist"
                if mesh_path.suffix.lower() == ".stl":
                    mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
                    mesh = mesh.transform(transformation)
                    obbs.append((joint_name, mesh))
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

        for link_name, mesh_path, trans in self.attached_objects:
            assert mesh_path.suffix == ".stl", "only stl files accepted"

            mesh = o3d.io.read_triangle_mesh(mesh_path.as_posix())
            trans = pose_to_affine(trans)
            mesh = mesh.transform(trans)

            obbs.append((link_name, mesh))

        return obbs

    def compute_forward_kinematics(
        self, q: torch.Tensor, link_name: Optional[str] = None, recursive: bool = False
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs] or [n_dofs]
            link_name: name of link or None
                if None: last link will be used

        Returns:
            translation and rotation of link frame [batch_size x 7] or [7]
            quarternion returned in [w, x, y, z] notation


        """
        orig_shape = q.shape
        q = q.reshape(-1, self.n_dofs)

        if link_name is None:
            link_name = self.ee_link

        wksp_pos, wksp_rot = self.__robot_model.compute_forward_kinematics(q, link_name, recursive)
        wksp_rot = wksp_rot[:, [3, 0, 1, 2]]  # [x,y,z,w] -> [w,x,y,z]
        wksp_pose = torch.concat((wksp_pos, wksp_rot), dim=-1)

        wksp_pose = wksp_pose.reshape(*orig_shape[:-1], wksp_pose.shape[-1])
        return wksp_pose

    def compute_inverse_kinematics(
        self,
        pose: torch.Tensor,
        link_name: Optional[str] = None,
        init_conf: Optional[torch.Tensor] = None,
        max_num_iter: int = 500,
        min_precision: float = 1e-6,  # distance to goal pose in cart. space  (l1 loss of affine matrices)
    ) -> torch.Tensor:
        """_summary_

        :param pose: workspace pose [..., 7]
        :param link_name: name of link, if None, will use self.ee_link, defaults to None
        :param init_conf: initial configuration to use, either one configuration for all or one for each `pose`, defaults to None
        :param max_num_iter: maximum number of iterations, defaults to 500
        :param min_precision: min prediction required for final solution, defaults to 1e-6
        :return: robot joints in radian [..., 6]
        """
        assert pose.shape[-1] == 7, "pose must be of size 7"

        not_batch = False
        if len(pose.shape) == 1:
            not_batch = True
            pose = pose[None]  # if not batch, expand

        # pose.requires_grad = True
        trans = pose[..., :3]
        rot = pose[..., 3:]
        rot = rot[:, [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]

        if link_name is None:
            link_name = self.ee_link

        if init_conf is None:
            init_conf = self.__good_initial_conf.clone()
        else:
            init_conf = init_conf.squeeze()
            assert init_conf.shape[-1] == self.__robot_model._n_dofs

        confs = self.__robot_model.compute_inverse_kinematics_gd(
            trans=trans,
            rot=rot,
            link_name=link_name,
            init_conf=init_conf,
            max_num_iter=max_num_iter,
            min_precision=min_precision,
        )
        # map to [-2pi, 2pi]
        confs = ((confs + torch.pi * 2) % (torch.pi * 4)) - torch.pi * 2

        return confs[0] if not_batch else confs

    def compute_inverse_kinematics_iteratively(
        self,
        pose: torch.Tensor,
        link_name: Optional[str] = None,
        init_conf: Optional[torch.Tensor] = None,
        max_num_iter: int = 500,
        min_precision: float = 1e-6,  # distance to goal pose in cart. space  (l1 loss of affine matrices)
    ) -> torch.Tensor:
        """compute the ik for a `trajectory`, which ensures that the joint transitions are smooth

        this will iteratively compute the joints for the pose, starting the ik computation from the joints of the previous
        iteration.

        Its significantly more time-consuming, as it does not compute the ik batch-wise.

        :param pose: workspace pose [..., 7]
        :param link_name: name of link, if None, will use self.ee_link, defaults to None
        :param init_conf: initial configuration to use, either one configuration for all or one for each `pose`, defaults to None
        :param max_num_iter: maximum number of iterations, defaults to 500
        :param min_precision: min prediction required for final solution, defaults to 1e-6
        :return: robot joints in radian [..., 6]
        """
        assert len(pose.shape) == 2 and pose.shape[1] == 7
        curr_conf = init_conf
        confs = []
        for p in pose:
            conf = self.compute_inverse_kinematics(
                pose=p,
                link_name=link_name,
                init_conf=curr_conf,
                max_num_iter=max_num_iter,
                min_precision=min_precision,
            )
            confs.append(conf)
            curr_conf = conf
        return torch.stack(confs)

    def display_state(self, q: Optional[torch.Tensor] = None):
        conf = q if q is not None else self.__good_initial_conf
        initial_pose = pose_to_affine(self.compute_forward_kinematics_all_links(conf))
        link_names = self.get_link_names()
        cs = []
        for link_name, (_, c) in self.get_oobs_of_links():
            idx = link_names.index(link_name)

            wksp_pose_quat_link = initial_pose[idx]
            c = c.compute_triangle_normals().compute_vertex_normals()
            c = c.transform(wksp_pose_quat_link.detach().cpu().numpy())
            cs.append(c)

        o3d.visualization.draw_geometries(cs)
        # assert len(q.shape) == 1
        # from matplotlib import pyplot as plt
        # import pybullet as p
        # import time

        # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        # p.setGravity(0, 0, -10)
        # robotId = p.loadURDF(self.__urdf_path)
        # joints = list(range(p.getNumJoints(robotId)))
        # movable_joints = [
        #     joint for joint in joints if p.getJointInfo(robotId, joint)[2] != p.JOINT_FIXED
        # ]

        # assert len(movable_joints) == len(q)
        # plt.show(block=False)
        # plt.pause(1)
        # for joint, value in zip(movable_joints, q):
        #     p.resetJointState(robotId, joint, targetValue=value, targetVelocity=0)
        # time.sleep(10)

        # p.disconnect()
