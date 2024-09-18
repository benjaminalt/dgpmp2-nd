#!/usr/bin/env python

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

import torch

from diff_gpmp2.utils.sdf import SignedDistanceField, bilinear_interpolate_3d


class HingeLossObstacleCost:

    def hinge_loss_signed(self, sphere_centers, r_vec, eps):
        raise NotImplementedError()
        eps_tot = eps + r_vec
        dist_signed, J = self.env.get_signed_obstacle_distance_vec(sphere_centers)
        cost = torch.where(
            dist_signed <= eps_tot,
            eps_tot - dist_signed,
            torch.zeros(dist_signed.shape, device=self.device),
        )
        H = torch.where(dist_signed <= eps_tot, -1.0 * J, torch.zeros(J.shape, device=self.device))
        return cost, H

    def hinge_loss_signed_full(self, sphere_centers, r_vec, eps):
        raise NotImplementedError()
        eps_tot = eps + r_vec
        z2 = torch.zeros(1, sphere_centers.shape[-1], device=self.device)
        dist_signed, J = self.env.get_signed_obstacle_distance(
            sphere_centers.view(sphere_centers.shape[0] * sphere_centers.shape[1], 1, -1)
        )
        cost = torch.where(
            dist_signed <= eps_tot,
            eps_tot - dist_signed,
            torch.tensor(0.0, device=self.device),
        )
        H = torch.where(dist_signed <= eps_tot, -1.0 * J, z2)
        return cost.reshape(sphere_centers.shape[0], sphere_centers.shape[1]), H

    def hinge_loss_signed_batch(
        self,
        sphere_centersb: torch.Tensor,
        sphere_radiib: torch.Tensor,
        epsb: float,
        sdf: SignedDistanceField,
    ):
        """
        Arguments:
            sphere_centersb: torch.tensor (samples, wksp_dim)
            r_vec: sphere radii vector (samples)
            epsb: epsilon safety distance to obstacles (float)
            sdfb: signed distance field batch
        Returns:
            cost: torch.tensor (batch_size, num_traj_states, nlinks, 1)
            H: torch.tensor (batch-size, num_traj_states, nlinks, nlinks*wksp_dim)
        """

        sphere_radii_with_margin = epsb + sphere_radiib

        if sphere_centersb.shape[-1] >= 3:  # 3d state
            dist_signed, J = bilinear_interpolate_3d(
                sdf,
                sphere_centersb,
                sdf.device,
            )

        else:  # 2d state
            raise NotImplementedError()
            # dist_signed, J = bilinear_interpolate(
            #     sdf,  # sdfb[:, 0, :, :],
            #     qpts,
            #     res,
            #     self.env_params["x_lims"],
            #     self.env_params["y_lims"],
            #     self.use_cuda,
            # )

        cost = torch.where(
            dist_signed <= sphere_radii_with_margin,
            sphere_radii_with_margin - dist_signed,
            torch.tensor(0.0, device=sdf.device),
        )
        cost = torch.round(cost, decimals=5)  # round to compensate for numerical instability

        H = -1.0 * J
        H[dist_signed > sphere_radii_with_margin] = 0

        return cost, dist_signed, H
