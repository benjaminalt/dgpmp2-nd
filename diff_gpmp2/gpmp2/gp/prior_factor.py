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

from typing import Optional

import torch

from diff_gpmp2.gpmp2.factor import Factor
from diff_gpmp2.utils import mat_utils


class PriorFactor(Factor):

    def __init__(
        self,
        name: str,
        state_dim: int,
        thb_point_idx: int,
        K: torch.Tensor,
        device: torch.device,
        threshold: float,
        position_only: bool = True,
        meanb: torch.Tensor = None,
        batch_size=1,
    ):
        super().__init__(name, threshold)
        self.device = device
        self.thb_point_idx = thb_point_idx
        self.state_dim = torch.tensor(state_dim, device=self.device)
        self.end_dim = torch.floor(self.state_dim / 2).to(int) if position_only else self.state_dim

        self.set_mean(meanb)

        self.set_inv_cov(K, batch_size)

    def to(self, device: torch.device) -> "PriorFactor":
        self.inv_cov = self.inv_cov.to(device)
        self.device = device
        if self.meanb is not None:
            self.meanb = self.meanb.to(device)
        return self

    def threshold_reached(self, error: torch.Tensor) -> bool:
        """prior error within threshold"""
        return torch.norm(error) <= self.threshold

    def admissible_solution(self, error: torch.Tensor) -> bool:
        return True

    def _get_error(self, thb, verbose: bool = False):
        error = self.meanb - thb[:, self.thb_point_idx, : self.end_dim]
        H = torch.zeros((thb.shape[0], self.end_dim, self.state_dim), device=self.device)
        H[:, :, : self.end_dim] = torch.eye(self.end_dim, device=self.device)

        if False:  # verbose:
            from matplotlib import pyplot as plt

            plt.plot(error.detach().numpy()[0])
            plt.title(self.name)
            plt.show()

        return error.view(thb.shape[0], 1, self.end_dim, 1), H, self.inv_cov

    def number_of_constraints(self):
        return self.end_dim

    def get_masks(self, columns: int):
        if self.thb_point_idx == -1:  # if end would be -0 the last element is ment
            end = columns
        else:
            end = self.state_dim * (self.thb_point_idx + 1)

        A = torch.zeros((self.number_of_constraints(), columns), dtype=bool)
        A[:, self.state_dim * self.thb_point_idx : end] = True
        K = torch.ones((self.number_of_constraints(), self.number_of_constraints()), dtype=bool)
        return A, K

    def get_inv_cov(self):
        return self.inv_cov.unsqueeze(1)

    def set_mean(self, meanb: Optional[torch.Tensor]):
        if meanb is not None and self.end_dim != self.state_dim:
            meanb = meanb[..., : self.end_dim]

        assert meanb is None or meanb.shape[0] == self.end_dim, f"mean vector must be of size {self.end_dim}"

        self.meanb = meanb

    def set_inv_cov(self, K, batch_size):
        inv_covb = (
            mat_utils.isotropic_matrix(1.0 / torch.pow(K, 2.0), self.end_dim, self.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        self.inv_cov = inv_covb.unsqueeze(1)
