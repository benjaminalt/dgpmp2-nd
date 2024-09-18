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

from typing import Dict, List

import torch

from diff_gpmp2.gpmp2.factor import Factor


class Factors:
    def __init__(
        self,
        state_dim: int,
        batch_size: int,
        device,
        factors: List[Factor] = [],
    ) -> None:
        self.__factors = factors
        self.__state_dim = state_dim
        self.__batch_size = batch_size
        self.__device = device
        self.__masks = None

    def to(self, device: torch.device):
        self.__device = device
        for factor in self.__factors:
            factor.to(device)
        if self.__masks is not None:
            updated_masks = [updated_masks.append({k: v.to(device) for k, v in mask.items()}) for mask in self.__masks]
            self.__masks = updated_masks

    def threshold_eached(self, errors: Dict[str, torch.Tensor]) -> bool:
        """only evaluates to true if all factors' threshold reached"""
        return all(map(lambda f: f.threshold_reached(error=errors[f.name]), self.__factors))

    def admissible_solution(self, errors: Dict[str, torch.Tensor]) -> bool:
        return all(map(lambda f: f.admissible_solution(error=errors[f.name]), self.__factors))

    def add_prior(self, factor: Factor):
        self.__factors.append(factor)
        if self.__masks is not None:
            self.compute_masks()

    def initialize(self, num_traj_states: int):
        assert num_traj_states is not None, "num_traj_states cannot be None"

        for factor in self.__factors:
            factor.initialize(num_traj_states)

        if self.__masks is None or self.__num_traj_states != num_traj_states:
            self.__num_traj_states = num_traj_states
            self.compute_masks()

    def get_factors(self):
        return self.__factors

    def get_num_constraints(self):
        return self.__num_constraints

    def get_num_traj_states(self):
        return self.__num_traj_states

    def get_N(self):
        return self.__state_dim * self.__num_traj_states

    def get_masks(self) -> List[Dict[str, torch.Tensor]]:
        assert self.__masks is not None, "call compute_masks() first"
        return self.__masks

    def compute_masks(self):
        self.__num_constraints = torch.sum(torch.tensor([f.number_of_constraints() for f in self.__factors]))

        N = self.get_N()
        M = self.__num_constraints

        offset = 0
        masks = []
        for factor in self.__factors:
            consts = factor.number_of_constraints()
            A, K = factor.get_masks(N)

            mask_A = A.to(self.__device)
            mask_K = K.to(self.__device)

            mask_A = mask_A.unsqueeze(0).repeat(self.__batch_size, 1, 1)
            mask_K = mask_K.unsqueeze(0).repeat(self.__batch_size, 1, 1)

            masks.append({"A": mask_A, "K": mask_K})

            offset += consts

        self.__masks = masks

    def compute_error_batch(self, thb: torch.Tensor, verbose: bool = False):
        """Return the non-linear normalized error for the factor graph at the current trajectory"""
        error = 0.0
        for factor in self.__factors:
            error_factor, _, inv_cov = factor.get_error(thb, verbose=verbose)

            weighted_error_factor = 0.5 * torch.einsum(
                "bsij,bsjk->bsik",
                torch.einsum("bsij,bsjk->bsik", error_factor.transpose(2, 3), inv_cov),
                error_factor,
            )

            error += torch.sum(weighted_error_factor)

        return error

    def get_errors(self, thb: torch.Tensor, verbose: bool = False) -> Dict[str, float]:
        errors = {}
        for factor in self.__factors:
            error_factor, _, inv_cov = factor.get_error(thb, verbose)
            weighted_error_factor = 0.5 * torch.einsum(
                "bsij,bsjk->bsik",
                torch.einsum("bsij,bsjk->bsik", error_factor.transpose(2, 3), inv_cov),
                error_factor,
            )
            errors[factor.name] = torch.sum(weighted_error_factor).item()
        return errors
