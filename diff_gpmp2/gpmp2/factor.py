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

from abc import abstractmethod
from typing import Optional

import torch


class Factor(object):
    def __init__(self, name: str, threshold: float, error_sum_weight: float = 1.0) -> None:
        self.name = name
        self.threshold = threshold
        self.error_sum_weight = error_sum_weight
        self.__num_traj_states = None

    def to(self, device: torch.device):
        return self

    @abstractmethod
    def threshold_reached(self, error: torch.Tensor) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def admissible_solution(self, error: torch.Tensor) -> bool:
        raise NotImplementedError()

    def initialize(self, num_traj_states: int):
        """
        (Re-)initialize the factor.
        """
        assert num_traj_states is not None, "num_traj_states cannot be None"

        self.__num_traj_states = num_traj_states

    def _get_num_traj_states(self) -> Optional[int]:
        return self.__num_traj_states

    def get_error(self, thb: torch.Tensor, verbose: bool = False):
        def run():
            return self._get_error(thb, verbose)

        return run()

    @abstractmethod
    def _get_error(self, thb: torch.Tensor, verbose: bool = False):
        """
        given the batch trajection, this function returns the estimated error and jacobian

        Arguments
        ===
            thb: torch.tensor (batch_size, num_traj_states, state_dim)

        Returns
        ===
            error: torch.tensor (batch_size, num_traj_influcenced, ...)
            jacobian: torch.tensor (batch_size, ...)
        """
        raise NotImplementedError()

    @abstractmethod
    def number_of_constraints(self):
        raise NotImplementedError()

    @abstractmethod
    def get_masks(self, columns: int):
        """
        returns masks to setup LGS

        Returns
        ===
            A: torch.tensor (number_of_constraints, columns)
            K: torch.tensor (number_of_constraints, number_of_constraints)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_inv_cov(self):
        """
        returns the inverse covariance matrix

        Returns
        ===
            inv_cov: torch.tensor (batch_size, num_traj_influenced)

        """
        raise NotImplementedError()
