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
POSSIBILITY OF SUCH DAMAGE."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from diff_gpmp2.gpmp2.factors import Factors
from diff_gpmp2.gpmp2.obstacle.obstacle_factor import ObstacleFactor
from diff_gpmp2.gpmp2.plan_layer import OptimParams, PlanLayer
from diff_gpmp2.gpmp2.plot_matplotlib import plot_matplotlib

logger = logging.getLogger(__name__)


class DiffGPMP2Planner(nn.Module):

    def __init__(
        self,
        factors: Factors,
        optim_params: OptimParams,
        state_dim: int,
        device: torch.device,
        batch_size=1,
        plot_output_dir: Optional[Path] = None,
        verbose=False,
    ):
        super(DiffGPMP2Planner, self).__init__()
        self.optim_params = optim_params
        state_dim = state_dim
        self.plot_output_dir = plot_output_dir
        if self.plot_output_dir and not self.plot_output_dir.exists():
            self.plot_output_dir.mkdir(parents=True)
        self.iter_num = 0
        self.verbose = verbose

        self.plan_layer = PlanLayer(
            factors,
            state_dim,
            optim_params=optim_params,
            device=device,
            batch_size=batch_size,
            verbose=verbose,
        )

    def to(self, device: torch.device) -> "DiffGPMP2Planner":
        self.plan_layer.to(device)
        return self

    def forward(self, initial_trajectory, verbose: bool = False, silent: bool = True):
        """
        imb: binary image of environment
        sdfb: signed distance field
        """
        current_trajectory = initial_trajectory.unsqueeze(0)
        num_states = initial_trajectory.shape[0]

        err_per_iter = {}

        min_error = torch.inf
        min_errors = None
        min_error_traj = initial_trajectory
        min_error_iter = 0
        error_patience = 0
        error_patience_error = torch.inf
        lr = self.optim_params.init_lr

        # initalize factors
        self.plan_layer.factors.initialize(num_states)

        good_criterion_reached = False

        for j in range(self.optim_params.max_iters):
            # compute trajectory delta
            last_errors, A, b, K = self.plan_layer.construct_linear_system_batch(current_trajectory)

            factors = self.plan_layer.factors.get_factors()
            last_error = sum(
                [
                    torch.sum(torch.abs(last_error)).item() * factor.error_sum_weight
                    for factor, last_error in zip(factors, last_errors.values())
                ]
            )

            errors = {key: torch.sum(torch.abs(value)).item() for key, value in last_errors.items()}
            print(errors)

            # Obstacle and robot joint limits are "relevant errors"
            relevant_error = last_error if "Obst." not in errors else errors["Obst."]
            relevant_error += errors["Goal"] if "Goal" in errors else 0
            relevant_error += errors["Start"] if "Start" in errors else 0
            relevant_error += errors["Lims"] if "Lims" in errors else 0
            errors["total"] = last_error

            if relevant_error == 0 and min_error > 0:
                lr *= 0.1

            error_patience += 1
            # if relevant_error > min_error: # TODO: or not self.plan_layer.factors.admissible_solution(last_errors):
            if relevant_error <= min_error:
                # only use trajectory if obst loss decreased or total loss decreased:
                if min_errors is None or (relevant_error < min_error or errors["total"] < min_errors["total"]):
                    if relevant_error < min_error or (
                        min_errors is not None and errors["total"] < error_patience_error * 0.95
                    ):
                        # reset error patience if total loss decreased by 5 percent since last error patience reset or relevant loss decreased
                        if min_errors is not None and not silent:
                            logger.debug(
                                "reduced from %f to %f (Rel: %f)"
                                % (min_errors["total"], errors["total"], relevant_error)
                            )
                        error_patience = 0
                        error_patience_error = errors["total"]
                    min_error = relevant_error
                    min_errors = errors
                    min_error_traj = current_trajectory
                    min_error_iter = j

                if self.plan_layer.factors.threshold_eached(last_errors):
                    # this is the typical case how to exist the for loop
                    # if this is true, all factors are within some acceptable error threshold,
                    # so optimization can stop
                    for key, value in last_errors.items():
                        if key not in err_per_iter:
                            err_per_iter[key] = []
                        err_per_iter[key].append(torch.sum(torch.abs(value)))
                    if verbose or self.verbose and not silent:
                        logger.debug("%d: factors' thresholds reached" % j)
                    good_criterion_reached = True
                    break

            if error_patience >= self.optim_params.error_patience:
                if verbose or self.verbose and not silent:
                    logger.info(
                        "%d: error did not improve for %d iterations: %f"
                        % (j, self.optim_params.error_patience, last_error)
                    )
                break

            for key, value in last_errors.items():
                if key not in err_per_iter:
                    err_per_iter[key] = []
                err_per_iter[key].append(torch.sum(torch.abs(value)).detach())

            if self.verbose and self.plot_output_dir:
                output_subdir = self.plot_output_dir / str(self.iter_num)
                output_subdir.mkdir(exist_ok=True)
                output_subsubdir = output_subdir / str(j)
                output_subsubdir.mkdir(exist_ok=True)
                plot_matplotlib(
                    last_errors,
                    A,
                    b,
                    K,
                    current_trajectory,
                    self.plan_layer.factors,
                    output_subsubdir,
                )

            # update trajectory by delta
            # update after min_error_traj is set, since error value of last trajectory
            dtheta = self.plan_layer.solve_linear_system_batch(A, b, K, delta=self.optim_params.reg)
            current_trajectory = current_trajectory + lr * dtheta

        if not good_criterion_reached and not silent:
            logger.warning("dGPMP finished without factors' thresholds reached %s" % (str(min_errors)))

        if not silent:
            logger.warning(f"Final error: {min_error} at iteration {min_error_iter} of {j}")

        if self.verbose and self.plot_output_dir:
            output_subdir = self.plot_output_dir / str(self.iter_num)
            output_subdir.mkdir(exist_ok=True)
            if j > 0:
                self.plot_errors(err_per_iter, output_subdir / "planner_errors.png", min_error_iter)
                self.dump_errors(err_per_iter, output_subdir / "planner_errors.json", min_error_iter)
                obstacle_factor = None
                for factor in self.plan_layer.factors.get_factors():
                    if isinstance(factor, ObstacleFactor):
                        obstacle_factor = factor
                if obstacle_factor is not None:
                    obstacle_factor.plot_trajectory(min_error_traj, output_subdir / "obstacle")

        if not verbose:
            err_per_iter = {}

        else:
            self.plan_layer.error_batch(min_error_traj, verbose=True)

        self.iter_num += 1
        return min_error_traj.squeeze(), err_per_iter, j

    def error_batch(self, thb):
        return self.plan_layer.error_batch(thb)

    def plot_errors(self, err_per_iter: Dict, output_filepath: Path, min_error_idx: int):
        from matplotlib import pyplot as plt

        ncols = len(err_per_iter)
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(6 * ncols, 4))
        for i, (key, errors) in enumerate(err_per_iter.items()):
            axes[i].plot(range(len(errors)), [e.item() for e in errors], label=key)
            axes[i].legend()
            axes[i].axvline(min_error_idx, color="black")
        fig.savefig(output_filepath.as_posix())
        plt.close(fig)

    def dump_errors(self, err_per_iter: Dict, output_filepath: Path, min_error_idx: int):
        with open(output_filepath, "w") as output_file:
            json.dump(
                {
                    "errors": {key: [val.item() for val in values] for key, values in err_per_iter.items()},
                    "min_error_idx": min_error_idx,
                    "min_error": {key: values[min_error_idx].item() for key, values in err_per_iter.items()},
                },
                output_file,
            )
