#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import ceil
from typing import List, Tuple

import torch
from botorch.sampling.pathwise.features.generators import gen_kernel_feature_map
from botorch.sampling.pathwise.utils import is_finite_dimensional, kernel_instancecheck
from botorch.utils.testing import BotorchTestCase
from gpytorch import kernels

from ..helpers import gen_module, TestCaseConfig


class TestGenKernelFeatureMap(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        config = TestCaseConfig(
            seed=0,
            device=self.device,
            num_inputs=2,
            num_tasks=3,
            batch_shape=torch.Size([2]),
        )

        self.kernels: List[Tuple[TestCaseConfig, kernels.Kernel]] = []
        for typ in (
            kernels.LinearKernel,
            kernels.IndexKernel,
            kernels.MaternKernel,
            kernels.RBFKernel,
            kernels.ScaleKernel,
            kernels.ProductKernel,
            kernels.MultitaskKernel,
            kernels.AdditiveKernel,
            kernels.LCMKernel,
        ):
            self.kernels.append((config, gen_module(typ, config)))

    def test_gen_kernel_feature_map(self, slack: float = 3.0):
        for config, kernel in self.kernels:
            with torch.random.fork_rng():
                torch.random.manual_seed(config.seed)
                feature_map = gen_kernel_feature_map(
                    kernel,
                    num_ambient_inputs=config.num_inputs,
                    num_random_features=config.num_random_features,
                )
                self.assertEqual(feature_map.batch_shape, kernel.batch_shape)

                n = 4
                m = ceil(n * kernel.batch_shape.numel() ** -0.5)

                input_batch_shapes = [(n**2,)]
                if not isinstance(kernel, kernels.MultitaskKernel):
                    input_batch_shapes.append((m, *kernel.batch_shape, m))

                for input_batch_shape in input_batch_shapes:
                    X = torch.rand(
                        (*input_batch_shape, config.num_inputs),
                        device=kernel.device,
                        dtype=kernel.dtype,
                    )
                    if isinstance(kernel, kernels.IndexKernel):  # random task IDs
                        X[..., kernel.active_dims] = torch.randint(
                            kernel.raw_var.shape[-1],
                            size=(*X.shape[:-1], len(kernel.active_dims)),
                            device=X.device,
                            dtype=X.dtype,
                        )

                    num_tasks = (
                        config.num_tasks
                        if kernel_instancecheck(kernel, kernels.MultitaskKernel)
                        else 1
                    )
                    test_shape = (
                        *kernel.batch_shape,
                        num_tasks * X.shape[-2],
                        *feature_map.output_shape,
                    )
                    if len(input_batch_shape) > len(kernel.batch_shape) + 1:
                        test_shape = (m,) + test_shape

                    features = feature_map(X).to_dense()
                    self.assertEqual(features.shape, test_shape)
                    covar = kernel(X).to_dense()

                    istd = covar.diagonal(dim1=-2, dim2=-1).rsqrt()
                    corr = istd.unsqueeze(-1) * covar * istd.unsqueeze(-2)
                    vec = istd.unsqueeze(-1) * features.view(*covar.shape[:-1], -1)
                    est = vec @ vec.transpose(-2, -1)
                    allclose_kwargs = {}
                    if not is_finite_dimensional(kernel):
                        num_random_features_per_map = config.num_random_features / (
                            1
                            if not is_finite_dimensional(kernel, max_depth=0)
                            else sum(
                                not is_finite_dimensional(k)
                                for k in kernel.modules()
                                if k is not kernel
                            )
                        )
                        allclose_kwargs["atol"] = (
                            slack * num_random_features_per_map**-0.5
                        )

                    if isinstance(kernel, (kernels.MultitaskKernel, kernels.LCMKernel)):
                        allclose_kwargs["atol"] = max(
                            allclose_kwargs.get("atol", 1e-5), slack * 2.0
                        )

                    self.assertTrue(corr.allclose(est, **allclose_kwargs))
