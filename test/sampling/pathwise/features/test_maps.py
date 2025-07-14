#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from math import prod
from unittest.mock import MagicMock, patch

import torch
from botorch.sampling.pathwise.features import maps
from botorch.sampling.pathwise.features.generators import gen_kernel_feature_map
from botorch.sampling.pathwise.utils.transforms import ChainedTransform, FeatureSelector
from botorch.utils.testing import BotorchTestCase
from gpytorch import kernels
from linear_operator.operators import KroneckerProductLinearOperator
from torch import Size
from torch.nn import Module, ModuleList

from ..helpers import gen_module, TestCaseConfig


class TestFeatureMaps(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config = TestCaseConfig(
            seed=0,
            device=self.device,
            num_inputs=2,
            num_tasks=3,
            batch_shape=Size([2]),
        )

        self.base_feature_maps = [
            gen_kernel_feature_map(gen_module(kernels.LinearKernel, self.config)),
            gen_kernel_feature_map(gen_module(kernels.IndexKernel, self.config)),
        ]

    def test_feature_map(self):
        feature_map = maps.FeatureMap()
        feature_map.raw_output_shape = Size([2, 3, 4])
        feature_map.output_transform = None
        feature_map.device = self.device
        feature_map.dtype = None
        self.assertEqual(feature_map.output_shape, (2, 3, 4))

        feature_map.output_transform = lambda x: torch.concat((x, x), dim=-1)
        self.assertEqual(feature_map.output_shape, (2, 3, 8))

    def test_feature_map_list(self):
        map_list = maps.FeatureMapList(feature_maps=self.base_feature_maps)
        self.assertEqual(map_list.device.type, self.config.device.type)
        self.assertEqual(map_list.dtype, self.config.dtype)

        X = torch.rand(
            16,
            self.config.num_inputs,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        output_list = map_list(X)
        self.assertIsInstance(output_list, list)
        self.assertEqual(len(output_list), len(map_list))
        for feature_map, output in zip(map_list, output_list):
            self.assertTrue(feature_map(X).to_dense().equal(output.to_dense()))

    def test_direct_sum_feature_map(self):
        feature_map = maps.DirectSumFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            Size([sum(f.output_shape[-1] for f in feature_map)]),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )
        self.assertTrue(
            features.equal(torch.concat([f(X).to_dense() for f in feature_map], dim=-1))
        )

        # Test mixture of matrix-valued and vector-valued maps
        real_map = feature_map[0]
        
        # Create a proper feature map with 2D output
        class Mock2DFeatureMap(maps.FeatureMap):
            def __init__(self, d, batch_shape):
                super().__init__()
                self.raw_output_shape = Size([d, d])
                self.batch_shape = batch_shape
                self.input_transform = None
                self.output_transform = None
                self.device = real_map.device
                self.dtype = real_map.dtype
                self.d = d
            
            def forward(self, x):
                return x.unsqueeze(-1).expand(*self.batch_shape, *x.shape, self.d)
        
        mock_map = Mock2DFeatureMap(d, real_map.batch_shape)
        with patch.dict(feature_map._modules, {"_feature_maps_list": ModuleList([mock_map, real_map])}):
            self.assertEqual(
                feature_map.output_shape, Size([d, d + real_map.output_shape[0]])
            )
            features = feature_map(X).to_dense()
            self.assertTrue(features[..., :d].equal(mock_map(X)))
            self.assertTrue(
                features[..., d:].eq((d**-0.5) * real_map(X).unsqueeze(-1)).all()
            )

    def test_hadamard_product_feature_map(self):
        feature_map = maps.HadamardProductFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            torch.broadcast_shapes(*(f.output_shape for f in feature_map)),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )
        self.assertTrue(features.equal(prod([f(X).to_dense() for f in feature_map])))

    def test_sparse_direct_sum_feature_map(self):
        feature_map = maps.SparseDirectSumFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            Size([sum(f.output_shape[-1] for f in feature_map)]),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )
        self.assertTrue(
            features.equal(torch.concat([f(X).to_dense() for f in feature_map], dim=-1))
        )

        # Test mixture of matrix-valued and vector-valued maps
        real_map = feature_map[0]
        
        # Create a proper feature map with 2D output
        class Mock2DFeatureMap(maps.FeatureMap):
            def __init__(self, d, batch_shape):
                super().__init__()
                self.raw_output_shape = Size([d, d])
                self.batch_shape = batch_shape
                self.input_transform = None
                self.output_transform = None
                self.device = real_map.device
                self.dtype = real_map.dtype
                self.d = d
            
            def forward(self, x):
                return x.unsqueeze(-1).expand(*self.batch_shape, *x.shape, self.d)
        
        mock_map = Mock2DFeatureMap(d, real_map.batch_shape)
        with patch.dict(feature_map._modules, {"_feature_maps_list": ModuleList([mock_map, real_map])}):
            self.assertEqual(
                feature_map.output_shape, Size([d, d + real_map.output_shape[0]])
            )
            features = feature_map(X).to_dense()
            self.assertTrue(features[..., :d, :d].equal(mock_map(X)))
            self.assertTrue(features[..., d:, d:].eq(real_map(X).unsqueeze(-2)).all())

    def test_outer_product_feature_map(self):
        feature_map = maps.OuterProductFeatureMap(self.base_feature_maps)
        self.assertEqual(
            feature_map.raw_output_shape,
            Size([prod(f.output_shape[-1] for f in feature_map)]),
        )
        self.assertEqual(
            feature_map.batch_shape,
            torch.broadcast_shapes(*(f.batch_shape for f in feature_map)),
        )

        d = self.config.num_inputs
        X = torch.rand((16, d), device=self.config.device, dtype=self.config.dtype)
        features = feature_map(X).to_dense()
        self.assertEqual(
            features.shape[-len(feature_map.output_shape) :],
            feature_map.output_shape,
        )

        test_features = (
            feature_map[0](X).to_dense().unsqueeze(-1)
            * feature_map[1](X).to_dense().unsqueeze(-2)
        ).view(features.shape)
        self.assertTrue(features.equal(test_features))


class TestKernelFeatureMaps(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.configs = [
            TestCaseConfig(
                seed=0,
                device=self.device,
                num_inputs=2,
                num_tasks=3,
                batch_shape=Size([2]),
            )
        ]

    def test_fourier_feature_map(self):
        for config in self.configs:
            tkwargs = {"device": config.device, "dtype": config.dtype}
            kernel = gen_module(kernels.RBFKernel, config)
            weight = torch.randn(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)
            bias = torch.rand(*kernel.batch_shape, 16, **tkwargs)
            feature_map = maps.FourierFeatureMap(
                kernel=kernel, weight=weight, bias=bias
            )
            self.assertEqual(feature_map.output_shape, (16,))

            X = torch.rand(32, config.num_inputs, **tkwargs)
            features = feature_map(X)
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            self.assertTrue(
                features.equal(X @ weight.transpose(-2, -1) + bias.unsqueeze(-2))
            )

    def test_index_kernel_feature_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.IndexKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            feature_map = maps.IndexKernelFeatureMap(kernel=kernel)
            self.assertEqual(feature_map.output_shape, kernel.raw_var.shape[-1:])

            X = torch.rand(*config.batch_shape, 16, config.num_inputs, **tkwargs)
            index_shape = (*config.batch_shape, 16, len(kernel.active_dims))
            indices = X[..., kernel.active_dims] = torch.randint(
                config.num_tasks, size=index_shape, **tkwargs
            )
            indices = indices.long().squeeze(-1)
            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )

            cholesky = kernel.covar_matrix.cholesky().to_dense()
            test_features = []
            for chol, idx in zip(
                cholesky.view(-1, *cholesky.shape[-2:]),
                indices.view(-1, *indices.shape[-1:]),
            ):
                test_features.append(chol.index_select(dim=-2, index=idx))
            test_features = torch.stack(test_features).view(features.shape)
            self.assertTrue(features.equal(test_features))

    def test_kernel_evaluation_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.RBFKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            points = torch.rand(4, config.num_inputs, **tkwargs)
            feature_map = maps.KernelEvaluationMap(kernel=kernel, points=points)
            self.assertEqual(
                feature_map.raw_output_shape, feature_map.points.shape[-2:-1]
            )

            X = torch.rand(16, config.num_inputs, **tkwargs)
            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            self.assertTrue(features.equal(kernel(X, points).to_dense()))

    def test_kernel_feature_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.RBFKernel, config)
            kernel.active_dims = torch.tensor([0], device=config.device)

            feature_map = maps.KernelFeatureMap(kernel=kernel)
            self.assertEqual(feature_map.batch_shape, kernel.batch_shape)
            self.assertIsInstance(feature_map.input_transform, FeatureSelector)
            self.assertIsNone(
                maps.KernelFeatureMap(kernel, ignore_active_dims=True).input_transform
            )
            self.assertIsInstance(
                maps.KernelFeatureMap(kernel, input_transform=Module()).input_transform,
                ChainedTransform,
            )

    def test_linear_kernel_feature_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.LinearKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            active_dims = (
                tuple(range(config.num_inputs))
                if kernel.active_dims is None
                else kernel.active_dims
            )
            feature_map = maps.LinearKernelFeatureMap(
                kernel=kernel, raw_output_shape=Size([len(active_dims)])
            )

            X = torch.rand(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)
            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            self.assertTrue(
                features.equal(kernel.variance.sqrt() * X[..., active_dims])
            )

    def test_multitask_kernel_feature_map(self):
        for config in self.configs:
            kernel = gen_module(kernels.MultitaskKernel, config)
            tkwargs = {"device": config.device, "dtype": config.dtype}
            data_map = gen_kernel_feature_map(
                kernel=kernel.data_covar_module,
                num_ambient_inputs=config.num_inputs,
                num_random_features=config.num_random_features,
            )
            feature_map = maps.MultitaskKernelFeatureMap(
                kernel=kernel, data_feature_map=data_map
            )
            self.assertEqual(
                feature_map.output_shape,
                (feature_map.num_tasks * data_map.output_shape[0],)
                + data_map.output_shape[1:],
            )

            X = torch.rand(*kernel.batch_shape, 16, config.num_inputs, **tkwargs)

            features = feature_map(X).to_dense()
            self.assertEqual(
                features.shape[-len(feature_map.output_shape) :],
                feature_map.output_shape,
            )
            cholesky = kernel.task_covar_module.covar_matrix.cholesky()
            test_features = KroneckerProductLinearOperator(data_map(X), cholesky)
            self.assertTrue(features.equal(test_features.to_dense()))
