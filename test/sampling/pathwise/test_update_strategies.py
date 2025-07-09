#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import torch
from botorch import models
from botorch.sampling.pathwise import (
    draw_kernel_feature_paths,
    gaussian_update,
    GeneralizedLinearPath,
    KernelEvaluationMap,
)
from botorch.sampling.pathwise.utils import get_train_inputs, get_train_targets
from botorch.utils.context_managers import delattr_ctx
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.utils.cholesky import psd_safe_cholesky
from linear_operator.operators import ZeroLinearOperator
from torch import Size

from .helpers import gen_module, gen_random_inputs, TestCaseConfig


class TestGaussianUpdates(BotorchTestCase):
    def setUp(self) -> None:
        super().setUp()
        config = TestCaseConfig(seed=0, device=self.device)
        batch_config = replace(config, batch_shape=Size([2]))

        self.base_models = [
            (batch_config, gen_module(models.SingleTaskGP, batch_config)),
            (batch_config, gen_module(models.SingleTaskGP, batch_config)),
            (batch_config, gen_module(models.MultiTaskGP, batch_config)),
            (config, gen_module(models.SingleTaskVariationalGP, config)),
        ]
        self.model_lists = [
            (batch_config, gen_module(models.ModelListGP, batch_config))
        ]

    def test_base_models(self):
        sample_shape = torch.Size([3])
        for config, model in self.base_models:
            tkwargs = {"device": config.device, "dtype": config.dtype}
            if isinstance(model, models.SingleTaskVariationalGP):
                Z = model.model.variational_strategy.inducing_points
                X = (
                    model.input_transform.untransform(Z)
                    if hasattr(model, "input_transform")
                    else Z
                )
                target_values = torch.randn(len(Z), **tkwargs)
                noise_values = None
                Kuu = Kmm = model.model.covar_module(Z)
            else:
                (X,) = get_train_inputs(model, transformed=False)
                (Z,) = get_train_inputs(model, transformed=True)
                target_values = get_train_targets(model, transformed=True)
                noise_values = torch.randn(
                    *sample_shape, *target_values.shape, **tkwargs
                )
                Kmm = model.forward(X if model.training else Z).lazy_covariance_matrix
                Kuu = Kmm + model.likelihood.noise_covar(shape=Z.shape[:-1])

            # Fix noise values used to generate `y = f + e`
            with delattr_ctx(model, "outcome_transform"), patch.object(
                torch,
                "randn_like",
                return_value=noise_values,
            ):
                prior_paths = draw_kernel_feature_paths(model, sample_shape=sample_shape)
                sample_values = prior_paths(X)
                update_paths = gaussian_update(
                    model=model,
                    sample_values=sample_values,
                    target_values=target_values,
                )

            # Test initialization
            self.assertIsInstance(update_paths, GeneralizedLinearPath)
            self.assertIsInstance(update_paths.feature_map, KernelEvaluationMap)
            self.assertTrue(update_paths.feature_map.points.equal(Z))
            self.assertIs(
                update_paths.feature_map.input_transform,
                getattr(model, "input_transform", None),
            )

            # Compare with manually computed update weights `Cov(y, y)^{-1} (y - f - e)`
            Luu = psd_safe_cholesky(Kuu.to_dense())
            errors = target_values - sample_values
            if noise_values is not None:
                errors -= (
                    model.likelihood.noise_covar(shape=Z.shape[:-1]).cholesky()
                    @ noise_values.unsqueeze(-1)
                ).squeeze(-1)
            weight = torch.cholesky_solve(errors.unsqueeze(-1), Luu).squeeze(-1)
            self.assertTrue(weight.allclose(update_paths.weight))

            # Compare with manually computed update values at test locations
            Z2 = gen_random_inputs(model, batch_shape=[16], transformed=True)
            X2 = (
                model.input_transform.untransform(Z2)
                if hasattr(model, "input_transform")
                else Z2
            )
            features = update_paths.feature_map(X2)
            expected_updates = (features @ update_paths.weight.unsqueeze(-1)).squeeze(-1)
            actual_updates = update_paths(X2)
            self.assertTrue(actual_updates.allclose(expected_updates))

            # Test passing `noise_covariance`
            m = Z.shape[-2]
            update_paths = gaussian_update(
                model=model,
                sample_values=sample_values,
                target_values=target_values,
                noise_covariance=ZeroLinearOperator(m, m, dtype=X.dtype),
            )
            Lmm = psd_safe_cholesky(Kmm.to_dense())
            errors = target_values - sample_values
            weight = torch.cholesky_solve(errors.unsqueeze(-1), Lmm).squeeze(-1)
            self.assertTrue(weight.allclose(update_paths.weight))

            if isinstance(model, models.SingleTaskVariationalGP):
                # Test passing non-zero `noise_covariance``
                with patch.object(model, "likelihood", new=BernoulliLikelihood()):
                    with self.assertRaisesRegex(NotImplementedError, "not yet supported"):
                        gaussian_update(
                            model=model,
                            sample_values=sample_values,
                            noise_covariance="foo",
                        )
            else:
                # Test exact models with non-Gaussian likelihoods
                with patch.object(model, "likelihood", new=BernoulliLikelihood()):
                    with self.assertRaises(NotImplementedError):
                        gaussian_update(model=model, sample_values=sample_values)

                with self.subTest("Exact models with `None` target_values"):
                    torch.manual_seed(0)
                    path_none_target_values = gaussian_update(
                        model=model,
                        sample_values=sample_values,
                    )
                    torch.manual_seed(0)
                    path_with_target_values = gaussian_update(
                        model=model,
                        sample_values=sample_values,
                        target_values=get_train_targets(model, transformed=True),
                    )
                    self.assertAllClose(
                        path_none_target_values.weight, path_with_target_values.weight
                    )
