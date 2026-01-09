# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

ADV_ESTIMATOR_REGISTRY = {}


def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}")
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    config=None,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2rawscore = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            # Keep raw (pre-normalization) sequence scores for debug
            id2rawscore[index[i]].append(float(scores[i].item()))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        # Keep a copy of normalized per-sample scores (1D) for debug selection
        norm_scores_1d = scores.clone().detach()
        scores = scores.unsqueeze(-1) * response_mask

        # Debug print after calculation if enabled via config
        if config is not None and bool(config.get("debug_tree_adv", False)) and len(id2score) > 0:
            max_i = int(torch.argmax(norm_scores_1d).item())
            max_score = norm_scores_1d[max_i].item()
            grp_key = index[max_i]
            grp_id = str(grp_key)
            grp_vals_norm = [float(v.item()) for v in id2score[grp_key]]
            grp_vals_raw = id2rawscore[grp_key]
            print(f"[adv_debug] max_norm_score={max_score} grpo max_i={max_i} group_id={grp_id} raw_scores={grp_vals_raw} norm_scores={grp_vals_norm}", flush=True)
            raise ValueError(f"[adv_debug] max_norm_score={max_score} grpo max_i={max_i} group_id={grp_id} raw_scores={grp_vals_raw} norm_scores={grp_vals_norm}")

    
    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6, config=None, **kwargs):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower



def compute_policy_loss_gspo(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective for vanilla GSPO (sequence-level optimization).
    See https://arxiv.org/pdf/2507.18071 for more details.
    
    Implements equations 6-8 from the paper:
    J_GSPO(θ) = E[x~D, {y_i}_{i=1}^G ~ π_θold(·|x)] [1/G Σ_{i=1}^G min(s_i(θ)Â_i, clip(s_i(θ), 1-ε, 1+ε)Â_i)]
    where s_i(θ) = (π_θ(y_i|x)/π_θold(y_i|x))^(1/|y_i|)
    
    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config (Optional[DictConfig | ActorConfig]): 
            Configuration parameters
    """
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
    


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data

@register_adv_est("grpo_iterative_old")
def compute_grpo_old(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray = None,
    *,
    tree_metadata: dict = None,
    responses: torch.Tensor = None,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Tree-based advantage for iterative-branching outcome supervision:
    - Treat each response as a path from root to a leaf in a tree of decisions (per prompt).
    - Leaf reward: sum of token-level rewards of that complete response (masked by response_mask).
    - For any internal node (prefix), its reward is the average of all leaf rewards under that node.
    - Advantage on each token t equals R(node_t+1) - R(node_t), i.e., reward difference between
      child node and parent node along the specific path. Expand that scalar to a token at position t.

    Inputs are expected from iterative-branching sampler with r-major batching and
    tree_metadata:
      - tree_metadata['per_sample'][b] contains:
          - prompt_index: int
          - parent_traj: int (>=0) or -1 for root trajectory within that prompt
          - branch_pos: int (token index on response)
          - prefix_len: int (equals branch_pos)
      - tree_metadata['n_per_prompt']: trajectories per prompt (n)

    Returns:
      advantages: (B, T) token-wise, only position t has non-zero at the exact branching/step token; masked by response_mask
      returns: same as advantages for PPO-style outcome supervision
    """
    # If no tree info, do not fallback; raise error explicitly
    if tree_metadata is None or "per_sample" not in tree_metadata:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            sys.stdout.flush()
            print("[adv_debug] grpo_iterative_branching fallback_to_grpo=False (error)")
        raise ValueError("tree_metadata is required for grpo_iterative_branching")

    B, T = token_level_rewards.shape
    # Normalize possible numpy object arrays to Python types
    per_sample_raw = tree_metadata.get("per_sample", [])
    try:
        import numpy as _np
        if isinstance(per_sample_raw, _np.ndarray):
            per_sample_raw = per_sample_raw.tolist()
    except Exception:
        pass

    n_val = tree_metadata.get("n_per_prompt", 1)
    try:
        import numpy as _np
        if isinstance(n_val, _np.ndarray):
            # handle 0-d or 1-d object arrays
            n = int(n_val.item() if n_val.shape == () else _np.asarray(n_val).reshape(-1)[0])
        else:
            n = int(n_val)
    except Exception:
        n = int(n_val)

    assert len(per_sample_raw) == B and n >= 1, f"Invalid tree metadata alignment,len(per_sample)={len(per_sample_raw)}, B={B}, n={n}"

    # Debug: basic inputs
    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            sys.stdout.flush()
            print(f"[adv_debug] start B={B} T={T} n={n} per_sample_len={len(per_sample_raw)} responses_present={responses is not None}")
    except Exception:
        pass

    # Try to detect the flatten order. If entries have explicit prompt_index, we can reconstruct p-major order;
    # otherwise assume r-major (original implementation).
    has_prompt_index = isinstance(per_sample_raw, list) and len(per_sample_raw) > 0 and isinstance(per_sample_raw[0], dict) and ("prompt_index" in per_sample_raw[0])
    if has_prompt_index:
        # p-major: for p in [0..bs-1], for r in [0..n-1]
        # Compute bs by scanning max prompt_index
        try:
            max_p = max(int(x.get("prompt_index", 0)) for x in per_sample_raw)
            bs = max_p + 1
        except Exception:
            bs = B // n if n > 0 else B
        # Reindex into r-major equivalent view to reuse the rest of the logic
        per_sample = [None] * B
        for idx_b, meta in enumerate(per_sample_raw):
            p = int(meta.get("prompt_index", 0))
            # position within this prompt's block
            r = idx_b - p * n
            # guard
            if r < 0 or r >= n:
                r = idx_b % n
            b_r_major = r * bs + p
            if b_r_major < B:
                per_sample[b_r_major] = meta
        # Fill any None due to inconsistencies by fallback to raw
        for i in range(B):
            if per_sample[i] is None:
                per_sample[i] = per_sample_raw[i]
    else:
        # assume r-major layout as produced by generate_sequences_by_iterative_entropy
        bs = B // n if n > 0 else B
        per_sample = per_sample_raw

    # Debug check: when debug_tree_adv is enabled, assert every trajectory starts from root
    # try:
    #     do_debug_check = bool(config.get("debug_tree_adv", False)) if isinstance(config, dict) else bool(getattr(config, "debug_tree_adv", False))
    # except Exception:
    #     do_debug_check = False
    # if do_debug_check:
    #     violations = []
    #     for idx_b, meta in enumerate(per_sample):
    #         if not isinstance(meta, dict):
    #             continue
    #         pr = int(meta.get("parent_traj", -1))
    #         bp = int(meta.get("branch_pos", -1))
    #         pf = int(meta.get("prefix_len", 0))
    #         if not (pr == -1 and bp <= 0 and pf <= 0):
    #             violations.append((idx_b, pr, bp, pf))
    #     if len(violations) > 0:
    #         examples = ", ".join([f"b={b} pr={pr} bp={bp} pf={pf}" for b, pr, bp, pf in violations[:8]])
    #         raise ValueError(
    #             f"[root_check][err] expected all trajectories to start from root (parent_traj=-1, branch_pos<=0, prefix_len<=0); "
    #             f"found {len(violations)}/{len(per_sample)} violations. examples: {examples}"
    #         )

    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            sys.stdout.flush()
            print(f"[adv_debug] layout ok bs={bs} (B={B}, n={n})")
    except Exception:
        pass

    # Compute leaf rewards per sample (sum over masked tokens)
    with torch.no_grad():
        masked_rewards = token_level_rewards * response_mask
        leaf_reward_per_sample = masked_rewards.sum(dim=-1)  # (B,)
    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            sys.stdout.flush()
            nz_mask = response_mask.sum(dim=1)
            print(f"[adv_debug] rewards: leaf_reward min={float(leaf_reward_per_sample.min()) if leaf_reward_per_sample.numel()>0 else 0:.6f} max={float(leaf_reward_per_sample.max()) if leaf_reward_per_sample.numel()>0 else 0:.6f} mask_nz_min={int(nz_mask.min()) if nz_mask.numel()>0 else -1} mask_nz_max={int(nz_mask.max()) if nz_mask.numel()>0 else -1}")
            # dump first few prompts' rewards per branch
            dump_prompts = min(bs, 2)
            for p in range(dump_prompts):
                vals = [float(leaf_reward_per_sample[r * bs + p].item()) for r in range(n)]
                print(f"[adv_debug] p={p} leaf_rewards(n={n}): {vals}")
    except Exception:
        pass

    # Prepare response lengths and per-sample prefix keys
    # Use responses tensor if available to know effective lengths for masking
    if responses is None:
        # Infer effective length from response_mask per sample
        eff_len = response_mask.sum(dim=1).to(torch.int64)
    else:
        eff_len = (responses != 0).sum(dim=1).to(torch.int64)  # rough fallback; mask preferred
        eff_len = torch.minimum(eff_len, response_mask.sum(dim=1).to(torch.int64))

    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            sys.stdout.flush()
            print(f"[adv_debug] eff_len: min={int(eff_len.min()) if eff_len.numel()>0 else -1} max={int(eff_len.max()) if eff_len.numel()>0 else -1}")
            print("[adv_debug] building traj_info...")
    except Exception:
        pass

    # First, build parent pointers at trajectory granularity within each prompt
    # traj_info[p][r] = {parent_r, branch_pos}; root has parent_r=-1
    traj_info = [[None for _ in range(n)] for _ in range(bs)]
    for r in range(n):
        for p in range(bs):
            b = r * bs + p
            meta = per_sample[b]
            traj_info[p][r] = {
                "parent_r": int(meta.get("parent_traj", -1)),
                "branch_pos": int(meta.get("branch_pos", -1)),
            }

    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            for p in range(min(bs, 2)):
                parents = [traj_info[p][r]["parent_r"] for r in range(n)]
                bpos = [traj_info[p][r]["branch_pos"] for r in range(n)]
                print(f"[adv_debug] traj_info p={p} parents={parents} branch_pos={bpos}")
            sys.stdout.flush()
            print("[adv_debug] traj_info built. building prefix_to_leaf_indices...")
    except Exception:
        pass

    # Build, for each (p, r), a mapping from prefix length k to a canonical (p, r_base, k) key by walking up parents
    def canonical_key_for_prefix(p: int, r: int, k: int) -> tuple:
        # Walk up the tree until we find the ancestor where this prefix originates
        cur_r = r
        visited = set()
        step = 0
        while True:
            # Debug/cycle guard before following parent
            if cur_r in visited or step > (n + bs + 10):
                try:
                    if config is not None and bool(config.get("debug_tree_adv", False)):
                        sys.stdout.flush()
                        print(f"[adv_debug][warn] canonical_key_for_prefix cycle/long path p={p} r={r} k={k} cur_r={cur_r} visited={len(visited)} step={step}")
                except Exception:
                    pass
                return (p, cur_r, k)
            visited.add(cur_r)
            step += 1

            parent_r = traj_info[p][cur_r]["parent_r"]
            branch_pos = traj_info[p][cur_r]["branch_pos"]
            if parent_r < 0:
                # root owns prefixes 0..len(root)
                return (p, 0, k)
            if k <= branch_pos:
                # This prefix lies entirely before this branching; continue to parent
                cur_r = parent_r
                continue
            else:
                # This prefix passes the branch point, so it's owned by this child
                return (p, cur_r, k)

    # Aggregate leaf rewards per canonical prefix key
    from collections import defaultdict

    prefix_to_leaf_indices: Dict[tuple, List[int]] = defaultdict(list)
    for r in range(n):
        for p in range(bs):
            b = r * bs + p
            L = int(eff_len[b].item())
            for k in range(0, L + 1):  # include node at length k (node before token k)
                key = canonical_key_for_prefix(p, r, k)
                prefix_to_leaf_indices[key].append(b)

    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            sys.stdout.flush()
            print(f"[adv_debug] prefix_to_leaf_indices size={len(prefix_to_leaf_indices)}")
            # sample a few keys
            shown = 0
            for key, leaves in prefix_to_leaf_indices.items():
                sys.stdout.flush()
                print(f"[adv_debug] key={key} leaves={len(leaves)}")
                shown += 1
                if shown >= 4:
                    break
    except Exception:
        pass

    # Compute mean leaf rewards per prefix key
    prefix_reward: Dict[tuple, torch.Tensor] = {}
    for key, leaves in prefix_to_leaf_indices.items():
        vals = leaf_reward_per_sample[leaves]
        prefix_reward[key] = vals.mean()

    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            sys.stdout.flush()
            print(f"[adv_debug] prefix_reward size={len(prefix_reward)}. assigning advantages...")
            shown = 0
            for key, val in prefix_reward.items():
                sys.stdout.flush()
                print(f"[adv_debug] prefix_reward sample key={key} val={float(val):.6f}")
                shown += 1
                if shown >= 4:
                    break
    except Exception:
        pass

    # Branch-token-based advantage:
    # For tokens between two consecutive branch tokens k0 and k1, set
    # advantage = value(k1) - value(k0), where value(k) is the mean leaf
    # reward under the prefix node at length k.
    advantages = torch.zeros_like(token_level_rewards)

    # Collect branch token positions per prompt (include root k=0)
    branch_positions_by_prompt = {p: set([0]) for p in range(bs)}
    for p in range(bs):
        for r in range(n):
            info = traj_info[p][r]
            if info["parent_r"] >= 0:
                k = int(info["branch_pos"])
                if k >= 0:
                    branch_positions_by_prompt[p].add(k)

    # Map (prompt, k) -> prefix value (mean leaf reward)
    prefix_value_by_prompt_k = {p: {} for p in range(bs)}
    for (p_key, r_base, k_key), val in prefix_reward.items():
        prefix_value_by_prompt_k[p_key][k_key] = val

    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            for p in range(min(bs, 3)):
                ks = sorted(list(branch_positions_by_prompt[p]))
                sys.stdout.flush()
                print(f"[adv_debug] p={p} branch_positions={ks} prefix_values={[(k, float(prefix_value_by_prompt_k[p].get(k, torch.tensor(float('nan'))))) for k in ks]}")
    except Exception:
        pass

    # Assign advantages per sequence between consecutive nodes (branch nodes and leaf)
    for r in range(n):
        for p in range(bs):
            b = r * bs + p
            L = int(eff_len[b].item())
            if L <= 0:
                continue

            points = sorted([k for k in branch_positions_by_prompt[p] if k <= L])
            if len(points) == 0 or points[0] != 0:
                points = [0] + points
            if points[-1] != L:
                points.append(L)
            if len(points) < 2:
                continue

            for i in range(len(points) - 1):
                k0 = points[i]
                k1 = points[i + 1]
                if k0 >= L or k1 <= k0:
                    continue

                v0 = prefix_value_by_prompt_k.get(p, {}).get(k0, None)
                if v0 is None:
                    continue
                if k1 == L:
                    v1 = leaf_reward_per_sample[b]
                else:
                    v1 = prefix_value_by_prompt_k.get(p, {}).get(k1, None)
                    if v1 is None:
                        continue

                adv_val = v1 - v0
                start_t = k0
                end_t = min(k1, L) - 1
                if end_t >= start_t:
                    advantages[b, start_t : end_t + 1] = adv_val

    # Optional weight sharing across shared edges to avoid over-counting common prefixes
    share_edge = False
    try:
        if config is not None:
            if isinstance(config, dict):
                share_edge = bool(config.get("share_branch_edge_weight", config.get("edge_weight_sharing", False)))
            else:
                share_edge = bool(getattr(config, "share_branch_edge_weight", getattr(config, "edge_weight_sharing", False)))
    except Exception:
        pass

    if share_edge:
        from collections import defaultdict as _dd
        # Segment-wise sharing: group by (prompt, k0, k1). Divide entire segment by number of responses sharing it.
        segment_to_members: Dict[tuple, set] = _dd(set)
        for r in range(n):
            for p in range(bs):
                b = r * bs + p
                L = int(eff_len[b].item())
                if L <= 0:
                    continue
                points = sorted([k for k in branch_positions_by_prompt[p] if k <= L])
                if len(points) == 0 or points[0] != 0:
                    points = [0] + points
                if points[-1] != L:
                    points.append(L)
                for i in range(len(points) - 1):
                    k0 = points[i]
                    k1 = points[i + 1]
                    if k0 >= L or k1 <= k0:
                        continue
                    segment_to_members[(p, k0, k1)].add(b)
        for (p_seg, k0_seg, k1_seg), members in segment_to_members.items():
            dup = len(members)
            if dup > 1:
                for b in members:
                    Lb = int(eff_len[b].item())
                    start_t = k0_seg
                    end_t = min(k1_seg, Lb) - 1
                    if end_t >= start_t:
                        advantages[b, start_t : end_t + 1] = advantages[b, start_t : end_t + 1] / dup

    try:
        if config is not None and bool(config.get("debug_tree_adv", False)):
            total_nonzero = int((advantages.abs() > 0).sum().item())
            sys.stdout.flush()
            print(f"[adv_debug] advantages nnz={total_nonzero} (of {advantages.numel()})")
            if total_nonzero == 0:
                # Possible reasons: no branch points, missing prefix values, zero reward diffs
                empty_prompts = [p for p in range(bs) if len(branch_positions_by_prompt[p]) < 2]
                print(f"[adv_debug] empty_prompts(len={len(empty_prompts)}): {empty_prompts[:8]}")
                missing_vals = []
                for p in range(min(bs, 4)):
                    ks = sorted(list(branch_positions_by_prompt[p]))
                    miss = [(k, (k not in prefix_value_by_prompt_k.get(p, {}))) for k in ks]
                    missing_vals.append((p, miss))
                print(f"[adv_debug] prefix_value coverage sample: {missing_vals}")
    except Exception:
        pass

    advantages = advantages * response_mask
    returns = advantages.clone()
    return advantages, returns

@register_adv_est("grpo_iterative_branching")
def compute_grpo_iterative_branching_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index=None,
    *,
    tree_metadata: dict = None,
    responses: torch.Tensor = None,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Tensor-oriented version aligned with project signature.
    - token_level_rewards: (B, T) tensor
    - response_mask: (B, T) tensor of 0/1
    - tree_metadata: { per_sample: List[dict], n_per_prompt: int }
    - responses: optional (B, T) tensor (unused for JSON recompute)

    Returns: (advantages, returns) tensors of shape (B, T)
    """
    if tree_metadata is None or "per_sample" not in tree_metadata:
        raise ValueError("tree_metadata with per_sample is required")

    if not isinstance(token_level_rewards, torch.Tensor) or not isinstance(response_mask, torch.Tensor):
        raise TypeError("token_level_rewards and response_mask must be torch.Tensor")

    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    B, T = token_level_rewards.shape
    if B == 0:
        return torch.zeros_like(token_level_rewards), torch.zeros_like(token_level_rewards)

    # Debug flags
    dbg = False
    dbg_limit = 8
    try:
        if isinstance(config, dict):
            dbg = bool(config.get("debug", config.get("debug_tree_adv", False)))
            dbg_limit = int(config.get("debug_limit", 8))
    except Exception:
        pass
    if dbg:
        sys.stdout.flush()
        print(f"[core.debug] B={B} T={T}")

    per_sample = tree_metadata.get("per_sample", [])
    if not isinstance(per_sample, list) or len(per_sample) != B:
        raise ValueError(f"per_sample must be a list of len B; got {type(per_sample)} len={len(per_sample) if hasattr(per_sample,'__len__') else 'NA'} B={B}")

    n_val = tree_metadata.get("n_per_prompt", 1)
    try:
        if isinstance(n_val, np.ndarray):
            if n_val.shape == ():
                n = int(n_val.item())
            else:
                n = int(np.asarray(n_val).reshape(-1)[0])
        else:
            n = int(n_val)
    except Exception:
        # Fallback for unexpected types
        n = int(np.asarray(n_val).reshape(-1)[0])
    if n <= 0 or (B % n != 0):
        raise ValueError(f"Invalid n_per_prompt: n={n}, B={B}")
    bs = B // n

    # If provided in p-major order (grouped by prompt), reindex to r-major (grouped by response index)
    # Build grouping by prompt_index if available, and define accessor b_index(p, r)
    has_prompt_index = (
        isinstance(per_sample, list)
        and len(per_sample) > 0
        and isinstance(per_sample[0], dict)
        and ("prompt_index" in per_sample[0])
    )
    use_groups = False
    group_b_indices: List[List[int]] = []
    p_to_slot: Dict[int, int] = {}
    if has_prompt_index:
        try:
            prompt_vals = [int(x.get("prompt_index", 0)) for x in per_sample]
        except Exception:
            prompt_vals = [0] * B
        uniq_prompts = sorted(set(prompt_vals))
        p_to_slot = {pval: i for i, pval in enumerate(uniq_prompts)}
        bs_from_prompts = len(uniq_prompts)
        if bs_from_prompts * n == B:
            bs = bs_from_prompts
            group_b_indices = [[] for _ in range(bs)]
            for b_idx, meta in enumerate(per_sample):
                try:
                    p_raw = int(meta.get("prompt_index", 0))
                except Exception:
                    p_raw = 0
                p_slot = p_to_slot.get(p_raw, 0)
                group_b_indices[p_slot].append(b_idx)
            ok = all(len(lst) == n for lst in group_b_indices)
            if ok:
                use_groups = True
                if dbg:
                    sys.stdout.flush()
                    print(f"[core.debug] grouped per_sample by prompt_index into {bs} prompts × {n} responses")
            else:
                if dbg:
                    sys.stdout.flush()
                    sizes = [len(lst) for lst in group_b_indices]
                    print(f"[core.debug][warn] uneven group sizes per prompt: {sizes}, expect all == {n}; fallback to flat indexing")
        else:
            if dbg:
                sys.stdout.flush()
                print(f"[core.debug][warn] prompt_count({bs_from_prompts})*n({n}) != B({B}); fallback to flat indexing")

    def b_index(p: int, r: int) -> int:
        if use_groups:
            return group_b_indices[p][r]
        # flat r-major assumption
        return r * bs + p
 
    # Compute final reward per sample using mask
    mask_bool = response_mask.to(torch.bool)
    # Leaf reward should be the sum of masked token rewards
    final_rewards = (token_level_rewards * response_mask).sum(dim=1)
    if dbg:
        fr_min = float(final_rewards.min().item()) if final_rewards.numel() else 0.0
        fr_max = float(final_rewards.max().item()) if final_rewards.numel() else 0.0
        sys.stdout.flush()
        print(f"[core.debug] final_rewards: min={fr_min:.6f} max={fr_max:.6f}")

    # Build traj_info[p][r] with parent and branch_pos
    traj_info: List[List[Dict[str, int]]] = [[{"parent_r": -1, "branch_pos": -1} for _ in range(n)] for _ in range(bs)]
    for p in range(bs):
        for r in range(n):
            b = b_index(p, r)
            meta = per_sample[b]
            pr = -1
            bp = -1
            try:
                pr = int(meta.get("parent_traj", -1))
            except Exception:
                pr = -1
            try:
                bp = int(meta.get("branch_pos", -1))
            except Exception:
                bp = -1
            traj_info[p][r] = {"parent_r": pr, "branch_pos": bp}
    # Validate parent_traj indices per prompt
    for p in range(bs):
        for r in range(n):
            pr = traj_info[p][r]["parent_r"]
            if pr >= n:
                if dbg:
                    sys.stdout.flush()
                    print(f"[core.debug][warn] invalid parent_r={pr} for p={p} r={r}, n={n}; clamping to -1")
                traj_info[p][r]["parent_r"] = -1
    if dbg:
        for p in range(min(bs, dbg_limit)):
            parents = [traj_info[p][r]["parent_r"] for r in range(n)]
            bpos = [traj_info[p][r]["branch_pos"] for r in range(n)]
            sys.stdout.flush()
            print(f"[core.debug] traj_info p={p} parents={parents} branch_pos={bpos}")

    # Per-response branch positions (include root 0, own branch_pos, and all ancestors)
    branch_positions_for_response: List[List[int]] = [[] for _ in range(B)]
    for p in range(bs):
        for r in range(n):
            b = b_index(p, r)
            positions_set = set([0])
            cur_r = r
            steps = 0
            visited = set()
            while cur_r >= 0 and (cur_r not in visited) and steps <= (n + bs + 10):
                visited.add(cur_r)
                info = traj_info[p][cur_r]
                k = info.get("branch_pos", -1)
                try:
                    k = int(k)
                except Exception:
                    k = -1
                if k is not None and k >= 0:
                    positions_set.add(int(k))
                cur_r = int(info.get("parent_r", -1))
                steps += 1
            branch_positions_for_response[b] = sorted(list(positions_set))

    # Upward propagation with path constraint
    positions_sets = [set(pos_list) for pos_list in branch_positions_for_response]
    for p in range(bs):
        for r in range(n):
            b = b_index(p, r)
            info = traj_info[p][r]
            pr = int(info.get("parent_r", -1))
            k = int(info.get("branch_pos", -1))
            if pr < 0 or k < 0:
                continue
            cur = pr
            steps = 0
            visited = set()
            while True:
                if cur < 0 or (cur in visited) or steps > (n + bs + 10):
                    break
                visited.add(cur)
                b_cur = b_index(p, cur)
                if 0 <= b_cur < B:
                    positions_sets[b_cur].add(k)
                parent_of_cur = int(traj_info[p][cur].get("parent_r", -1))
                cur_bp = int(traj_info[p][cur].get("branch_pos", -1))
                # Only propagate upward if k lies on parent's path: k <= branch_pos(cur,parent)
                if parent_of_cur < 0 or not (k <= cur_bp):
                    break
                cur = parent_of_cur
                steps += 1

    # Downward inheritance: child inherits parent's positions strictly less than its own k
    for p in range(bs):
        for r in range(n):
            b = b_index(p, r)
            info = traj_info[p][r]
            pr = int(info.get("parent_r", -1))
            k_child = int(info.get("branch_pos", -1))
            if pr < 0 or k_child < 0:
                continue
            parent_b = b_index(p, pr)
            if 0 <= parent_b < B:
                for pos in positions_sets[parent_b]:
                    if pos < k_child:
                        positions_sets[b].add(pos)

    branch_positions_for_response = [sorted(list(s)) for s in positions_sets]
    if dbg:
        for b in range(min(B, dbg_limit)):
            sys.stdout.flush()
            # derive (p, r) for this b for clarity
            if use_groups:
                # build inverse map lazily
                pr = None
                for p in range(bs):
                    if b in group_b_indices[p]:
                        r = group_b_indices[p].index(b)
                        pr = (p, r)
                        break
                if pr is None:
                    print(f"[core.debug] b={b} branch_positions={branch_positions_for_response[b]}")
                else:
                    print(f"[core.debug] b={b} (p={pr[0]}, r={pr[1]}) branch_positions={branch_positions_for_response[b]}")
            else:
                p = b % bs
                r = b // bs
                print(f"[core.debug] b={b} (p={p}, r={r}) branch_positions={branch_positions_for_response[b]}")
        # also dump per-prompt per-r mapping for first few prompts
        for p in range(min(bs, dbg_limit)):
            for r in range(min(n, dbg_limit)):
                b = b_index(p, r)
                sys.stdout.flush()
                print(f"[core.debug] p={p} r={r} b={b} positions={branch_positions_for_response[b]}")
 
    # Recompute branch values using per-response positions
    # p_idx for each b
    p_idx_for_b: List[int] = [0] * B
    if use_groups:
        for p in range(bs):
            for r in range(n):
                b = b_index(p, r)
                p_idx_for_b[b] = p
    else:
        for r in range(n):
            for p in range(bs):
                b = r * bs + p
                p_idx_for_b[b] = p

    prompt_to_kset: Dict[int, set] = {}
    for b in range(B):
        p_idx = p_idx_for_b[b]
        s = prompt_to_kset.setdefault(p_idx, set())
        for k in branch_positions_for_response[b]:
            s.add(k)

    branch_value: Dict[Tuple[int, int], float] = {}
    fr_list = final_rewards.detach().cpu().tolist()
    for p_idx, kset in prompt_to_kset.items():
        for k in sorted(kset):
            vals: List[float] = []
            for b in range(B):
                if p_idx_for_b[b] != p_idx:
                    continue
                if k in branch_positions_for_response[b]:
                    vals.append(float(fr_list[b]))
            if len(vals) > 0:
                branch_value[(p_idx, k)] = float(sum(vals) / len(vals))
    if dbg:
        show = 0
        for (pidx, k), v in branch_value.items():
            sys.stdout.flush()
            print(f"[core.debug] V(p={pidx},k={k})={v:.6f}")
            show += 1
            if show >= dbg_limit:
                break

    # Effective lengths
    eff_len = response_mask.sum(dim=1).to(torch.int64)
    if responses is not None:
        resp_len = (responses != 0).sum(dim=1).to(torch.int64)
        eff_len = torch.minimum(eff_len, resp_len)

    # Assign advantages per sample
    advantages = torch.zeros_like(token_level_rewards)
    for p in range(bs):
        for r in range(n):
            b = b_index(p, r)
            p_idx = p if not use_groups else p
            L = int(eff_len[b].item())
            if L <= 0:
                continue
            points = [k for k in branch_positions_for_response[b] if k <= L]
            if len(points) == 0 or points[0] != 0:
                points = [0] + points
            if points[-1] != L:
                points.append(L)
            if len(points) < 2:
                continue
            for i in range(len(points) - 1):
                k0 = points[i]
                k1 = points[i + 1]
                if k0 >= L or k1 <= k0:
                    continue
                v0 = branch_value.get((p_idx, k0), None)
                if v0 is None:
                    continue
                if k1 == L:
                    v1 = float(final_rewards[b].item())
                else:
                    v1 = branch_value.get((p_idx, k1), None)
                    if v1 is None:
                        continue
                adv_val = float(v1 - v0)
                if dbg and i < dbg_limit and b < dbg_limit:
                    sys.stdout.flush()
                    print(f"[core.debug] b={b} seg[{k0},{k1}) v0={v0:.6f} v1={v1:.6f} adv={adv_val:.6f}")
                start_t = k0
                end_t = min(k1, L) - 1
                if end_t >= start_t:
                    advantages[b, start_t : end_t + 1] = torch.tensor(adv_val, dtype=dtype, device=device)

    length_penalty = config.get("length_penalty", 0.0)
    if length_penalty > 0.0:
        # Penalize longer correct rollouts relative to the shortest correct rollout per prompt.
        print(f"[core.debug] length_penalty={length_penalty}")
        sys.stdout.flush()
        for p in range(bs):
            correct_bs = []
            shortest_b = -1
            shortest_L = None
            for r in range(n):
                b = b_index(p, r)
                fr = float(final_rewards[b].item())
                if abs(fr) > 1e-6:
                    Lb = int(eff_len[b].item())
                    correct_bs.append(b)
                    if shortest_L is None or Lb < shortest_L:
                        shortest_L = Lb
                        shortest_b = b
            if len(correct_bs) <= 1 or shortest_b < 0 or shortest_L is None or shortest_L <= 0:
                continue
            set_short = set(branch_positions_for_response[shortest_b])
            L_short = int(shortest_L)
            for b in correct_bs:
                if b == shortest_b:
                    continue
                L_b = int(eff_len[b].item())
                if L_b <= 0:
                    continue
                common_positions = set_short.intersection(set(branch_positions_for_response[b]))
                if not common_positions:
                    k_common = 0
                else:
                    lim = min(L_b, L_short)
                    valid_common = [k for k in common_positions if k <= lim]
                    k_common = max(valid_common) if len(valid_common) > 0 else 0
                remain_short = max(L_short - k_common, 0)
                remain_other = max(L_b - k_common, 0)
                if remain_other <= 0:
                    continue
                penalty_factor = 1 - (float(remain_short) / float(remain_other))**length_penalty
                print(f"[core.debug] penalty_factor={penalty_factor}")
                sys.stdout.flush()
                if penalty_factor <= 0:
                    continue
                start_t = int(k_common)
                end_t = int(L_b)
                if end_t > start_t:
                    advantages[b, start_t:end_t] -= advantages[b, start_t:end_t].abs() * penalty_factor
    advantages = advantages * response_mask
    returns = advantages.clone()

    # Expose branch values for optional debugging/inspection
    compute_grpo_iterative_branching_outcome_advantage.branch_value = branch_value  # type: ignore[attr-defined]
    return advantages, returns 