# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import List

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from verl import DataProto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = int(self.config.get("max_num_batched_tokens", 8192))

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp")
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": prompts.meta_info.get("top_k", self.config.val_kwargs.top_k),
                "top_p": prompts.meta_info.get("top_p", self.config.val_kwargs.top_p),
                "temperature": prompts.meta_info.get("temperature", self.config.val_kwargs.temperature),
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            # self.inference_engine.llm_engine.list_loras
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = output[0].to(idx.device)
            log_probs = output[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.n > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

    @GPUMemoryLogger(role="vllm rollout entropy branching", logger=logger)
    @torch.no_grad()
    def generate_sequences_by_entropy(self, prompts: DataProto, max_new_tokens: int = None, top_k: int = None, **kwargs) -> DataProto:
        """
        返回类型与generate_sequences完全一致，batch字段包含：
          - 'prompts': (bs * n, prompt_len)
          - 'responses': (bs * n, response_len)
          - 'input_ids': (bs * n, prompt_len+response_len)
          - 'rollout_log_probs': (bs * n, response_len)
          - 'attention_mask': (bs * n, prompt_len+response_len)
          - 'position_ids': (bs * n, prompt_len+response_len)
        meta_info包含分支点、entropy等分析信息。
        n为该次采样配置的返回序列数。
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        from verl.utils.torch_functional import pad_sequence_to_length, get_response_mask

        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        # Step 0: Initial Setup
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        idx_list = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]

        sampling_params = deepcopy(self.sampling_params)
        if max_new_tokens is not None:
            sampling_params.max_tokens = int(max_new_tokens)
        else:
            sampling_params.max_tokens = int(self.config.response_length)
        
        n_param = self.sampling_params.n
        if top_k is None:
            top_k = max(0, int(n_param) - 1)
        else:
            top_k = int(top_k)

        # Step 1: Generate one sequence per prompt to find branch points
        sampling_params_step1 = deepcopy(sampling_params)
        sampling_params_step1.logprobs = 1  # We need full distribution for entropy
        sampling_params_step1.n = 1
        
        generate_kwargs = {
            "prompts": None, "sampling_params": sampling_params_step1, "prompt_token_ids": idx_list,
            "lora_request": None, "use_tqdm": False
        }
        if hasattr(self.inference_engine, 'generate') and 'return_full_logprobs' in self.inference_engine.generate.__code__.co_varnames:
            generate_kwargs['return_full_logprobs'] = True

        output = self.inference_engine.generate(**generate_kwargs)
        original_responses = output[0].to(idx.device)  # (bs, resp_len)
        log_probs = output[1]  # (bs, resp_len, vocab_size)

        if log_probs.dim() != 3 or log_probs.shape[0] != batch_size:
            raise NotImplementedError("Logprobs must have shape (batch_size, resp_len, vocab_size) for entropy branching.")

        # If we don't need to branch, format and return the original sequences
        if top_k == 0:
            final_batch = self._prepare_output_batch(
                prompts_tensor=idx,
                responses_tensor=original_responses,
                log_probs_tensor=torch.gather(log_probs, 2, original_responses.unsqueeze(-1)).squeeze(-1),
                eos_token_id=eos_token_id,
                position_ids_template=position_ids,
            )
            return DataProto(batch=final_batch)

        # Step 2: Calculate entropy and find branch points for each sequence
        probs = log_probs.exp()
        entropies = -(probs * probs.log()).sum(dim=-1)  # (bs, resp_len)
        
        response_mask = get_response_mask(response_id=original_responses, eos_token=eos_token_id, dtype=torch.bool)
        entropies[~response_mask] = -1.0  # Ignore padding tokens
        
        all_branch_prompts_list = []
        original_full_sequences = torch.cat([idx, original_responses], dim=1)
        prompt_len = idx.shape[1]

        for i in range(batch_size):
            entropies_i = entropies[i].cpu().numpy()
            num_valid_tokens = response_mask[i].sum().item()
            # Ensure we don't pick more branch points than available tokens
            current_top_k = min(top_k, num_valid_tokens)
            if current_top_k <= 0: continue
            
            top_k_indices_i = np.argpartition(-entropies_i, range(current_top_k))[:current_top_k]
            for entropy_idx in top_k_indices_i:
                branch_point = prompt_len + entropy_idx + 1
                new_prompt = original_full_sequences[i, :branch_point]
                all_branch_prompts_list.append(new_prompt)

        # If no valid branch points were found across the batch
        if not all_branch_prompts_list:
             final_batch = self._prepare_output_batch(
                prompts_tensor=idx, responses_tensor=original_responses,
                log_probs_tensor=torch.gather(log_probs, 2, original_responses.unsqueeze(-1)).squeeze(-1),
                eos_token_id=eos_token_id, position_ids_template=position_ids,
            )
             return DataProto(batch=final_batch)

        # Step 3: Generate new sequences from branch points
        sampling_params_step2 = deepcopy(sampling_params)
        sampling_params_step2.n = 1
        sampling_params_step2.logprobs = 0 # Only need token logprobs now

        branched_output = self.inference_engine.generate(
            prompts=None,
            sampling_params=sampling_params_step2,
            prompt_token_ids=[p.tolist() for p in all_branch_prompts_list],
            use_tqdm=False,
        )
        branched_responses = branched_output[0].to(idx.device)
        branched_log_probs = branched_output[1].to(idx.device)

        # Step 4: Combine original and branched sequences into a single batch
        # This part is complex due to varied lengths and the need for careful interleaving.
        # Let's handle it by creating two separate DataProto objects and merging them.
        
        # Package original results
        original_batch = self._prepare_output_batch(
            prompts_tensor=idx, responses_tensor=original_responses,
            log_probs_tensor=torch.gather(log_probs, 2, original_responses.unsqueeze(-1)).squeeze(-1),
            eos_token_id=eos_token_id, position_ids_template=position_ids,
        )

        # Package branched results
        padded_branch_prompts = torch.stack(
            [pad_sequence_to_length(p, -1, self.pad_token_id, right_pad=False) for p in all_branch_prompts_list]
        ).to(idx.device)

        branched_batch = self._prepare_output_batch(
            prompts_tensor=padded_branch_prompts, responses_tensor=branched_responses,
            log_probs_tensor=branched_log_probs, eos_token_id=eos_token_id,
            # For branched position_ids, we can't easily use the template, so we pass None
            position_ids_template=None
        )

        # Here we assume a simple concatenation. A more sophisticated interleaving might be needed
        # depending on downstream consumer expectations, but this is a robust starting point.
        final_batch_size = original_batch.batch_size[0] + branched_batch.batch_size[0]
        final_batch = original_batch.union(branched_batch).to_tensordict()
        final_batch.set("batch_size", torch.tensor([final_batch_size]))
        
        meta_info = {"eos_token_id": eos_token_id}
        return DataProto(batch=final_batch, meta_info=meta_info)

    def _prepare_output_batch(self, prompts_tensor: torch.Tensor, responses_tensor: torch.Tensor,
                              log_probs_tensor: torch.Tensor, eos_token_id: int, 
                              position_ids_template: torch.Tensor = None) -> TensorDict:
        """Helper to construct a standard output TensorDict from generated sequences."""
        batch_size = prompts_tensor.shape[0]
        prompt_len = prompts_tensor.shape[1]
        resp_len = responses_tensor.shape[1]

        # Pad all tensors to consistent lengths before cat
        max_len = prompt_len + resp_len
        input_ids = torch.cat([prompts_tensor, responses_tensor], dim=1)
        
        # Attention Mask
        prompt_mask = (prompts_tensor != self.pad_token_id).to(torch.long)
        response_mask = get_response_mask(response_id=responses_tensor, eos_token=eos_token_id, dtype=torch.long)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        
        # Position IDs
        if position_ids_template is not None and position_ids_template.shape[0] == batch_size:
             # Reuse original prompt position_ids and extend them
            response_pos_delta = torch.arange(1, resp_len + 1, device=prompts_tensor.device).unsqueeze(0)
            if position_ids_template.dim() == 3: # Handle multimodal case
                last_pos = position_ids_template[:, :, -1:]
                response_pos_delta = response_pos_delta.view(1, 1, -1)
            else:
                last_pos = position_ids_template[:, -1:]
            
            response_position_ids = last_pos + response_pos_delta
            position_ids = torch.cat([position_ids_template, response_position_ids], dim=-1)
        else: # Fallback for branched prompts where template doesn't match
            prompt_pos_ids = torch.cumsum(prompt_mask, dim=1) - 1
            prompt_pos_ids[prompt_pos_ids < 0] = 0
            last_prompt_pos_id = (prompt_mask.sum(1) - 1).clamp(min=0)
            response_pos_delta = torch.arange(1, resp_len + 1, device=prompts_tensor.device).unsqueeze(0)
            response_pos_ids = last_prompt_pos_id.unsqueeze(1) + response_pos_delta
            position_ids = torch.cat([prompt_pos_ids, response_pos_ids], dim=1)

        return TensorDict(
            {
                "prompts": prompts_tensor,
                "responses": responses_tensor,
                "input_ids": input_ids,
                "rollout_log_probs": log_probs_tensor,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

