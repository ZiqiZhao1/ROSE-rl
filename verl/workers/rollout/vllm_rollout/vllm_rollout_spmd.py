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
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import imp
import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from re import S
from typing import Any, Dict, List, Union
import sys
import copy
import math

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from .embedding import get_vocab_embeddings_from_vllm

import asyncio
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
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
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=min(20, config.logprobs),  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        self.response_max_tokens = config.response_length

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        # Keep a reference to tokenizer for decoding prompts in iterative branching
        self.tokenizer = tokenizer
        print(f"[DEBUG] pad_token_id: {self.pad_token_id}")
        # print(f"[DEBUG] self.config.adaptive_temperature: {self.config.adaptive_temperature}")
        # print(f"[DEBUG] self.config.adaptive_temperature_high: {self.config.adaptive_temperature_high}")
        if hasattr(tokenizer, 'vocab_size'):
            print(f"[DEBUG] tokenizer vocab_size: {tokenizer.vocab_size}")
        # Cache vocab size for prompt id filtering
        self.vocab_size = None
        self.vocab_size = max(tokenizer.added_tokens_decoder.keys()) + 1
        print(f"[DEBUG] vocab_size: {self.vocab_size}")

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
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

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
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    curr_log_prob = []
                    for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        print(f"[DEBUG] input_ids: {batch['input_ids'].shape}")
        print(f"[DEBUG] prompts: {batch['prompts'].shape}")
        print(f"[DEBUG] responses: {batch['responses'].shape}")
        """
        [DEBUG] input_ids: torch.Size([256, 4096])
        [DEBUG] prompts: torch.Size([256, 2048])
        [DEBUG] responses: torch.Size([256, 2048])
        """   

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @GPUMemoryLogger(role="vllm rollout entropy branching", logger=logger)
    @torch.no_grad()
    def generate_sequences_by_entropy(self, prompts: DataProto, **kwargs) -> DataProto:
         # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

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
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size
        
        n = kwargs.get("n", self.sampling_params.n)
        if n <= 1:
            # If n is 1 or less, just use the regular generate_sequences method
            return self.generate_sequences(prompts, **kwargs)
            
         # Step 1: Generate one sequence per prompt
        sampling_params_step1 = deepcopy(kwargs)
        sampling_params_step1["n"] = 1
        with self.update_sampling_params(**sampling_params_step1):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

        # Step 2: Extract responses and calculate entropy for each token
        first_responses = []
        first_log_probs = []
        token_entropies = []
        
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_ids = output.outputs[sample_id].token_ids
                first_responses.append(response_ids)
                
                # Extract log probs and calculate entropy for each token position
                curr_log_probs = []
                curr_entropies = []
                
                for i, logprob_dict in enumerate(output.outputs[sample_id].logprobs):
                    # Get the log prob of the selected token
                    token_id = response_ids[i]
                    token_log_prob = logprob_dict[token_id].logprob
                    curr_log_probs.append(token_log_prob)
                    
                    # Calculate entropy from all token probabilities at this position
                    probs = [np.exp(lp.logprob) for lp in logprob_dict.values()]
                    log_probs = [lp.logprob for lp in logprob_dict.values()]
                    entropy = -sum(p * lp for p, lp in zip(probs, log_probs))
                    curr_entropies.append(entropy)
                
                first_log_probs.append(curr_log_probs)
                token_entropies.append(curr_entropies)
        
        # Step 3: For each prompt, find the n-1 positions with highest entropy
        all_responses = [first_responses]
        all_log_probs = [first_log_probs]
        
        for prompt_idx in range(batch_size):
            # Get entropy values for current prompt's response
            entropy_values = token_entropies[prompt_idx]
            response_length = len(entropy_values)
            
            if response_length == 0:
                continue  # Skip if response is empty
                
            # Find top n-1 positions with highest entropy
            if response_length <= n-1:
                # If response is shorter than n-1, use all positions
                high_entropy_positions = list(range(response_length))
            else:
                # Sort positions by entropy (highest first)
                high_entropy_positions = sorted(
                    range(response_length),
                    key=lambda i: entropy_values[i],
                    reverse=True
                )[:n-1]
            
            # Sort positions in ascending order to process from left to right
            high_entropy_positions.sort()
            
            # Step 4: For each high entropy position, generate a new response
            for pos in high_entropy_positions:
                # Create a prompt that includes the original prompt + response up to the high entropy position
                original_prompt_ids = vllm_inputs[prompt_idx]["prompt_token_ids"]
                prefix_response_ids = first_responses[prompt_idx][:pos]
                
                new_prompt_ids = original_prompt_ids + prefix_response_ids
                
                # Prepare the new prompt for vLLM
                new_vllm_input = {"prompt_token_ids": new_prompt_ids}
                if "multi_modal_data" in vllm_inputs[prompt_idx]:
                    new_vllm_input["multi_modal_data"] = vllm_inputs[prompt_idx]["multi_modal_data"]
                
                # Generate continuation from this position
                with self.update_sampling_params(**sampling_params_step1):
                    continuation_output = self.inference_engine.generate(
                        prompts=[new_vllm_input],
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests[prompt_idx:prompt_idx+1] if lora_requests else None,
                        use_tqdm=False,
                    )
                
                # Extract the generated continuation
                continuation_ids = continuation_output[0].outputs[0].token_ids
                
                # Collect log probs for the continuation
                continuation_log_probs = []
                for i, logprob in enumerate(continuation_output[0].outputs[0].logprobs):
                    token_id = continuation_ids[i]
                    continuation_log_probs.append(logprob[token_id].logprob)
                
                # Append the new response and its log probs to the respective batch lists
                if prompt_idx < len(all_responses):
                    all_responses.append([])
                    all_log_probs.append([])
                
                all_responses[-1].append(prefix_response_ids + continuation_ids)
                all_log_probs[-1].append(first_log_probs[prompt_idx][:pos] + continuation_log_probs)
        
        # Step 5: Combine all responses and prepare the final output
        combined_responses = []
        combined_log_probs = []
        
        # Flatten the response lists
        for prompt_idx in range(batch_size):
            # Add the first response
            combined_responses.append(first_responses[prompt_idx])
            combined_log_probs.append(first_log_probs[prompt_idx])
            
            # Add the additional responses
            for resp_batch in all_responses[1:]:
                if prompt_idx < len(resp_batch):
                    combined_responses.append(resp_batch[prompt_idx])
                    
            for log_prob_batch in all_log_probs[1:]:
                if prompt_idx < len(log_prob_batch):
                    combined_log_probs.append(log_prob_batch[prompt_idx])
        
        # Pad the responses and log probs
        response_tensor = pad_2d_list_to_length(combined_responses, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        log_probs_tensor = pad_2d_list_to_length(combined_log_probs, -1, max_length=self.config.response_length).to(idx.device)
        log_probs_tensor = log_probs_tensor.to(torch.float32)
        
        # Repeat the prompt tensors to match the number of responses
        expanded_idx = idx.repeat(n, 1)
        expanded_attention_mask = attention_mask.repeat(n, 1)
        expanded_position_ids = position_ids.repeat(n, 1) if position_ids.dim() == 2 else position_ids.repeat(n, 1, 1)
        
        # Create the final batch using the helper method
        batch = self._prepare_output_batch(
            prompts_tensor=expanded_idx,
            responses_tensor=response_tensor,
            log_probs_tensor=log_probs_tensor,
            eos_token_id=eos_token_id,
            position_ids_template=expanded_position_ids
        )
        
        # Update non_tensor_batch if needed
        if "tools_kwargs" in non_tensor_batch.keys():
            non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], n)
        
        # Free vllm cache engine if needed
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @GPUMemoryLogger(role="vllm rollout spmd (loop)", logger=logger)
    @torch.no_grad()
    def generate_sequences_loop(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            sample_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            sample_kwargs = {
                "top_k": prompts.meta_info.get("top_k", self.config.val_kwargs.top_k),
                "top_p": prompts.meta_info.get("top_p", self.config.val_kwargs.top_p),
                "temperature": prompts.meta_info.get("temperature", self.config.val_kwargs.temperature),
                "n": 1,
            }
        else:
            sample_kwargs = {}

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # Determine number of iterations n
        n_target = kwargs.get("n", getattr(self.sampling_params, "n", 1))
        if not do_sample or is_validate:
            n_target = 1

        # Run n iterations with n=1 per iteration
        responses_flat: List[List[int]] = [None] * (batch_size * n_target)
        log_probs_flat: List[List[float]] = [None] * (batch_size * n_target)

        step_kwargs = dict(sample_kwargs)
        step_kwargs["n"] = 1
        with self.update_sampling_params(**step_kwargs):
            for j in range(n_target):
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

                p = 0
                for output in outputs:
                    # since n == 1 here
                    sample = output.outputs[0]
                    response_ids = sample.token_ids
                    curr_log_prob: List[float] = []
                    for i, logprob in enumerate(sample.logprobs):
                        curr_log_prob.append(logprob[response_ids[i]].logprob)
                    # place in interleaved order [p0_s0, p0_s1, ..., p1_s0, ...]
                    responses_flat[p * n_target + j] = response_ids
                    log_probs_flat[p * n_target + j] = curr_log_prob
                    p += 1

        # Pad and convert to tensors
        response = pad_2d_list_to_length(responses_flat, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        rollout_log_probs = pad_2d_list_to_length(log_probs_flat, -1, max_length=self.config.response_length).to(idx.device)
        rollout_log_probs = rollout_log_probs.to(torch.float32)

        if n_target > 1 and do_sample:
            idx = _repeat_interleave(idx, n_target)
            attention_mask = _repeat_interleave(attention_mask, n_target)
            position_ids = _repeat_interleave(position_ids, n_target)
            batch_size = batch_size * n_target
            # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
            if "tools_kwargs" in non_tensor_batch.keys():
                non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], n_target)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "rollout_log_probs": rollout_log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        # sys.stdout.flush()
        # print(f"[DEBUG] input_ids: {batch['input_ids'].shape}")
        # print(f"[DEBUG] prompts: {batch['prompts'].shape}")
        # print(f"[DEBUG] responses: {batch['responses'].shape}")
        """
        [DEBUG] input_ids: torch.Size([256, 4096])
        [DEBUG] prompts: torch.Size([256, 2048])
        [DEBUG] responses: torch.Size([256, 2048])
        """   

        # Add simple tree metadata: r=0 is root; r>0 branches from r=0 at k=0
        try:
            bs_calc = int(batch_size // n_target) if n_target > 0 and batch_size % n_target == 0 else None
        except Exception:
            bs_calc = None
        if bs_calc is not None and bs_calc > 0 and n_target > 0:
            combined_meta = []
            for p in range(bs_calc):
                for r in range(n_target):
                    combined_meta.append({
                        "prompt_index": p,
                        "parent_traj": -1 if r == 0 else 0,
                        "branch_pos": 0,
                        "prefix_len": 0,
                    })
            try:
                per_sample_arr = np.array(combined_meta, dtype=object)
                n_arr = np.array([n_target] * len(combined_meta), dtype=object)
                non_tensor_batch["per_sample"] = per_sample_arr
                non_tensor_batch["n_per_prompt"] = n_arr
            except Exception:
                non_tensor_batch["per_sample"] = combined_meta
                non_tensor_batch["n_per_prompt"] = [n_target] * len(combined_meta)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    @GPUMemoryLogger(role="vllm rollout spmd (loop_entropy)", logger=logger)
    @torch.no_grad()
    def generate_sequences_loop_entropy(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            sample_kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            sample_kwargs = {
                "top_k": prompts.meta_info.get("top_k", self.config.val_kwargs.top_k),
                "top_p": prompts.meta_info.get("top_p", self.config.val_kwargs.top_p),
                "temperature": prompts.meta_info.get("temperature", self.config.val_kwargs.temperature),
                "n": 1,
            }
        else:
            sample_kwargs = {}

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # Determine number of iterations n
        n_target = kwargs.get("n", getattr(self.sampling_params, "n", 1))
        # print(f"[DEBUG loop] do_sample={do_sample} is_validate={is_validate} kwargs_n={kwargs.get('n', None)} sampling_n={getattr(self.sampling_params, 'n', None)} initial_n_target={n_target}")
        if not do_sample or is_validate:
            n_target = 1
            # print("[DEBUG loop] forcing n_target=1 due to do_sample/is_validate")

        # Per-prompt storage for iterative branching
        per_prompt_responses: List[List[List[int]]] = [[] for _ in range(batch_size)]
        per_prompt_log_probs: List[List[List[float]]] = [[] for _ in range(batch_size)]
        per_prompt_entropies: List[List[List[float]]] = [[] for _ in range(batch_size)]
        # Track per-sample branching metadata for advantage computation
        per_prompt_meta: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]

        # epsilon strategy: probability epsilon to branch from head, otherwise from max-entropy token
        try:
            import random
            eps_cfg = float(getattr(self.config, "entropy_branching_epsilon", 0.0))
            eps_meta = float(prompts.meta_info.get("entropy_branching_epsilon", eps_cfg))
            epsilon = max(0.0, min(1.0, eps_meta))
        except Exception:
            import random  # ensure available
            epsilon = 0.0

        # sys.stdout.flush()
        # print(f"[DEBUG] epsilon: {epsilon}")
        # Run n iterations; each iteration generates 1 new response per prompt
        use_unique_seed = bool(getattr(self.config, "iterative_sampling_unique_seed", False) or prompts.meta_info.get("iterative_sampling_unique_seed", False))
        for it in range(n_target):
            step_kwargs = dict(sample_kwargs)
            step_kwargs["n"] = 1
            if use_unique_seed:
                # Assign a different seed each iteration to reduce correlation across calls
                step_kwargs["seed"] = int(np.random.randint(0, 2**31 - 1))
            # with self.update_sampling_params(**step_kwargs):
            if it == 0:
                # First iteration: use original prompts
                iter_inputs = vllm_inputs
                prompt_indices = list(range(batch_size))
                branch_prefix_ids = [[] for _ in range(batch_size)]
                branch_prefix_logps = [[] for _ in range(batch_size)]
                branch_prefix_ents = [[] for _ in range(batch_size)]
            else:
                # Build branching prompts at highest-entropy token or head (epsilon)
                iter_inputs = []
                prompt_indices = []
                branch_prefix_ids = []
                branch_prefix_logps = []
                branch_prefix_ents = []
                # Helper: filter out-of-vocab ids (>= vocab_size) if vocab_size is known
                def _filter_oov_ids(_ids: List[int]) -> List[int]:
                    if self.vocab_size is None:
                        return _ids
                    vsz = int(self.vocab_size)
                    new_list = []
                    for t in _ids:
                        if int(t) >= 0 and int(t) < vsz:
                            new_list.append(int(t))
                        else:
                            # sys.stdout.flush()
                            # # new_list.append(self.pad_token_id)
                            # print(f"[DEBUG] oov id: {t}")
                    return new_list

                # Record selection info per constructed input
                iter_parent_traj: List[int] = []
                iter_branch_pos: List[int] = []
                iter_prefix_len: List[int] = []
                for p in range(batch_size):
                    # Decide strategy per prompt
                    use_head = epsilon > 0.0 and random.random() < epsilon
                    # sys.stdout.flush()
                    # print(f"[DEBUG] use_head: {use_head}")
                    if use_head or len(per_prompt_entropies[p]) == 0:
                        prefix_ids = []
                        prefix_logps = []
                        prefix_ents = []
                        parent_traj = -1
                        branch_pos = 0
                    else:
                        # Find (trajectory, position) with maximum entropy
                        best_ent = float("-inf")
                        best_traj = 0
                        best_pos = 0
                        for t_idx, ent_list in enumerate(per_prompt_entropies[p]):
                            for pos_i, ent_val in enumerate(ent_list):
                                if ent_val > best_ent:
                                    best_ent = ent_val
                                    best_traj = t_idx
                                    best_pos = pos_i
                        # Prefix excludes the high-entropy token itself; resample from that position
                        prefix_ids = per_prompt_responses[p][best_traj][:best_pos]
                        prefix_logps = per_prompt_log_probs[p][best_traj][:best_pos]
                        prefix_ents = per_prompt_entropies[p][best_traj][:best_pos]
                        parent_traj = int(best_traj)
                        branch_pos = int(best_pos)

                    new_prompt_ids = vllm_inputs[p]["prompt_token_ids"] + prefix_ids
                    input_rec = {"prompt_token_ids": new_prompt_ids}
                    if "multi_modal_data" in vllm_inputs[p]:
                        input_rec["multi_modal_data"] = vllm_inputs[p]["multi_modal_data"]
                    if isinstance(input_rec["prompt_token_ids"], np.ndarray):
                        input_rec["prompt_token_ids"] = input_rec["prompt_token_ids"].tolist()
                    # Filter OOV ids (>= vocab_size); keep ordering of remaining ids
                    input_rec["prompt_token_ids"] = _filter_oov_ids(input_rec["prompt_token_ids"])  # do not replace

                    iter_inputs.append(input_rec)
                    prompt_indices.append(p)
                    branch_prefix_ids.append(prefix_ids)
                    branch_prefix_logps.append(prefix_logps)
                    branch_prefix_ents.append(prefix_ents)
                    # Save selection info aligned with iter_inputs
                    iter_parent_traj.append(parent_traj)
                    iter_branch_pos.append(branch_pos)
                    iter_prefix_len.append(len(prefix_ids))
            with self.update_sampling_params(**step_kwargs):
                # Generate one response per constructed input
                outputs = self.inference_engine.generate(
                    prompts=iter_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )
            
            # Collect results back to per-prompt stores
            for i, output in enumerate(outputs):
                p = prompt_indices[i]
                sample = output.outputs[0]
                cont_ids = sample.token_ids

                # Respect total response_length budget
                prefix_len = len(branch_prefix_ids[i])
                remaining = max(self.config.response_length - prefix_len, 0)
                allowed_len = min(len(cont_ids), remaining)
                cont_ids = cont_ids[:allowed_len]

                # Collect logprobs and entropies for continuation
                cont_logps = []
                cont_ents = []
                branch_metric = self.config.get("branch_metric", "entropy")
                if branch_metric == "entropy":
                    for t, logprob_dict in enumerate(sample.logprobs[:allowed_len]):
                        token_id = cont_ids[t]
                        cont_logps.append(logprob_dict[token_id].logprob)
                        probs = [np.exp(lp.logprob) for lp in logprob_dict.values()]
                        logps = [lp.logprob for lp in logprob_dict.values()]
                        # print(f"[DEBUG] probs: {len(probs)}")
                        # print(f"[DEBUG] logps: {len(logps)}")
                        # sys.stdout.flush()
                        cont_ents.append(float(-sum(pv * lpv for pv, lpv in zip(probs, logps))))
                elif branch_metric == "cosine":
                    # print(f"[DEBUG] cosine")
                    # sys.stdout.flush()
                    top_k = self.config.get("cosine_top_k", 20)
                    if not hasattr(self, '_cached_vocab_embeddings'):
                        try:
                            self._cached_vocab_embeddings = get_vocab_embeddings_from_vllm(self.inference_engine, self.tokenizer).to(idx.device)
                        except Exception as e:
                            raise RuntimeError(f"Cosine mode requires vocab embeddings but failed to load: {e}")
                    for t, logprob_dict in enumerate(sample.logprobs[:allowed_len]):
                        token_id = cont_ids[t]
                        cont_logps.append(logprob_dict[token_id].logprob)

                        sorted_logprobs = sorted(logprob_dict.items(), key=lambda x: x[1].logprob, reverse=True)
                        top_k_logprobs = sorted_logprobs[:top_k]
                        if len(top_k_logprobs) < 2:
                            raise RuntimeError(
                                f"Cosine entropy requires at least 2 candidates, got {len(top_k_logprobs)} at position {t}."
                            )
                        top_k_token_ids = [int(item[0]) for item in top_k_logprobs]
                        top_k_probs = [np.exp(item[1].logprob) for item in top_k_logprobs]
                        try:
                            # 获取候选token的embeddings
                            candidate_embeddings = self._cached_vocab_embeddings[top_k_token_ids]  # (n_candidates, hidden_size)
                            # print(f"[DEBUG] candidate_embeddings: {candidate_embeddings.shape};device:{candidate_embeddings.device};idx.device:{idx.device}")
                            # sys.stdout.flush()

                            # 计算cosine相似度矩阵
                            normalized_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)

                            E = normalized_embeddings
                            prob_vec = torch.tensor(top_k_probs, device=E.device, dtype=E.dtype)
                            y_vec = E.t().matmul(prob_vec)               # (H,)
                            cosine_weighted_entropy = (y_vec * y_vec).sum() - (prob_vec * prob_vec).sum()
                            cont_ents.append(float(-cosine_weighted_entropy))
                        except Exception as e:
                            raise RuntimeError(
                                f"Cosine entropy calculation failed at position {t}: {e}"
                            )
                elif branch_metric == "cosine-entropy":
                    # print(f"[DEBUG] cosine-entropy")
                    # sys.stdout.flush()
                    top_k = self.config.get("cosine_top_k", 20)
                    if not hasattr(self, '_cached_vocab_embeddings'):
                        try:
                            self._cached_vocab_embeddings = get_vocab_embeddings_from_vllm(self.inference_engine, self.tokenizer).to(idx.device)
                        except Exception as e:
                            raise RuntimeError(f"Cosine mode requires vocab embeddings but failed to load: {e}")
                    for t, logprob_dict in enumerate(sample.logprobs[:allowed_len]):
                        token_id = cont_ids[t]
                        cont_logps.append(logprob_dict[token_id].logprob)

                        # 1. 计算标准熵 H_t = -sum(p * log(p))，转为 GPU tensor 加速
                        emb_device = getattr(self._cached_vocab_embeddings, "device", idx.device)
                        logprob_items = list(logprob_dict.items())
                        token_ids_tensor = torch.tensor(
                            [int(item[0]) for item in logprob_items],
                            device=emb_device,
                            dtype=torch.long,
                        )
                        logprob_tensor = torch.tensor(
                            [item[1].logprob for item in logprob_items],
                            device=emb_device,
                            dtype=torch.float32,
                        )
                        probs_tensor = torch.exp(logprob_tensor)
                        standard_entropy = float(
                            -(probs_tensor * logprob_tensor).sum().item()
                        )
                        # 2. 选出 top-k 个概率最高的词
                        actual_top_k = min(top_k, logprob_tensor.numel())
                        if actual_top_k < 2:
                            raise RuntimeError(
                                f"Cosine entropy requires at least 2 candidates, got {actual_top_k} at position {t}."
                            )
                        topk_vals, topk_idx = torch.topk(logprob_tensor, k=actual_top_k)
                        topk_token_ids = token_ids_tensor.index_select(0, topk_idx)
                        topk_probs = torch.exp(topk_vals)
                        try:
                            # 获取候选token的embeddings
                            candidate_embeddings = self._cached_vocab_embeddings.index_select(0, topk_token_ids)  # (n_candidates, hidden_size)

                            # 计算cosine相似度矩阵
                            normalized_embeddings = torch.nn.functional.normalize(candidate_embeddings, p=2, dim=1)
                            # similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())  # (n_candidates, n_candidates)

                            # # 4. 计算相似度加权熵 F_t = sum(p_i * p_j * c_ij)
                            # cosine_weighted_entropy = 0.0
                            # for i in range(len(topk_probs)):
                            #     for j in range(len(topk_probs)):
                            #         if i != j:  # 排除自相似度
                            #             cosine_weighted_entropy += topk_probs[i] * topk_probs[j] * similarity_matrix[i, j].item()

                            E = normalized_embeddings
                            prob_vec = topk_probs.to(dtype=E.dtype).unsqueeze(1)
                            y_vec = E.t().matmul(prob_vec)               # (H,)
                            flat_probs = prob_vec.squeeze(1)
                            cosine_weighted_entropy = (y_vec * y_vec).sum() - (flat_probs * flat_probs).sum()
                            cont_ents.append(float((-cosine_weighted_entropy) * standard_entropy))
                        except Exception as e:
                            raise RuntimeError(
                                f"Cosine entropy calculation failed at position {t}: {e}"
                            )
                        

                new_resp = branch_prefix_ids[i] + cont_ids
                new_logps = branch_prefix_logps[i] + cont_logps
                new_ents = branch_prefix_ents[i] + cont_ents
                max_resp_len = int(self.config.response_length)
                if len(new_resp) > max_resp_len:
                    new_resp = new_resp[:max_resp_len]
                    new_logps = new_logps[:max_resp_len]
                    new_ents = new_ents[:max_resp_len]

                per_prompt_responses[p].append(new_resp)
                per_prompt_log_probs[p].append(new_logps)
                per_prompt_entropies[p].append(new_ents)
                # Append per-sample metadata for this newly created trajectory
                if it == 0:
                    per_prompt_meta[p].append({
                        "parent_traj": -1,
                        "branch_pos": -1,
                        "prefix_len": 0,
                    })
                else:
                    per_prompt_meta[p].append({
                        "parent_traj": int(iter_parent_traj[i]),
                        "branch_pos": int(iter_branch_pos[i]),
                        "prefix_len": int(iter_prefix_len[i]),
                    })

        # Flatten responses in interleaved order matching idx.repeat(n, 1)
        # print(f"[DEBUG loop] per_prompt_responses_count={[len(lst) for lst in per_prompt_responses]} n_target={n_target} batch_size={batch_size}")
        combined_responses: List[List[int]] = []
        combined_log_probs: List[List[float]] = []
        combined_meta: List[Dict[str, Any]] = []
        for p in range(batch_size):
            for r in range(n_target):
                combined_responses.append(per_prompt_responses[p][r])
                combined_log_probs.append(per_prompt_log_probs[p][r])
                # per-sample tree metadata aligned with p-major flattening
                combined_meta.append({
                    "prompt_index": p,
                    **(per_prompt_meta[p][r] if r < len(per_prompt_meta[p]) else {"parent_traj": -1, "branch_pos": -1, "prefix_len": 0}),
                })
        # print(f"[DEBUG loop] combined_responses_len={len(combined_responses)} expected={batch_size * n_target}")

        # Pad and build tensors
        response = pad_2d_list_to_length(combined_responses, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
        rollout_log_probs = pad_2d_list_to_length(combined_log_probs, -1, max_length=self.config.response_length).to(idx.device)
        rollout_log_probs = rollout_log_probs.to(torch.float32)
        # print(f"[DEBUG loop] response_tensor_shape={tuple(response.shape)}")

        if n_target > 1 and do_sample:
            # print(f"[DEBUG loop] before_repeat idx={tuple(idx.shape)} n_target={n_target} do_sample={do_sample}")
            idx = _repeat_interleave(idx, n_target)
            attention_mask = _repeat_interleave(attention_mask, n_target)
            position_ids = _repeat_interleave(position_ids, n_target)
            batch_size = batch_size * n_target
            if "tools_kwargs" in non_tensor_batch.keys():
                non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], n_target)
            # print(f"[DEBUG loop] after_repeat idx={tuple(idx.shape)} attention_mask={tuple(attention_mask.shape)} position_ids={tuple(position_ids.shape)}")

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "rollout_log_probs": rollout_log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        # sys.stdout.flush()
        # print(f"[DEBUG] input_ids: {batch['input_ids'].shape}")
        # print(f"[DEBUG] prompts: {batch['prompts'].shape}")
        # print(f"[DEBUG] responses: {batch['responses'].shape}")
        """
        [DEBUG] input_ids: torch.Size([128, 4096])
        [DEBUG] prompts: torch.Size([128, 2048])
        [DEBUG] responses: torch.Size([128, 2048])
        """

        # Attach tree structure metadata (top-level keys, avoid nested dict to ease union/concat)
        try:
            # Validate branching metadata to prevent cycles/invalid parents
            try:
                do_validate = True
                try:
                    # allow disabling via config/meta
                    do_validate = bool(getattr(self.config, "validate_branch_metadata", True))
                    do_validate = bool(prompts.meta_info.get("validate_branch_metadata", do_validate))
                except Exception:
                    pass
                if do_validate:
                    bs_check = None
                    try:
                        bs_check = int(batch_size // n_target) if n_target > 0 and batch_size % n_target == 0 else None
                    except Exception:
                        bs_check = None
                    if bs_check is not None and len(combined_meta) == batch_size:
                        # p-major order: index = p * n + r
                        for p in range(bs_check):
                            parents = [None] * n_target
                            for r in range(n_target):
                                idx_meta = p * n_target + r
                                meta = combined_meta[idx_meta]
                                pr = int(meta.get("parent_traj", -1))
                                parents[r] = pr
                                # prompt_index consistency
                                if int(meta.get("prompt_index", p)) != p:
                                    # sys.stdout.flush()
                                    raise ValueError(f"[branch_meta][err] prompt_index mismatch p={p} r={r} meta_p={meta.get('prompt_index', None)}")
                                # invalid parent range or self-loop
                                if not (pr == -1 or (0 <= pr < n_target and pr != r)):
                                    # sys.stdout.flush()
                                    raise ValueError(f"[branch_meta][err] invalid parent p={p} r={r} parent={pr} n={n_target}")
                            # cycle check
                            for r0 in range(n_target):
                                seen = set()
                                cur = r0
                                steps = 0
                                while cur >= 0 and steps <= n_target + 2:
                                    if cur in seen:
                                        # sys.stdout.flush()
                                        raise ValueError(f"[branch_meta][err] cycle detected p={p} r0={r0} at cur={cur} parents={parents}")
                                        break
                                    seen.add(cur)
                                    steps += 1
                                    cur = parents[cur] if 0 <= cur < n_target else -1
                    else:
                        sys.stdout.flush()
                        print(f"[branch_meta][info] skip meta validation: len={len(combined_meta)} batch_size={batch_size} n={n_target}", flush=True)
            except Exception as _e:
                sys.stdout.flush()
                print(f"[branch_meta][warn] validation error: {_e}", flush=True)
            non_tensor_batch["per_sample"] = np.array(combined_meta, dtype=object)
            non_tensor_batch["n_per_prompt"] = np.array(n_target, dtype=object)
        except Exception:
            pass

        # Normalize non-tensor fields to numpy arrays (dtype=object) with correct batch length
        import numpy as _np
        bs = batch.batch_size[0]
        normalized_non_tensor = {}
        for _k, _v in non_tensor_batch.items():
            if isinstance(_v, _np.ndarray):
                # Broadcast/tile numpy arrays to match batch size
                if _v.ndim == 0:
                    normalized_non_tensor[_k] = _np.array([_v.item()] * bs, dtype=object)
                elif _v.shape[0] == bs:
                    normalized_non_tensor[_k] = _v
                elif _v.size == 1:
                    normalized_non_tensor[_k] = _np.array([_v.reshape(-1)[0]] * bs, dtype=object)
                elif _v.size > 0 and bs % _v.size == 0:
                    repeats = bs // _v.size
                    normalized_non_tensor[_k] = _np.repeat(_v, repeats, axis=0)
                else:
                    normalized_non_tensor[_k] = _np.array(list(_v), dtype=object)
            elif isinstance(_v, list):
                # Convert lists and align to batch size
                if len(_v) == bs:
                    normalized_non_tensor[_k] = _np.array(_v, dtype=object)
                elif len(_v) == 1:
                    normalized_non_tensor[_k] = _np.array([_v[0]] * bs, dtype=object)
                elif len(_v) > 0 and bs % len(_v) == 0:
                    normalized_non_tensor[_k] = _np.array(_v * (bs // len(_v)), dtype=object)
                else:
                    normalized_non_tensor[_k] = _np.array(_v, dtype=object)
            else:
                # Scalar or other types -> broadcast
                normalized_non_tensor[_k] = _np.array([_v] * bs, dtype=object)

        return DataProto(batch=batch, non_tensor_batch=normalized_non_tensor)

    def _prepare_output_batch(self, prompts_tensor: torch.Tensor, responses_tensor: torch.Tensor,
                              log_probs_tensor: torch.Tensor, eos_token_id: int,
                              position_ids_template: torch.Tensor = None) -> TensorDict:
        """Helper to construct a standard output TensorDict from generated sequences."""
        batch_size = prompts_tensor.shape[0]
        prompt_len = prompts_tensor.shape[1]
        resp_len = responses_tensor.shape[1]

        input_ids = torch.cat([prompts_tensor, responses_tensor], dim=1)
        
        prompt_mask = (prompts_tensor != self.pad_token_id).to(torch.long)
        response_mask = get_response_mask(response_id=responses_tensor, eos_token=eos_token_id, dtype=torch.long)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        
        if position_ids_template is not None and position_ids_template.shape[0] == batch_size:
            response_pos_delta = torch.arange(1, resp_len + 1, device=prompts_tensor.device).unsqueeze(0)
            if position_ids_template.dim() == 3:
                last_pos = position_ids_template[:, :, -1:]
                response_pos_delta = response_pos_delta.view(1, 1, -1)
            else:
                last_pos = position_ids_template[:, -1:]
            
            response_position_ids = last_pos + response_pos_delta
            position_ids = torch.cat([position_ids_template, response_position_ids], dim=-1)
        else:
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

    # def _linear_adaptive_temperature(self,temperature_low,temperature_high,current_step,total_steps):
    #     return temperature_low + (temperature_high - temperature_low) * (current_step / total_steps)

class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
