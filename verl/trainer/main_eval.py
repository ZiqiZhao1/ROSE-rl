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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)

    # Aggregate generated length statistics per data source
    data_source_gen_len = defaultdict(list)
    for i in range(total):
        ds = data_sources[i]
        resp_list = responses[i]
        # Each row may contain a list of responses; collect lengths for all
        if isinstance(resp_list, (list, tuple, np.ndarray)):
            iterable = resp_list
        else:
            iterable = [resp_list]
        for r in iterable:
            if isinstance(r, (list, tuple, np.ndarray)):
                l = len(r)  # token-id list length
            elif isinstance(r, str):
                l = len(r)  # character length as a fallback when tokens are not available
            else:
                l = len(str(r))
            data_source_gen_len[ds].append(l)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)
    for data_source, lens in data_source_gen_len.items():
        if len(lens) > 0:
            metric_dict[f"gen_length/mean/{data_source}"] = float(np.mean(lens))
            metric_dict[f"gen_length/max/{data_source}"] = float(np.max(lens))
            metric_dict[f"gen_length/min/{data_source}"] = float(np.min(lens))

    # Overall mean across all data_sources
    all_lens = [l for ls in data_source_gen_len.values() for l in ls]
    if len(all_lens) > 0:
        metric_dict["gen_length/mean/all"] = float(np.mean(all_lens))

    print(metric_dict)


if __name__ == "__main__":
    main()
