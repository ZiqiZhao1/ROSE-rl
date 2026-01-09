"""
process math dataset in Qwen manner
set CUSTOM_REWARD_FUNCTION_PATH="verl/utils/reward_score/math.py"
Test_dataset_list = ["AIME2024", "AIME2025", "AMC", "MATH500", "MinervaMath"]
    MinervaMath:
        data_source:    "math-ai/minervamath"
        question_key:   "question"
        answer_key:     "answer"
        need_extract_solution:False
        split:          "test"
    AIME2024:
        data_source: "Maxwell-Jia/AIME_2024"
        question_key: "Problem"
        answer_key: "Answer"
        need_extract_solution: False
        split:          "train"
    AIME2025:
        data_source:    "MathArena/aime_2025"
        question_key:   "problem"
        answer_key:     "answer"
        need_extract_solution: False
        split:          "train"
    AMC:
        data_source:    "math-ai/amc23"
        question_key:   "question"
        answer_key:     "answer"
        need_extract_solution:False
        split:          "test"
    MATH500:
        data_source:    "HuggingFaceH4/MATH-500"
        question_key:   "problem"
        answer_key:     "answer"
        need_extract_solution:False
        split:          "test"

Train_dataset_list = ["MATH"]
    MATH:DigitalLearningGmbH/MATH-lighteval
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--test_dataset_list", default=["AIME2024", "AIME2025", "AMC", "MATH500"])
    parser.add_argument("--train_dataset_list", default=["MATH"])

    args = parser.parse_args()
    instruction="Please reason step by step, and  put your final answer within \\boxed{}"


    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    # process train dataset
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]


    # add a row to each data item that represents a unique id
    def make_map_fn(source,split,question_key,answer_key,need_extract_solution=False):
        def process_fn(example, idx):
            question = example.pop(question_key)

            solution = example.pop(answer_key)
            if need_extract_solution:
                solution = extract_solution(solution)
            solution = str(solution)
            data = {
                "data_source": source,
                "prompt": [{"role": "system", "content": instruction},
                           {"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn(data_source,"train","problem","solution",need_extract_solution=True), with_indices=True)
    train_dataset_save_path = os.path.join(args.local_dir, args.train_dataset_list[0])
    train_dataset.to_parquet(os.path.join(train_dataset_save_path, "train.parquet"))

    # process test datasets
    test_datasets_config = {
        "AIME2024": {
            "data_source": "Maxwell-Jia/AIME_2024",
            "question_key": "Problem",
            "answer_key": "Answer",
            "need_extract_solution": False,
            "split": "train",
        },
        "AIME2025": {
            "data_source": "MathArena/aime_2025",
            "question_key": "problem",
            "answer_key": "answer",
            "need_extract_solution": False,
            "split": "train",
        },
        "AMC": {
            "data_source": "math-ai/amc23",
            "question_key": "question",
            "answer_key": "answer",
            "need_extract_solution": False,
            "split": "test",
        },
        "MATH500": {
            "data_source": "HuggingFaceH4/MATH-500",
            "question_key": "problem",
            "answer_key": "answer",
            "need_extract_solution": False,
            "split": "test",
        },
        "MinervaMath": {
            "data_source": "math-ai/minervamath",
            "question_key": "question",
            "answer_key": "answer",
            "need_extract_solution": False,
            "split": "test",
        },
    }

    for test_name in args.test_dataset_list:
        config = test_datasets_config[test_name]
        data_source = config["data_source"]
        question_key = config["question_key"]
        answer_key = config["answer_key"]
        need_extract_solution = config["need_extract_solution"]
        split = config["split"]
        print(f"Loading the {data_source} dataset from huggingface for {test_name}...", flush=True)
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)
        test_dataset = dataset[split]
        test_dataset = test_dataset.map(function=make_map_fn(source=test_name,split="test", question_key=question_key, answer_key=answer_key, need_extract_solution=need_extract_solution), with_indices=True)
        test_dataset_save_path = os.path.join(args.local_dir, test_name)
        os.makedirs(test_dataset_save_path, exist_ok=True)
        test_dataset.to_parquet(os.path.join(test_dataset_save_path, "test.parquet"))
    
    

