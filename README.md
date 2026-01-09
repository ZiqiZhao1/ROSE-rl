# Reinforced Efficient Reasoning via Semantically Diverse Exploration

This repository contains the official implementation of the paper [**Reinforced Efficient Reasoning via Semantically Diverse Exploration**](https://arxiv.org/abs/2601.05053).
This project is built on top of [VERL](https://github.com/volcengine/verl), a framework for reinforcement learning on large language models.

## Pre-process datasets

```python
python examples/data_preprocess/preprocess_all_math.py
```

## Quick Start

To run an example training script, execute:

```bash
./example.sh
```

## Requirements

See `requirements.txt` for Python dependencies.
**Citation**

If you find our work useful, please cite our paper:

```
@article{zhao2026reer-sde,
  title={Reinforced Efficient Reasoning via Semantically Diverse Exploration},
  author={Ziqi Zhao and Zhaochun Ren and Jiahong Zou and Liu Yang and Zhiwei Xu and Xuri Ge and Zhumin Chen and Xinyu Ma and Daiting Shi and Shuaiqiang Wang and Dawei Yin and Xin Xin},
  journal={arXiv preprint arXiv:2601.05053},
  year={2026},
  url={https://arxiv.org/abs/2601.05053}
}