echo "=====Start====="
# use all gpus
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
MODEL_NAME="Llama-3.2-3B-Instruct"   # Qwen2.5-Math-1.5 Qwen3-4B-Base, Qwen3-4B, Qwen3-8B, Qwen2.5-0.5B-Instruct
FILTER_GROUPS_ENABLE=True
# group filter defaults
FILTER_GROUPS_METRIC="seq_final_reward"   # or seq_reward
MAX_NUM_GEN_BATCHES=8
ENTROPY_BRANCHING_EPSILON=0.5
NORM_ADV_BY_STD_IN_GRPO=True
LENGTH_PENALTY=0
ALGORITHM="grpo-tree-lengthPenalty"
LOSS_AGG_MODE="seq-mean-token-sum-norm"
BRANCH_METRIC="cosine-entropy"
ADV_ESTIMATOR="grpo_iterative_branching"
USE_ENTROPY_BRANCHING=True
ENTROPY_BRANCHING_MODE="iterative"
LOSS_TYPE="ppo"

TRAIN_FILE="<path-to-datasets>/MATH/train.parquet"
TEST_FILE=(
    "<path-to-datasets>/AIME2024/test.parquet"
    "<path-to-datasets>/AIME2025/test.parquet"
    "<path-to-datasets>/AMC/test.parquet"
    "<path-to-datasets>/MATH500/test.parquet"
)
VAL_FILES=$(IFS=, ; echo "${TEST_FILE[*]}")

MODEL_PATH="<path-to-model>/${MODEL_NAME}"
EXP_NAME="${MODEL_NAME}@${ALGORITHM}-EP${ENTROPY_BRANCHING_EPSILON}-${BRANCH_METRIC}-LP${LENGTH_PENALTY}"
FILE_NAME="${MODEL_NAME}@${ALGORITHM}_$(date +%Y%m%d_%H%M%S)"
PROJECT_NAME="RLVR-Sample"
TRAIN_BATCH_SIZE=512
TEST_FREQ=2
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
ROLLOUT_N_LIST=8
CUSTOM_REWARD_FUNCTION_PATH="<path-to-repo>/verl/utils/reward_score/math.py"
TOTAL_EPOCHS=10
EVAL_ROLLOUT_N_RANDOM=8
EVAL_TOP_P_RANDOM=0.95
EVAL_TEMPERATURE_RANDOM=0.6

# Define directories for saving rollouts
ROLLOUT_DATA_DIR="./rollouts/${FILE_NAME}/train"
VALIDATION_DATA_DIR="./rollouts/${FILE_NAME}/eval"
mkdir -p ${ROLLOUT_DATA_DIR}
mkdir -p ${VALIDATION_DATA_DIR}
echo "Saving rollouts to: ${ROLLOUT_DATA_DIR}"
echo "Saving validation rollouts to: ${VALIDATION_DATA_DIR}"

echo "Using GPUs: $DEVICES"
echo "Start exp: $EXP_NAME"

CUDA_VISIBLE_DEVICES=$DEVICES PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=${ADV_ESTIMATOR} \
  algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO} \
  algorithm.length_penalty=${LENGTH_PENALTY} \
  actor_rollout_ref.rollout.use_entropy_branching=${USE_ENTROPY_BRANCHING} \
  actor_rollout_ref.rollout.branch_metric=${BRANCH_METRIC} \
  actor_rollout_ref.rollout.entropy_branching_mode=${ENTROPY_BRANCHING_MODE} \
  actor_rollout_ref.actor.loss_type=${LOSS_TYPE} \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="[${VAL_FILES}]" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.entropy_branching_epsilon=${ENTROPY_BRANCHING_EPSILON} \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N_LIST}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger=['console','swanlab'] \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=${NUM_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=5 \
  trainer.test_freq=${TEST_FREQ} \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.log_val_generations=30 \
  algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE} \
  algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC} \
  algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES} \
  custom_reward_function.path="${CUSTOM_REWARD_FUNCTION_PATH}" \
  actor_rollout_ref.rollout.val_kwargs.n=${EVAL_ROLLOUT_N_RANDOM} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${EVAL_TEMPERATURE_RANDOM} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${EVAL_TOP_P_RANDOM} \
  2>&1 | tee ${FILE_NAME}.log


sleep 30
echo "=====End====="