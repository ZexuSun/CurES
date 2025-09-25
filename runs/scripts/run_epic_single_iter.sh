set -x

export WANDB_API_KEY="YOUR WANDB API KEY"

project_name="EPIC"
algorithm=grpo
model=Qwen2.5-Math-1.5B
policy_loss=plusplus
rollout_n=4
iter=1
total_epochs=1

HOME=/root/paddlejob/workspace/env_run/output/YOUR_DATA_ROOT
aime24=$HOME/speed-rl/data/AIME2024-dup16-instruct/train.parquet
aime25=$HOME/speed-rl/data/AIME2025-dup16-instruct/train.parquet
math500=$HOME/speed-rl/data/Math500-instruct/test.parquet
amc23=$HOME/speed-rl/data/AMC23-instruct/train.parquet
numina=$HOME/GVM/data/numina_math_${iter}/train.parquet

enable_filter_groups=False
filter_groups_metric=seq_final_reward
max_num_gen_batches=10

INTER_DIR=/root/paddlejob/workspace/env_run/output/EPIC/epic/intermidate/EPIC/debug/1
start_model=/root/paddlejob/workspace/env_run/output/models/${model}
sample_sizes_data=$INTER_DIR/sample_size_rank_all.json
difficulty_data=$INTER_DIR/difficulty_rank_all.json
counter_data=$INTER_DIR/counter_rank_all.json

train_files="['$numina']"
test_files="['$aime24', '$aime25', '$math500', '$amc23']"

mkdir -p logs/${project_name}

GPUS=(0 1 2 3 4 5 6 7)
my_world_size=${#GPUS[@]}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m epic.main_epic \
    algorithm.adv_estimator=$algorithm \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    +data.use_epic=True \
    +data.epic_difficulty_data="$difficulty_data" \
    +data.epic_counter_data="$counter_data" \
    +data.epic_entropy_param=1.0 \
    +data.epic_replacement=True \
    +data.epic_acceptance_threshold=0.5 \
    data.shuffle=False \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.path="$start_model" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss=$policy_loss \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.use_em=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$rollout_n \
    +actor_rollout_ref.rollout.sample_sizes_data="$sample_sizes_data" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=$my_world_size \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=5 \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs 2>&1 | tee logs/${project_name}/${experiment_name}.log