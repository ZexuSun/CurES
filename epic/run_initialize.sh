model_name_or_path=$1
iter_idx=$2
num_rollout_min=$3
num_rollout_max=$4
correct_threshold=$5
entropy_param=$6
train_batch_size=$7
output_dir=$8

seed=42
max_response_length=3072
data_path=ScaleML-RLHF/numina_math_${iter_idx}
data_split="train"
data_start=0
data_end=999999999
GPUS=(0 1 2 3 4 5 6 7)
world_size=${#GPUS[@]}
system_prompt="qwen25-math-cot"

mkdir -p $output_dir
export MASTER_ADDR=localhost
export MASTER_PORT=50030
export HF_ENDPOINT=https://hf-mirror.com

echo "world_size: $world_size"

for rank in $(seq 0 $((world_size - 1))); do
    export VLLM_PORT=$((MASTER_PORT + rank + 1))
    CUDA_VISIBLE_DEVICES=${GPUS[$rank]} python ./epic/stage_0_init_difficulty_sample_size.py \
        --seed $seed \
        --max_length $max_response_length \
        --model_name_or_path $model_name_or_path \
        --data_path $data_path \
        --data_split $data_split \
        --start $data_start \
        --end $data_end \
        --num_rollout_min $num_rollout_min \
        --num_rollout_max $num_rollout_max \
        --local_rank $rank \
        --world_size $world_size \
        --iter_idx $iter_idx \
        --correct_threshold $correct_threshold \
        --system_prompt $system_prompt \
        --entropy_param $entropy_param \
        --output_dir $output_dir \
        --batch_size $train_batch_size &
done

wait

python ./epic/stage_1_merge_init.py \
    --output_dir $output_dir

echo "🎉 Iter ${iter_idx} successfully initialized."