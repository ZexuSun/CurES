model="/root/paddlejob/workspace/env_run/output/models/Qwen2.5-Math-1.5B"
iter_idx=1
num_rollout_min=4
num_rollout_max=8
correct_threshold=0.5
entropy_param=1.0
project_name="EPIC"
experiment_name="debug"
batch_size=1024

bash run_initialize.sh $model $iter_idx $num_rollout_min $num_rollout_max $correct_threshold $entropy_param $project_name $experiment_name $batch_size
