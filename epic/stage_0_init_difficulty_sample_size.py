from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
import torch
import json
import os
import torch.nn as nn
from torch.distributions.beta import Beta
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment
)
from tqdm import tqdm
import utils
import numpy as np
import time
import gc
import contextlib
import ray


@dataclass
class ScriptArguments:
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed"})
    max_length: Optional[int] = field(default=3072, metadata={"help": "Max length of newly generated tokens"})
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-Math-7B", metadata={"help": "Model name or path"})
    data_path: Optional[str] = field(default="ScaleML-RLHF/numina_math_all", metadata={"help": "Path to the dataset"})
    data_split: Optional[str] = field(default="train", metadata={"help": "Split of the dataset"})
    start: Optional[int] = field(default=0, metadata={"help": "Start index"})
    end: Optional[int] = field(default=100000, metadata={"help": "End index"})
    num_rollout_min: Optional[int] = field(default=4, metadata={"help": "Minimum number of rollouts per prompt"})
    num_rollout_max: Optional[int] = field(default=16, metadata={"help": "Maximum number of rollouts per prompt"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "Local rank"})
    world_size: Optional[int] = field(default=1, metadata={"help": "World size"})
    iter_idx: Optional[int] = field(default=0, metadata={"help": "Iteration index"})
    correct_threshold: Optional[float] = field(default=0.5, metadata={"help": "Correct threshold"})
    system_prompt: Optional[str] = field(default="qwen25-math-cot", metadata={"help": "System prompt"})
    entropy_param: Optional[float] = field(default=1.0, metadata={"help": "Entropy parameter"})
    output_dir: Optional[str] = field(default="outputs", metadata={"help": "Output directory"})
    act_params: Optional[str] = field(default="embed_tokens", metadata={"help": "Act parameters"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "Batch size"})


class EPICInitializer:
    def __init__(self, script_args, dataset, tokenizer) -> None:
        self.args = script_args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.counter = [{"Accepted": 0, "All": 0} for _ in range(len(dataset))]
        self.difficulties = np.ones(len(dataset), dtype=np.float32)
        self.cot_conductor = f" Let's think step by step and output the final answer within \\boxed{{}}"
        self.sample_sizes = [self.args.num_rollout_min for _ in range(len(dataset))]

    def collect_rollouts(self, sample_indices, llm):
        if len(sample_indices) < len(self.dataset):
            batch = [self.dataset[index] for index in sample_indices]
        else:
            batch = self.dataset

        # Gather all prompts
        prompts = []
        for idx, item in enumerate(batch):
            item_idx = sample_indices[idx]
            sampling_params = SamplingParams(max_tokens=self.args.max_length, temperature=1.0, n=self.sample_sizes[item_idx])
            if self.args.system_prompt:
                conv = [
                    {"role": "system", "content": utils.SYSTEM_PROMPTS[self.args.system_prompt]},
                    {"role": "user", "content": item["problem"] + self.cot_conductor},
                ]
            else:
                conv = [{"role": "user", "content": item["problem"] + self.cot_conductor}]
            conv_chat = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            prompts.append(conv_chat)

        # Gather all rollouts
        outputs = llm.generate(prompts, sampling_params)
        rollouts = [[output.text for output in outputs[i].outputs] for i in range(len(outputs))]
        return rollouts
    
    def verify_answer(self, rollouts, sample_indices):
        """
        Verify the rollouts using the math_verify library.
        Considering rollouts that gained score above the threshold as 'Accepted'.
        NOTE: Since we empolied binary score here, so only rollouts with correct answers are 'Accepted'.
        """
        if len(sample_indices) < len(self.dataset):
            batch = [self.dataset[index] for index in sample_indices]
        else:
            batch = self.dataset

        # Format rollouts
        collected_data_all = []
        for i, item in enumerate(tqdm(batch, desc=f"Formatting rollouts , rank {self.args.local_rank}")):
            sample = {"problem": item["problem"], "answer": item["answer"], "outputs": []}
            for j in range(len(rollouts[0])):
                sample["outputs"].append(rollouts[i][j])
            collected_data_all.append(sample)

        # Verify rollouts
        collected_data_accpected = []
        correct_indices = []
        for i, item in enumerate(tqdm(batch, desc=f"Verifying rollouts, rank {self.args.local_rank}")):
            sample = {"problem": item["problem"], "answer": item["answer"], "outputs": []}
            correct_indices_ = []
            for j in range(len(rollouts[0])):
                try:
                    score = utils.compute_score_math_verify(rollouts[i][j], item["answer"])
                except Exception as e:
                    print(f"Cannot verify {sample} due to error: {e}")
                    score = 0
                if score > self.args.correct_threshold:
                    correct_indices_.append(j)
                    sample["outputs"].append(rollouts[i][j])
            correct_indices.append(correct_indices_)
            collected_data_accpected.append(sample)
        return collected_data_all, collected_data_accpected, correct_indices
    
    def update_difficulty(self, sample_indices, correct_indices):
        """
        Here we have two 'indices':
        - `sample_indices` is the index of the prompt in the dataset
        - `correct_indices` is the index of the trajectory in the rollouts list

        NOTE: The difficulty is not yet transferred to the sampling probabilities.
        We will perform the transformation in the `EpicSampler` during the downstream training process.
        """
        # Update counter
        for i, indices in enumerate(correct_indices):
            item_idx = sample_indices[i]
            delta_accepted_cnt = len(indices)
            self.counter[sample_indices[i]]["Accepted"] += delta_accepted_cnt
            self.counter[sample_indices[i]]["All"] += self.sample_sizes[item_idx]
        # Update difficulty
        problem_difficulties = []
        for cnt in self.counter:
            alpha = cnt["Accepted"]
            beta = cnt["All"] - alpha
            beta_dist = Beta(alpha + 1, beta + 1)
            difficulty = beta_dist.mean.item()
            problem_difficulties.append(difficulty)
        self.difficulties = np.array(problem_difficulties)

    def init_difficulty(self, llm):
        """
        Initialize the difficulty distribution with `num_rollout_min` rollouts per prompt for all prompts in the dataset
        """
        sample_indices = np.arange(len(self.dataset)).astype("int")
        rollouts = self.collect_rollouts(sample_indices, llm)
        collected_data_all, _, correct_indices = self.verify_answer(rollouts, sample_indices)
        self.update_difficulty(sample_indices, correct_indices)
        return collected_data_all, correct_indices

    def init_sample_size(self, collected_data_all, correct_indices, hf_model):
        """
        Initialize the sample size distribution with `num_rollout_max` rollouts per prompt for all prompts in the dataset
        NOTE: We use the implementation according to eq. 127 in our paper, which is different from GVM.
        """
        for n, p in hf_model.named_parameters():
            if self.args.act_params not in n:
                p.requires_grad = False
        params = [p for p in hf_model.parameters() if p.requires_grad]
        hf_model.cuda()
        all_grads = self.cal_grad_logp(collected_data_all, correct_indices, hf_model, params)
        all_sigmas = []
        for grads, correct_index in zip(all_grads, correct_indices):
            alpha, beta = len(correct_index), self.args.num_rollout_min - len(correct_index)
            alpha, beta = alpha / self.args.num_rollout_min, beta / self.args.num_rollout_min
            sigma = alpha * (beta ** 2) * np.mean(np.power(grads['correct'], 2))
            sigma += (alpha ** 2) * beta * np.mean(np.power(grads['wrong'], 2))
            sigma -= (alpha ** 2) * (beta ** 2) * np.power((np.mean(grads['correct']) - np.mean(grads['wrong'])), 2)
            sigma = np.sqrt(sigma)
            all_sigmas.append(sigma)
        rollout_budget = self.args.batch_size * self.args.num_rollout_max
        weights = np.array(all_sigmas) / np.sum(all_sigmas)
        self.sample_sizes = np.round(rollout_budget * weights).astype("int").tolist()

    def cal_grad_logp(self, collected_data_all, correct_indices, hf_model, params):
        r"""
        Calculate $\nabla \log p(y_i|x_i)$ for all samples in the dataset
        """
        all_grads = []
        for i, sample in enumerate(tqdm(collected_data_all, desc=f"Calculating gradients, rank {self.args.local_rank}")):
            grads = {
                "correct": [],
                "wrong": []
            }
            if len(sample["outputs"]) == 0:
                grads = {
                    "correct": [0],
                    "wrong": [0]
                }
            else:
                grads = {
                    "correct": [],
                    "wrong": []
                }
                for j, output in enumerate(sample['outputs']):
                    conv = [
                        {'role': 'system', 'content': utils.SYSTEM_PROMPTS[self.args.system_prompt]},
                        {'role': 'user', 'content': sample['problem'] + self.cot_conductor}
                    ]
                    conv_chat = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                    resp_start = len(tokenizer(conv_chat)['input_ids'])
                    conv_chat += output
                    input_ids = tokenizer(conv_chat, return_tensors='pt').input_ids.to(hf_model.device)
                    o = hf_model(input_ids, output_hidden_states=True)
                    logits = o.logits
                    log_probs = nn.functional.log_softmax(logits, dim=-1)
                    if resp_start == -1:
                        grad_norm = 0
                    else:
                        output_log_probs = log_probs[0, resp_start-1:-1]
                        resp_input_ids = input_ids[0, resp_start:]
                        indices = resp_input_ids.unsqueeze(1)
                        seq_logp = torch.gather(output_log_probs, 1, indices).squeeze(1).sum()
                        loss = -seq_logp
                        gradients = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)[0]
                        grad_norm = torch.norm(gradients, p=2).item()
                    if j in correct_indices[i]:
                        grads["correct"].append(grad_norm)
                    else:
                        grads["wrong"].append(grad_norm)
            if len(grads['correct']) == 0:
                grads["correct"] = [0]
            if len(grads['wrong']) == 0:
                grads["wrong"] = [0]
            all_grads.append(grads)
            model.zero_grad()
            torch.cuda.empty_cache()
        return all_grads

    def save_initialization(self, path):
        """
        Save the sample size, difficulties and counter to the given path
        """
        rank = self.args.local_rank
        sample_size_path = os.path.join(path, f"sample_size_rank_{rank}.json")
        json.dump(self.sample_sizes, open(sample_size_path, "w"))
        difficulty_path = os.path.join(path, f"difficulty_rank_{rank}.json")
        json.dump(self.difficulties.tolist(), open(difficulty_path, "w"))
        counter_path = os.path.join(path, f"counter_rank_{rank}.json")
        json.dump(self.counter, open(counter_path, "w"))

    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    utils.set_seed(script_args.seed)

    # If the dataset is already processed and saved, skip this stage
    if os.path.exists(os.path.join(script_args.output_dir, "difficulty_rank_all.json")):
        print("Initialization has been done before.")
        exit()

    # Load raw dataset
    try:
        dataset = load_dataset(script_args.data_path)[script_args.data_split]
    except Exception as e:
        print(
            f"Failed to load {script_args.data_path} using `load_dataset` function."
            + f"\nError message: {e}"
            + "\nTrying to use `load_from_disk` instead..."
        )
        dataset = load_from_disk(script_args.data_path)

    # Ensure end index does not exceed the dataset size
    data_size = len(dataset)
    script_args.end = min(data_size, script_args.end)
    # Cut-off the dataset to the desired range
    one_num_share = data_size // script_args.world_size
    dataset = dataset.select(range(script_args.start, script_args.end))
    if script_args.local_rank == script_args.world_size - 1:
        dataset = dataset.select(range(one_num_share * script_args.local_rank, data_size))
    else:
        dataset = dataset.select(range(one_num_share * script_args.local_rank, one_num_share * (script_args.local_rank + 1)))
    print(f'Local index: {script_args.local_rank}, World size: {script_args.world_size}, Data size: {len(dataset)}')
    print(f'Start: {one_num_share * script_args.local_rank}, End: {one_num_share * script_args.local_rank + len(dataset)}')
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    # Load the model for sampling
    llm = LLM(
        script_args.model_name_or_path,
        dtype=torch.bfloat16
    )
    # Build EPIC sampler
    epic_sampler = EPICInitializer(script_args=script_args, tokenizer=tokenizer, dataset=dataset)
    # Initialize difficulty distribution
    collected_data_all, correct_indices = epic_sampler.init_difficulty(llm)
    # Discard vLLM model and load HF model
    destroy_model_parallel()
    destroy_distributed_environment()
    llm.llm_engine.engine_core.shutdown()
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("\n ⏰ Waiting for vLLM to be fully released...\n")
    time.sleep(10)
    # Load HF model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    # Initialize sample size
    epic_sampler.init_sample_size(collected_data_all=collected_data_all, correct_indices=correct_indices, hf_model=model)
    # Save initialization
    os.makedirs(script_args.output_dir, exist_ok=True)
    epic_sampler.save_initialization(script_args.output_dir)