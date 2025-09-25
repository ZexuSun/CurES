import os
import json
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    output_dir: Optional[str] = field(default=0, metadata={"help": "The output directory of initialization results"})

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    root = script_args.output_dir
    
    difficulty_files = [f for f in os.listdir(root) if 'difficulty' in f]
    ordered_total_difficulty = []
    for rank in range(len(difficulty_files)):
        ordered_total_difficulty.extend(json.load(open(os.path.join(root, f"difficulty_rank_{rank}.json"), "r")))
    json.dump(ordered_total_difficulty, open(os.path.join(root, "difficulty_rank_all.json"), "w"))

    sample_size_files = [f for f in os.listdir(root) if 'sample_size' in f]
    ordered_total_sample_size = []
    for rank in range(len(sample_size_files)):
        ordered_total_sample_size.extend(json.load(open(os.path.join(root, f"sample_size_rank_{rank}.json"), "r")))
    json.dump(ordered_total_sample_size, open(os.path.join(root, "sample_size_rank_all.json"), "w"))
    
    counter_files = [f for f in os.listdir(root) if 'counter' in f]
    ordered_total_counter = []
    for rank in range(len(counter_files)):
        ordered_total_counter.extend(json.load(open(os.path.join(root, f"counter_rank_{rank}.json"), "r")))
    json.dump(ordered_total_counter, open(os.path.join(root, "counter_rank_all.json"), "w"))
