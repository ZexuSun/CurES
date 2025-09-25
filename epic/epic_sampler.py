import numpy as np
from torch.utils.data.sampler import Sampler
from torch.distributions.beta import Beta
from typing import Iterator

class EpicSampler(Sampler[int]):
    def __init__(self, dataset, difficulties, counter, batch_size, entropy_param=0.1, replacement=False) -> None:
        self.dataset = dataset
        self.difficulties = np.array(difficulties, dtype=np.float32)
        self.alpha_vec = np.array([cnt["Accepted"] for cnt in counter], dtype=np.float32)
        self.batch_size = batch_size
        self.entropy_param = entropy_param
        self.replacement = replacement
        self.counter = counter

    def update_difficulty(self):
        # Update difficulty
        problem_difficulties = []
        alphas = []
        for cnt in (self.counter):
            alpha = cnt["Accepted"]
            beta = cnt["All"] - alpha
            beta_dist = Beta(alpha + 1, beta + 1)
            difficulty = beta_dist.mean.item()
            alphas.append(alpha)
            problem_difficulties.append(difficulty)
        self.difficulties = np.array(problem_difficulties)
        self.alpha_vec = np.array(alphas)

    def __iter__(self) -> Iterator[int]:
        all_indices = np.arange(len(self.dataset))
        sampling_weights = self.difficulties
        t_vec = (self.alpha_vec + 1) / 1e-2
        print(f"Sampling Weights:\n{sampling_weights}\n")
        sampling_scores = np.exp(np.sqrt(sampling_weights * (1 - sampling_weights)) / t_vec)
        probs = sampling_scores / sampling_scores.sum()
        print(f"Probabilities:\n{probs}\n")

        if self.replacement:
            while True:
                batch = np.random.choice(all_indices, size=self.batch_size, replace=False, p=probs)
                yield batch.tolist()
        else:
            remaining_indices = all_indices.copy()
            remaining_probs = probs.copy()
            while len(remaining_indices) >= self.batch_size:
                norm_probs = remaining_probs / remaining_probs.sum()
                batch = np.random.choice(remaining_indices, size=self.batch_size, replace=False, p=norm_probs)
                yield batch.tolist()
                mask = np.isin(remaining_indices, batch, invert=True)
                remaining_indices = remaining_indices[mask]
                remaining_probs = remaining_probs[mask]
                if len(remaining_indices) < self.batch_size: # Restart if the number of the dataset is not enough
                    remaining_indices = all_indices.copy()
                    remaining_probs = probs.copy()
        
    def __len__(self) -> int:
        if self.replacement:
            return 10
        else:
            return 10