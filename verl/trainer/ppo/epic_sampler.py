import numpy as np
from torch.utils.data.sampler import Sampler
from torch.distributions.beta import Beta
from typing import Iterator

class EpicSampler(Sampler[int]):
    def __init__(self, dataset_size, difficulties, batch_size, entropy_param=0.1, replacement=False) -> None:
        self.dataset_size = dataset_size
        self.difficulties = np.array(difficulties, dtype=np.float32)
        self.batch_size = batch_size
        self.entropy_param = entropy_param
        self.replacement = replacement

    def __iter__(self) -> Iterator[int]:
        all_indices = np.arange(len(self.dataset))
        sampling_weights = self.difficulties
        sampling_scores = np.exp(sampling_weights * (1 - sampling_weights) / self.args.entropy_param)
        probs = sampling_scores / sampling_scores.sum()

        if self.replacement:
            while True:
                batch = np.random.choice(all_indices, size=self.batch_size, replace=True, p=probs)
                yield batch.tolist()
        else:
            remaining_indices = all_indices.copy()
            while len(remaining_indices) >= self.batch_size:
                batch = np.random.choice(remaining_indices, size=self.batch_size, replace=False, p=probs)
                yield batch.tolist()
                mask = np.isin(remaining_indices, batch, invert=True)
                remaining_indices = remaining_indices[mask]
                if len(remaining_indices) < self.batch_size:
                    break
        
        def __len__(self) -> int:
            if self.replacement:
                return np.inf
            else:
                return len(self.dataset) // self.batch_size
        
        def update_dfficulty(self):
            pass
