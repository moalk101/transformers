from torch.utils.data import Sampler
import random

class TokenBatchSampler(Sampler):
    def __init__(self, lengths, max_tokens, shuffle=True):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        total_tokens = 0

        for idx in indices:
            length = self.lengths[idx]

            if total_tokens + length > self.max_tokens and batch:
                yield batch
                batch = []
                total_tokens = 0

            batch.append(idx)
            total_tokens += length

        if batch:
            yield batch

    def __len__(self):
        return len(self.lengths)