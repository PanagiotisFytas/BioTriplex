# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import random
from itertools import islice

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        data_source.in_batch_sampler = True
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        data_source.in_batch_sampler = False
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class LengthBasedWeightedBatchSampler(LengthBasedBatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True, weights=None,
                 num_samples=None, replacement=True, generator=None) -> None:
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        super().__init__(data_source, batch_size, drop_last, shuffle)

    def __iter__(self):
        # do weighted sampling of the lengths
        sampled_ids = list(WeightedRandomSampler(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=self.generator
        ))
        sampled_lengths = [self.lengths[i] for i in sampled_ids]
        ids = np.argsort(sampled_lengths, kind='mergesort')
        sorted_sampled_ids = [sampled_ids[i] for i in ids]
        if self.drop_last:
            sorted_sampled_ids = sorted_sampled_ids[:len(sorted_sampled_ids) // self.batch_size * self.batch_size]

        batches = [sorted_sampled_ids[i:i+self.batch_size] for i in range(0, len(sorted_sampled_ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


class DistributedLengthBasedWeightedBatchSampler(LengthBasedWeightedBatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0, weights=None,
                 num_samples=None, replacement=True, generator=None) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedWeightedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle, weights=weights,
            num_samples=num_samples, replacement=replacement, generator=generator
            )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


if __name__ == "__main__":
    weights = [5.,1.,1.,1.,1.,5.,1.]
    num_samples = len(weights)
    dataset = [list(range(5)), list(range(100)), list(range(15)), list(range(30)), list(range(1)), list(range(10)), list(range(3))]
    batch_size = 1
    drop_last = True
    shuffle = True
    print("dataset lengths")
    print(list(enumerate([len(d) for d in dataset])))
    print("weights with idx")
    print(list(enumerate(weights)))
    sampler = LengthBasedWeightedBatchSampler(dataset, batch_size, drop_last, shuffle, weights, num_samples)
    print(list(sampler))
    print(len(sampler))