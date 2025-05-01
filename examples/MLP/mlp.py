import sys
import subprocess
import itertools
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.optim import SGD
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
import torch.multiprocessing as mp

def chunk_jobs(jobs, n_chunks):
    """Split a list of jobs into n_chunks as evenly as possible, tagging each job with a unique index."""
    chunk_sizes = [len(jobs) // n_chunks] * n_chunks
    for i in range(len(jobs) % n_chunks):
        chunk_sizes[i] += 1

    chunks = []
    start = 0
    idx = 0
    for size in chunk_sizes:
        chunk = []
        for job in jobs[start:start + size]:
            chunk.append((idx, job[0], job[1]))  # actually tag the job
            idx += 1
        chunks.append(chunk)
        start += size

    return chunks

def get_available_gpus(min_free_mem_gb=4):
    """Returns a list of GPU IDs with at least min_free_mem_gb available."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
    
    free_memories = [int(x) for x in result.stdout.strip().split('\n')]
    return [i for i, mem in enumerate(free_memories) if mem >= min_free_mem_gb * 1024]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
data_dir = '/tmp'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

# train on subset of the dataset
subset_percentage = 0.2
subset_size = int(len(train_ds) * subset_percentage)
indices = np.random.choice(len(train_ds), subset_size, replace=False)
train_subset = torch.utils.data.Subset(train_ds, indices)
train_dl = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
xs = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
ys = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])
preloaded_dataset = torch.utils.data.TensorDataset(xs, ys)
preloaded = torch.utils.data.DataLoader(preloaded_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class MLP(nn.Module):
    def __init__(self, width=128, num_classes=10):
        super().__init__()
        self.width = width
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.fc_1(x)
        h = F.relu(h)
        h = self.fc_2(h)
        h = F.relu(h)
        h = self.fc_3(h)
        return h
    
class muMLPTab9(nn.Module):
    def __init__(self, width=128, num_classes=10):
        super().__init__()
        self.width = width
        self.input_mult = self.width**0.5
        self.output_mult = self.width**-0.5
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.fc_1.weight, std=(self.width*3072)**-0.5)
        nn.init.normal_(self.fc_2.weight, std=self.width**-0.5)
        nn.init.normal_(self.fc_3.weight, std=self.width**-0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        activations = []
        h = self.input_mult * self.fc_1(x)
        activations.append(h)
        h = self.fc_2(F.relu(h))
        activations.append(h)
        h = self.output_mult * self.fc_3(F.relu(h))
        activations.append(h)
        return h
    
    def get_param_groups(self, base_lr):
        return [
            dict(params=self.fc_1.parameters(), lr=base_lr),
            dict(params=self.fc_2.parameters(), lr=base_lr),
            dict(params=self.fc_3.parameters(), lr=base_lr),
        ]

    
def train(model, train_dl, optimizer, num_epochs=1, device=device):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


epochs = 10
seeds = [0, 1, 2]
log2lrs = [-1, -2, -3, -4, -5, -6, -7, -8][::-1]
widths = [128, 256, 512]
results_df = pd.DataFrame(index=log2lrs, columns=widths)


# for log2lr in log2lrs:
#     for width in widths:
#         losses = []
#         for seed in tqdm(seeds):
#             torch.manual_seed(seed)
#             np.random.seed(seed)

#             # model = MLP(width=width).to(device)
#             model = muMLPTab9(width=width).to(device)
#             optimizer = SGD(model.parameters(), lr=2**log2lr, momentum=0.9, weight_decay=5e-4)
#             loss = train(model, train_dl, optimizer, num_epochs=epochs)
            
#             losses.append(loss)

#         loss = np.mean(losses)
#         results_df.loc[log2lr, width] = loss
#         print(f"Width: {width}, Log2LR: {log2lr}, Loss: {loss:.4f}, Losses: {[round(ls, 3) for ls in losses]}")

# plt.figure(figsize=(10, 6))
# for width in widths:
#     plt.plot(results_df.index, results_df[width], label=f'Width {width}')

mp.set_start_method('spawn', force=True)

def run_chunk(jobs, device, shared_tensor, preloaded, seeds, model_class):
    torch.cuda.set_device(device)
    for job in jobs:
        job_id, log2lr, width = job
        run_experiment(log2lr, width, seeds, job_id, device, shared_tensor, preloaded, model_class)

def run_experiment(log2lr, width, seeds, job_id, device, shared_tensor, preloaded, model_class):
    train_dl = preloaded
    losses = []
    print(f"Running job {job_id} on device {device} with log2lr={log2lr}, width={width}")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # model = MLP(width=width).to(device)
        model = model_class(width=width).to(device)
        optimizer = SGD(model.parameters(), lr=2**log2lr)
        loss = train(model, train_dl, optimizer, num_epochs=epochs, device=device)
        
        losses.append(loss)

    loss = np.mean(losses)
    shared_tensor[job_id] = loss
    print(f"Width: {width}, Log2LR: {log2lr}, Loss: {loss:.4f}, Losses: {[round(ls, 3) for ls in losses]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP or muMLP model.")
    parser.add_argument('--model', type=str, choices=['MLP', 'muMLP'], required=True, help="Choose the model type: 'MLP' or 'muMLP'")
    args = parser.parse_args()
    print(f"Running with model: {args.model}")

    model_class = MLP if args.model == 'MLP' else muMLPTab9
    epochs = 20
    seeds = [0, 1, 2, 3, 4]
    log2lrs = np.linspace(-8, 0, 20)
    widths = [128, 256, 512, 1024, 2048]

    devices = [f"cuda:{i}" for i in get_available_gpus(min_free_mem_gb=8)]
    print(f"Available devices: {len(devices)}, {[d.split(':')[-1] for d in devices]}")

    jobs = list(itertools.product(log2lrs, widths))
    jobs_chunks = chunk_jobs(jobs, len(devices))
    print(f"Jobs: {len(jobs)}, Chunks: {len(jobs_chunks)}")
    
    processes = []
    shared_tensor = torch.zeros(len(jobs)).to(device).share_memory_()
    for enum, job_chunk in enumerate(jobs_chunks):
        device = devices[enum]

        print(f"Starting process {enum} on {device} with {len(job_chunk)} jobs")
        p = mp.Process(target=run_chunk, args=(job_chunk, device, shared_tensor, preloaded, seeds, model_class))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


    results_df = pd.DataFrame(index=log2lrs, columns=widths)
    for i, job in enumerate(jobs):
        log2lr, width = job
        loss = shared_tensor[i].item()
        results_df.loc[log2lr, width] = loss

    plt.figure(figsize=(8, 4))
    for width in widths:
        plt.plot(results_df.index, results_df[width], label=f'Width {width}')
    plt.xlabel('Log2LR')
    plt.ylabel('Loss')
    plt.title(f'{model_class}\nLoss vs Log2LR for different widths')
    plt.legend()
    plt.grid()
    plt.savefig(f'loss_vs_log2lr_{args.model}.png')
    plt.show()
