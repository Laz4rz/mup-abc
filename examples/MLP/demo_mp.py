import torch.multiprocessing as mp
import itertools
import subprocess

import time
import math

import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.optim import SGD
import seaborn as sns
import pandas as pd

from mup import MuSGD, get_shapes, set_base_shapes, make_base_shapes, MuReadout


'''
    PyTorch MLP on CIFAR-10, with μP.

    This is minimal notebook that demonstrates the hyperparameter stability of muP.
'''

def chunk_jobs(jobs, n_chunks):
    """Split a list of jobs into n_chunks as evenly as possible with no jobs left out."""
    chunk_sizes = [len(jobs) // n_chunks] * n_chunks
    for i in range(len(jobs) % n_chunks):
        chunk_sizes[i] += 1

    chunks = []
    start = 0
    for size in chunk_sizes:
        chunks.append(jobs[start:start + size])
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

log_interval = 300

def train(model, device, train_loader, optimizer, epoch,
            scheduler=None, criterion=F.cross_entropy):
    model.train()
    train_loss = 0
    correct = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item() * data.shape[0]  # sum up batch loss
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     elapsed = time.time() - start_time
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} | ms/batch {:5.2f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item(),
        #         elapsed * 1000 / log_interval))
        #     start_time = time.time()
        if scheduler is not None:
            scheduler.step()
    train_loss /= len(train_loader.dataset)
    # print('\nEpoch {} Train set: Average loss: {:.4f}\n'.format(
    #     epoch, train_loss, correct, len(train_loader.dataset)))
    return train_loss

class MLP(nn.Module):
    def __init__(self, width=128, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult**0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        nn.init.zeros_(self.fc_3.weight)

    def forward(self, x):
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out) * self.output_mult


class muMLP(nn.Module):
    def __init__(self, width=128, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super(muMLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = MuReadout(width, num_classes, bias=False, output_mult=self.output_mult)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult**0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        nn.init.zeros_(self.fc_3.weight)

    def forward(self, x):
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out)



def run_jobs_on_gpu(gpu_id, uid, jobs, return_dict, shared_tensor_of_epochs):
    shared_tensor_of_epochs[uid] = 0
    nonlin = torch.relu
    criterion = F.cross_entropy
    torch.manual_seed(1)

    batch_size = 64
    epochs = 20
    data_dir = '/tmp'
    base_shapes_path = './demo_width256.bsh'
    kwargs = {'num_workers': 1, 'pin_memory': True}

    # optimal HPs
    output_mult = 32
    input_mult = 0.00390625

    print(f"[{gpu_id}:{uid}] Loading CIFAR-10 dataset", flush=True)
    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)
    testset = datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    device = torch.device(f'cuda:{gpu_id}')
    local_logs = []
    for width, log2lr in jobs:
        try:
            mynet = muMLP(width=width, nonlin=nonlin, output_mult=output_mult, input_mult=input_mult).to(device)
        except Exception as e:
            print(f'[{gpu_id}] Error creating model: {e}', flush=True)

        print(f"[{gpu_id}:{uid}] Training muP MLP with width {width} and log2lr {log2lr}, allocated process memory {torch.cuda.memory_allocated(gpu_id)/1024**3:.2f} GB", flush=True)
        set_base_shapes(mynet, base_shapes_path)

        optimizer = MuSGD(mynet.parameters(), lr=2**log2lr)

        for epoch in range(1, epochs + 1):
            train_loss = train(mynet, device, train_loader, optimizer, epoch, criterion=criterion)

            local_logs.append(dict(
                epoch=epoch,
                model_type='muP MLP',
                log2lr=log2lr,
                train_loss=train_loss,
                width=width,
                gpu_id=gpu_id,
                uid=uid,
            ))

            if math.isnan(train_loss):
                print(f"[{gpu_id}:{uid}] NaN loss at epoch {epoch}, width {width}, log2lr {log2lr}", flush=True)
                shared_tensor_of_epochs[uid] = torch.nan
                break

            shared_tensor_of_epochs[uid] = epoch

    return_dict[uid] = local_logs

if __name__ == '__main__':
    data_dir = '/tmp'
    base_shapes_path = './demo_width256.bsh'

    nonlin = torch.relu
    criterion = F.cross_entropy
    
    # Create base shapes
    base_shapes = get_shapes(MLP(width=256, nonlin=nonlin))
    delta_shapes = get_shapes(
        # just need to change whatever dimension(s) we are scaling
        MLP(width=256+1, nonlin=nonlin)
    )
    make_base_shapes(base_shapes, delta_shapes, savefile=base_shapes_path)

    # muP MLP
    widths = [256, 512, 1024, 2048, 4096, 8192]
    log2lrs = np.linspace(-8, 0, 20)
    combos = list(itertools.product(widths, log2lrs))

    gpus = get_available_gpus(min_free_mem_gb=40)
    print(f'Available GPUs: {gpus}')
    n_gpus = len(gpus)+1
    copies = 20
    jobs_split = [combos[i::n_gpus] for i in range(n_gpus)]

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    jobs_per_worker = chunk_jobs(combos, n_gpus * copies)
    gpu_ids = [gpu for gpu in gpus for _ in range(copies)]

    shared_tensor_of_epochs = (-1) * torch.ones(copies*len(gpus), dtype=torch.float32)
    jobst = 0
    for uid, (gpu_id, joblist) in enumerate(zip(gpu_ids, jobs_per_worker)):
        if len(joblist) == 0:
            print(f'[{gpu_id}:{uid}] No jobs to run', flush=True)
            continue
        p = mp.Process(target=run_jobs_on_gpu, args=(gpu_id, uid, joblist, return_dict, shared_tensor_of_epochs))
        print(f'Starting GPU {gpu_id}:{uid} with {len(joblist)} jobs', flush=True)
        p.start()
        processes.append(p)

    while any(p.is_alive() for p in processes):
        print(shared_tensor_of_epochs.reshape(10, -1), flush=True)
        time.sleep(10)

    for p in processes:
        p.join()

    # Combine logs from all GPUs
    logs = []
    for gpu_logs in return_dict.values():
        logs.extend(gpu_logs)
    logs.sort(key=lambda x: (x['width'], x['log2lr'], x['epoch']))

    import json
    with open('mup_mlp_logs.json', 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"Finished training {len(logs)} models", flush=True)