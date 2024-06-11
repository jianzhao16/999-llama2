import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def train(gpu, args):
    # Corrected line: using bracket notation to access dictionary values
    rank = args['nr'] * args['gpus'] + gpu
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args['world_size'], rank=rank)
    torch.cuda.set_device(gpu)
    # your training code here

def main():
    world_size = 4  # total number of processes
    gpus_per_node = 4  # number of gpus per node
    args = {'world_size': world_size, 'gpus': gpus_per_node, 'nr': 0}  # nr is node rank
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(world_size)
    mp.spawn(train, nprocs=gpus_per_node, args=(args,))

if __name__ == "__main__":
    main()
