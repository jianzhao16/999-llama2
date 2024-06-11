import gc
import torch

# After
torch.cuda.empty_cache()
#del variables
gc.collect()
