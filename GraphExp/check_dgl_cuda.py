import dgl
import torch

if torch.cuda.is_available():
    print("PyTorch CUDA is available!")
    try:
        dgl.cuda.empty_cache() # Try to use DGL CUDA functionality
        print("DGL CUDA is also working!")
    except Exception as e:
        print(f"DGL CUDA is NOT working: {e}")
else:
    print("PyTorch CUDA is NOT available.")