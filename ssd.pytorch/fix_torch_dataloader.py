import torch
from torch.utils.data import _utils
import torch.utils.data.sampler as sampler

# 保存原始函数
original_randperm = torch.randperm

# 完全替换 randperm 函数，忽略 CUDA 生成器
def patched_randperm(n, *, generator=None, **kwargs):
    if generator is not None and hasattr(generator, 'device') and generator.device.type != 'cpu':
        # 完全忽略 CUDA 生成器，不传递给原始函数
        return original_randperm(n, **kwargs)
    return original_randperm(n, generator=generator, **kwargs)

# 替换原函数
torch.randperm = patched_randperm

# 修补 RandomSampler 的 __iter__ 方法
original_random_sampler_iter = sampler.RandomSampler.__iter__

def patched_random_sampler_iter(self):
    # 强制临时设置 generator=None
    old_generator = getattr(self, 'generator', None)
    self.generator = None
    
    try:
        return original_random_sampler_iter(self)
    finally:
        # 恢复原始生成器
        self.generator = old_generator

# 替换方法
sampler.RandomSampler.__iter__ = patched_random_sampler_iter

# 修补 DataLoader 的内部函数
if hasattr(_utils.worker, '_IterableDatasetStopIteration'):
    original_init = _utils.worker._IterableDatasetStopIteration.__init__
    
    def patched_init(self, *args, **kwargs):
        if 'generator' in kwargs:
            kwargs.pop('generator')
        return original_init(self, *args, **kwargs)
    
    _utils.worker._IterableDatasetStopIteration.__init__ = patched_init

print("已应用 PyTorch DataLoader CUDA 生成器补丁")