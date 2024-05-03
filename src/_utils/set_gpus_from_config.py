import os

from utils.config import CONFIG


def set_gpus(gpu_index: list[int] | int):
    if isinstance(gpu_index, list):
        cuda_visible_devices = ','.join(map(str, gpu_index))
    else:
        cuda_visible_devices = str(gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices


gpu_indexes = CONFIG['gpu_indexes']
assert gpu_indexes is not None, 'gpu_indexes is not set in config file'
set_gpus(gpu_indexes)
