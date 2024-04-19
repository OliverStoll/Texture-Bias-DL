import os


def set_gpus(gpu_index: list[int] | int):
    if isinstance(gpu_index, list):
        cuda_visible_devices = ','.join(map(str, gpu_index))
    else:
        cuda_visible_devices = str(gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

