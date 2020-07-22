import os
import time
import torch.utils.data
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision.datasets as datasets
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

# credit: https://github.com/tanglang96/DataLoaders_DALI/blob/master/imagenet.py

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, data_dir, crop, shuffle, scale, ratio, world_size=1,seed=10,da=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, 0,seed=seed) # , RANDOM_SEED_TRAINING
        self.input = ops.FileReader(file_root=data_dir, num_shards=world_size, pad_last_batch=False,
                                    prefetch_queue_depth=2,random_shuffle=shuffle)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=scale,random_aspect_ratio=ratio)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                            std=[0.5 * 255, 0.5 * 255, 0.5 * 255])
        self.da = da
        if self.da:
            self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        if self.da:
            rng = self.coin()
            output = self.cmnp(images, mirror=rng)
        else:
            output = self.cmnp(images)
        return [output, self.labels]

def get_imgs_iter_dali(image_dir, batch_size, num_threads, num_gpus,  crop, shuffle, scale, ratio,seed,da=False):
    classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads,
                                data_dir=image_dir,
                                crop=crop, shuffle=shuffle, scale=scale, ratio=ratio, world_size=num_gpus,seed=seed,da=da)
    pip_train.build()
    dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // num_gpus, auto_reset=True)
    return dali_iter_train, class_to_idx