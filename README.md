# FastGAN
The official code release of the paper: [*Improving the Speed and Quality of GAN by Adversarial Training*](https://arxiv.org/abs/2008.03364)

## Requirements
Code are mainly implemented via using PyTorch.
Below is our running environment:

+ Python == 3.6 
+ PyTorch == 1.2.0
+ Tensorflow ==1.10.0 （optional, required if you want to calculate the official TF Inception score ）
+ Nvidia dali >= 0.12 （optional, required if you want to use nvidia dali to read and process data. Using dali requires GPU(s). [Reference](https://github.com/tanglang96/DataLoaders_DALI) for install）

## File structure

### codes
+ `train_free.py`: FastGAN training 
+ `train.py`: RobGAN, SNGAN, SAGAN training
+ `eval_[IS, FID, IS_single, FID_single].py`: IS and FID evaluation
+ `cal_FID_stat.py`: computing the FID stat to .npz from data
+ `/tools/dali.py`: Nvidia dali 
+ `/tools/evalution.py`: Evaluation tools
+ `/tools/loading.py`: models and data loading tools
+ `/tools/metric.py`: loss objective tools
+ `/tools/adv_tools/[linf_sgd, pgd_free, pgd].py`: pgd adv training tools
+ `/tools/networks/gan/`: GAN model archetecture tools
+ `/tools/networks/inception_eval/`: Inception model for eval

### scripts
+ `/scripts_train/*.sh`: scripts for training
+ `/scripts_eval/*.sh`: scripts for evaluation

## Data Preparation

We use data [Pytorch DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) style tools for reading images, please refer and prepare the images according to the format. Basicly, you need to put imgs files under classes folder, and read the parent folder of those class folders.

Datasets source:
+ [CIFAR10 and CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
+ [ImageNet-143 and ImageNet](https://github.com/pfnet-research/sngan_projection#preprocess-dataset): We follow the steps in [SNGAN](https://github.com/pfnet-research/sngan_projection)



## Training
We prepare the training code in `./scripts_train/` folder.  Here we assume the path you want to save model is `/home/fastgan/cifar10/` and 10 class folders of images are placed  under `/home/fastgan/data/cifar10/`
Below are a cifar10 FastGAN training example for introduction (cannot be directly used due to the comments):
```
save_path=/home/fastgan/cifar10/
data_path=/home/fastgan/data/cifar10 
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python3.6 -u ./train_free.py \
    --random_seed 10 \ # you may set your lucky random seed  
    --net sngan_cifar \ # select cifar10 sngan model backbone
    --dim_z 128 \ # latent code dimension
    --img_c 3 \ # image channel 
    --fm_base_d 128 --fm_base_g 128 \ # feature maps sizes
    --n_classes 10 \ # No. of classes
    --num_bp 1 \ # if you don't have enough GPU memory you may use mutiple bp steps to train the model. The `num_bp` here mainly break the batch size into `num_bp` parts to proccess 
    --r_batch_size 64 --data_path ${data_path} \ # set real data batch size
    --g_batch_size 64 \ # set fake/generated data batch size
    --shuffle \ # shuffle the data
    --out_path ${save_path} \ # set save path
    --bottom_width 4 \ # the base width of feature map in G(4 for most backbones)
    --img_width 32 \ # width of images
    --itr_d 1 --lr_d 0.0002 --beta1_d 0.0 --beta2_d 0.9 \ # max_D_step, adam lr, beta1, and beta2 for D
    --itr_g 1 --lr_g 0.0002 --beta1_g 0.0 --beta2_g 0.9 \ # adam lr, beta1, and beta2 for G
    --anneal_lr exp --anneal_lr_p1 0.5 --anneal_lr_p2 80000 \ # lr rate decay settings
    --n_gpu 1 --dali --n_workers 1 --da \  # specify the number of GPU(s) visible to you, if use dali, no. of dataloader works, if data augmentation
    --G_total_itrs 240001 --save_bias 1000 --start_itr 0 \ # total training iter, save bias and start pts(itr/save_bias)
    --bn_g \ # c-bn in G
    --b_metric hinge --c_metric ce_kl \ # choose different loss types
    --c_alph_f 1.0 --c_alph_g 1.0 \ # alph hyper-params
    --pgd_type Linf --pgd_free_steps 2 --pgd_eps 0.006 --pgd_tau 0.006 # adv training: norm type is L infinty, pgd_eps is the max bound of PGD, pgd_tau is the step size of PGD

```


## Evaluation
We prepare the evaluation code in `./scripts_eval/` folder. You may use `eval_IS_single.sh` and `eval_FID_single.sh` to measure a single model file(.pth). Or you may use `eval_IS.sh` and `eval_FID.sh` to eval a sequence of check points.

eval_IS.sh example:
```
#!/bin/bash
load_path=/path/to/the/check_pts/
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python3.6 -u ./eval_IS.py \
    --net sngan_cifar10 \
    --dim_z 128 \
    --img_c 3 \
    --fm_base 128 \
    --n_classes 10 \
    --bottom_width 4 \
    --img_width 32 \
    --n_gpu 1 --dali \ 
    --bn \ 
    --Z_dist normal --Z_p1 0 --Z_p2 1 \ # above are the setting for GAN training
    --load_path ${load_path} \
    --start 1 --end 400 --skip 1 \ # start .pth, end .pth, and the skip interval
    --batch_size 100 \
    --n_splits 10 \
    --n_samples 10000 --tf # No. of samples, --tf is the flag to se official tensorflow IS model (need to install tensorflow)
```
Notice that, `[eval_FID, eval_FID_single].sh` require `.npz` files which stores pre-calculated mean and covariance FID statistics for different dataset. 

Due to the sapce limitation we only provide the `.npz` files for CIFAR10(cifar10_FID_stat.npz) and ImageNet-143-64px (catdog143_64_FID_stat.npz) in the `../FID_stat` folder for now. 

Thus, we provide the code(`./scripts_eval/cal_FID_stat.sh`) for computing and saving FID statistics.



## Cite
Please consider to cite this paper if you find it helpful in your research:

    @article{zhong2020improving,
    title={Improving the Speed and Quality of GAN by Adversarial Training},
    author={Zhong, Jiachen and Liu, Xuanqing and Hsieh, Cho-Jui},
    journal={arXiv preprint arXiv:2008.03364},
    year={2020}
    }



## Acknowledgement

We would like to thank for the source code of [SN-GAN](https://github.com/pfnet-research/sngan_projection), [Rob-GAN](https://github.com/xuanqing94/RobGAN), [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) and any other resources which are helpful to us.
