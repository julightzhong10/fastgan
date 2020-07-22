import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, FashionMNIST,STL10
from torch.utils.data import DataLoader
import numpy as np

ANNEAL_TYPE  = ['linear','exp']

def load_optimizer(model,lr=2e-4,betas=(0,0.9)):
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=betas)
    return optimizer

def load_inception_FID(num_gpu,ipt_dims):
    from tools.networks.inception_eval.FID_inception_v3 import InceptionV3_FID
    block_idx = InceptionV3_FID.BLOCK_INDEX_BY_DIM[ipt_dims]
    inception_model = InceptionV3_FID([block_idx])
    if num_gpu>0:
        inception_model.cuda()
        inception_model = torch.nn.DataParallel(inception_model, device_ids=range(num_gpu))
    inception_model.eval()
    return inception_model


def load_inception_IS(num_gpu):
    from torchvision.models import inception_v3
    inception_model = inception_v3(pretrained=True, transform_input=False)
    if num_gpu>0:
        inception_model.cuda()
        inception_model = torch.nn.DataParallel(inception_model, device_ids=range(num_gpu))
    inception_model.eval()
    return inception_model


def load_models(net="sngan_catdog64",n_gpu=1,dim_z=128,img_c=3,img_width=64,fm_base_d=64,fm_base_g=64,bn_d=True,bn_g=True,sn_d=False,sn_g=False,sa_d=False,sa_g=False,\
                bottom_width=4,n_classes=10,c_metric='',wide=False):
    if net == "sngan_cifar":
        from tools.networks.gan.sn_resnet_cifar10 import SN_ResNet_CIFAR10_Gen,SN_ResNet_CIFAR10_Dis
        gen = SN_ResNet_CIFAR10_Gen(dim_z=dim_z,img_c=img_c,fm_base=fm_base_g, bottom_width=bottom_width, n_classes=n_classes,bn=bn_g,sn=sn_g,sa=sa_g)
        dis = SN_ResNet_CIFAR10_Dis(img_c=img_c,fm_base=fm_base_d, n_classes=n_classes,bn=bn_d,sn=sn_d,sa=sa_d,c_metric=c_metric,wide=wide)
    elif net == "sngan_imgnet128":
        from tools.networks.gan.sn_resnet_imgnet128 import SN_ResNet_ImgNet128_Gen,SN_ResNet_ImgNet128_Dis
        gen = SN_ResNet_ImgNet128_Gen(dim_z=dim_z,img_c=img_c,fm_base=fm_base_g, bottom_width=bottom_width, n_classes=n_classes,bn=bn_g,sn=sn_g,sa=sa_g)
        dis = SN_ResNet_ImgNet128_Dis(img_c=img_c,fm_base=fm_base_d, n_classes=n_classes,bn=bn_d,sn=sn_d,sa=sa_d,c_metric=c_metric,wide=wide)
    elif net == "sngan_catdog64":
        from tools.networks.gan.sn_resnet_catdog64 import SN_ResNet_CatDog64_Gen ,SN_ResNet_CatDog64_Dis
        gen = SN_ResNet_CatDog64_Gen(dim_z=dim_z,img_c=img_c,fm_base=fm_base_g, bottom_width=bottom_width, n_classes=n_classes,bn=bn_g,sn=sn_g,sa=sa_g)
        dis = SN_ResNet_CatDog64_Dis(img_c=img_c,fm_base=fm_base_d, n_classes=n_classes,bn=bn_d,sn=sn_d,sa=sa_d,c_metric=c_metric,wide=wide)
    elif net == "sngan_catdog128":
        from tools.networks.gan.sn_resnet_catdog128 import SN_ResNet_CatDog128_Gen ,SN_ResNet_CatDog128_Dis
        gen = SN_ResNet_CatDog128_Gen(dim_z=dim_z,img_c=img_c,fm_base=fm_base_g, bottom_width=bottom_width, n_classes=n_classes,bn=bn_g,sn=sn_g,sa=sa_g)
        dis = SN_ResNet_CatDog128_Dis(img_c=img_c,fm_base=fm_base_d, n_classes=n_classes,bn=bn_d,sn=sn_d,sa=sa_d,c_metric=c_metric,wide=wide)
    elif net == "biggan":
        from tools.networks.gan.biggan import BigGAN_Gen,BigGAN_Dis
        gen = BigGAN_Gen(dim_z=dim_z,G_ch=fm_base_g,bottom_width=bottom_width,resolution=img_width,n_classes=n_classes,bn=bn_g,sn=sn_g,sa=sa_g,G_init='ortho')
        dis = BigGAN_Dis(D_ch=fm_base_d,resolution=img_width,n_classes=n_classes,bn=bn_d,sn=sn_d,sa=sa_d,c_metric=c_metric,wide=wide,D_init='ortho')
    else:
        raise ValueError(f"Unknown model name: {net}")
    if n_gpu > 0:
        gen, dis = gen.cuda(), dis.cuda()
        gen, dis = nn.DataParallel(gen, device_ids=range(n_gpu)), \
                   nn.DataParallel(dis, device_ids=range(n_gpu))
    return gen, dis


def load_models_gen(net="sngan_catdog64",n_gpu=1,dim_z=128,img_c=3,img_width=64,fm_base=64,bn=False,sn=False,sa=False,bottom_width=4,n_classes=10,load_path='./'):
    if net == "sngan_cifar":
        from tools.networks.gan.sn_resnet_cifar10 import SN_ResNet_CIFAR10_Gen
        gen = SN_ResNet_CIFAR10_Gen(dim_z=dim_z,img_c=img_c,fm_base=fm_base, bottom_width=bottom_width, n_classes=n_classes,bn=bn,sn=sn,sa=sa)
    elif net == "sngan_imgnet128":
        from tools.networks.gan.sn_resnet_imgnet128 import SN_ResNet_ImgNet128_Gen
        gen = SN_ResNet_ImgNet128_Gen(dim_z=dim_z,img_c=img_c,fm_base=fm_base, bottom_width=bottom_width, n_classes=n_classes,bn=bn,sn=sn,sa=sa)
    elif net == "sngan_catdog64":
        from tools.networks.gan.sn_resnet_catdog64 import SN_ResNet_CatDog64_Gen
        gen = SN_ResNet_CatDog64_Gen(dim_z=dim_z,img_c=img_c,fm_base=fm_base, bottom_width=bottom_width, n_classes=n_classes,bn=bn,sn=sn,sa=sa)
    elif net == "sngan_catdog128":
        from tools.networks.gan.sn_resnet_catdog128 import SN_ResNet_CatDog128_Gen
        gen = SN_ResNet_CatDog128_Gen(dim_z=dim_z,img_c=img_c,fm_base=fm_base, bottom_width=bottom_width, n_classes=n_classes,bn=bn,sn=sn,sa=sa)
    elif net == "biggan":
        from tools.networks.gan.biggan import BigGAN_Gen
        gen = BigGAN_Gen(dim_z=dim_z,G_ch=fm_base,bottom_width=bottom_width,resolution=img_width,n_classes=n_classes,bn=bn,sn=sn,sa=sa,G_init='ortho')
    else:
        raise ValueError(f"Unknown model name: {net}")
    #gen.load_state_dict(torch.load(load_path))
    if n_gpu > 0:
        gen = gen.cuda()
        gen = nn.DataParallel(gen, device_ids=range(n_gpu))
    if load_path is not None:
        gen.load_state_dict(torch.load(load_path))
    #gen.eval()
    return gen


def load_models_dis(net="sngan_catdog64",num_gpu=1,img_c=3,img_width=64,fm_base=64,bn=False,sn=False,sa=False,bottom_width=4,
                n_classes=10,c_metric='',load_path='./',wide=False):
    if net == "sngan_cifar":
        from tools.networks.gan.sn_resnet_cifar10 import SN_ResNet_CIFAR10_Dis
        dis = SN_ResNet_CIFAR10_Dis(img_c=img_c,fm_base=fm_base, n_classes=n_classes,bn=bn,sn=sn,sa=sa,c_metric=c_metric,wide=wide)
    elif net == "sngan_imgnet128":
        from tools.networks.gan.sn_resnet_imgnet128 import SN_ResNet_ImgNet128_Dis
        dis = SN_ResNet_ImgNet128_Dis(img_c=img_c,fm_base=fm_base, n_classes=n_classes,bn=bn,sn=sn,sa=sa,c_metric=c_metric,wide=wide)
    elif net == "sngan_catdog64":
        from tools.networks.gan.sn_resnet_catdog64 import SN_ResNet_CatDog64_Dis
        dis = SN_ResNet_CatDog64_Dis(img_c=img_c,fm_base=fm_base, n_classes=n_classes,bn=bn,sn=sn,sa=sa,c_metric=c_metric,wide=wide)
    elif net == "sngan_catdog128":
        from tools.networks.gan.sn_resnet_catdog128 import SN_ResNet_CatDog128_Dis
        dis = SN_ResNet_CatDog128_Dis(img_c=img_c,fm_base=fm_base, n_classes=n_classes,bn=bn,sn=sn,sa=sa,c_metric=c_metric,wide=wide)
    elif net == "biggan":
        from tools.networks.gan.biggan import BigGAN_Dis
        dis = BigGAN_Dis(D_ch=fm_base,resolution=img_width,n_classes=n_classes,bn=bn,sn=sn,sa=sa,c_metric=c_metric,wide=wide,D_init='ortho')
    else:
        raise ValueError(f"Unknown model name: {net}")
    if num_gpu > 0:
        dis = dis.cuda()
        dis = nn.DataParallel(dis, device_ids=range(num_gpu))
    if load_path is not None:
        dis.load_state_dict(torch.load(load_path))
    #dis.eval()
    return dis


def load_datasets(data_path,img_width,batch_size,n_workers,n_gpu,shuffle=True,dali=False,seed = 10,da=False):
    loader, class_to_idx = None, None
    if dali and n_gpu>0:
        from tools.dali import get_imgs_iter_dali
        scale = [0.8, 1.0] if da else [1.0, 1.0]
        ratio = [1.0,1.0]
        loader, class_to_idx = get_imgs_iter_dali(image_dir=data_path,batch_size=batch_size,num_threads=n_workers,
                            num_gpus=n_gpu,crop=img_width,shuffle=shuffle, scale=scale,ratio=ratio,seed=seed,da=da)
    else:
        scale = (0.8, 1.0) if da else (1.0, 1.0)
        ratio = (1.0,1.0)
        loader = None
        if da:
            trans_ops = [tfs.RandomResizedCrop(img_width, scale=scale, ratio=ratio),
                    tfs.RandomHorizontalFlip(),
                    tfs.ToTensor(),
                    tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])]
        else:
            trans_ops = [tfs.RandomResizedCrop(img_width, scale=scale, ratio=ratio),
                    tfs.ToTensor(),
                    tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])]            
        trans = tfs.Compose(trans_ops)
        data = ImageFolder(data_path, transform=trans)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
        class_to_idx = data.class_to_idx 
    return loader, class_to_idx
