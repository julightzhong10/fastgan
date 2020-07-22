import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
import random
from tools.others import sample_ZCs
from scipy.stats import entropy
from scipy import linalg


def generate_imgs(netG,total, batch_size,img_width,z_dim,n_classes,Z_dist,Z_params,n_gpu):
    img_list = []
    for i in range(int(total/batch_size)):
        noise,conditions_int,conditions=sample_ZCs(batch_size,z_dim,n_classes,Z_dist,Z_params,n_gpu)
        tmp_imgs = netG(noise,conditions)
        tmp_imgs = tmp_imgs.detach().cpu().numpy()
        tmp_imgs = np.asarray(np.clip(tmp_imgs * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        img_list.append(tmp_imgs)
    img_list = np.asarray(img_list)
    img_list = img_list.reshape((total, 3, img_width, img_width))
    return img_list

def sample_imgs(train_loader,total,total_iters,img_width,dali=True):
    img_list = []
    for i, data in enumerate(train_loader, 0):
        if dali:
            x_real = data[0]["data"]
            y_real_c = data[0]["label"].view(-1).long()
        else:
            (x_real,y_real_c) = data
        #x_real = F.interpolate(input=x_real,size=(299, 299), mode='bilinear',align_corners=False)
        x_real = x_real.cpu().numpy()
        x_real = np.asarray(np.clip(x_real * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        img_list.append(x_real)
        if i>=total_iters-1:
            break
    img_list = np.asarray(img_list)
    #print(img_list.shape)
    img_list = img_list.reshape((total, 3, img_width, img_width))
    return img_list



def inception_score(netG,ipt_net, total, batch_size,z_dim,n_classes,img_width,n_splits,Z_dist,Z_params,n_gpu):
    scores = []
    for i in range(int(total/batch_size)):
        noise,conditions_int,conditions=sample_ZCs(batch_size,z_dim,n_classes,Z_dist,Z_params,n_gpu)
        imgs = netG(noise,conditions)
        if img_width != 299:
            imgs = F.interpolate(input=imgs,size=(299, 299), mode='bilinear',align_corners=False)
        #print(ipt_net(imgs))
        s = ipt_net(imgs)
        scores.append(s)
    scores=F.softmax(torch.cat(scores, 0), 1)
    split_scores=[]
    for i in range(n_splits):
        p_yx = scores[(i*scores.size(0)//n_splits):((i + 1)*scores.size(0)//n_splits)]
        p_y = p_yx.mean(0,keepdim=True).expand(p_yx.size(0), -1)
        #KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
        KL_d = p_yx * (torch.log(p_yx/p_y))
        score_mean= KL_d.sum(1).mean().exp()
        split_scores.append(score_mean)
    split_scores=torch.tensor(split_scores)
    f_mean,f_std=split_scores.mean(),split_scores.std()
    return f_mean,f_std

def inception_score_test(ipt_net,train_loader,img_width,n_splits,total_iters,n_gpu=0,dali=True):
    scores = []
    for i, data in enumerate(train_loader, 0):
        if dali:
            x_real = data[0]["data"]
            y_real_c = data[0]["label"].view(-1).long()
        else:
            (x_real,y_real_c) = data

        if n_gpu>0:
            x_real= x_real.cuda()
        if img_width != 299:
            imgs = F.interpolate(input=x_real,size=(299, 299), mode='bilinear',align_corners=False)
        s = ipt_net(imgs)
        scores.append(s)
        if i>total_iters:
            break
    scores=F.softmax(torch.cat(scores, 0), 1)
    split_scores=[]
    for i in range(n_splits):       
        p_yx = scores[(i*scores.size(0)//n_splits):((i + 1)*scores.size(0)//n_splits)]
        p_y = p_yx.mean(0,keepdim=True).expand(p_yx.size(0), -1)
        #KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
        KL_d = p_yx * (torch.log(p_yx/p_y))
        #KL_d = F.kl_div(p_yx.log(),p_y,reduce=False,reduction='none')
        #print(KL_d.size())
        score_mean= KL_d.sum(1).mean().exp()
        split_scores.append(score_mean)
    split_scores=torch.tensor(split_scores)
    f_mean,f_std=split_scores.mean(),split_scores.std()
    return f_mean,f_std

def cal_valaccu(model,data,n_gpu=0):
    correct=0
    total=0
    for i, (x, y_c_) in enumerate(data,0):
        if n_gpu>0:
            x, y_c_=x.cuda(), y_c_.cuda()
        with torch.no_grad():
            _,y_c = model(x)
            idx = torch.argmax(y_c.data, dim=1)
            label_correct = idx.eq(y_c_)
            correct += torch.sum(label_correct)
            total += y_c_.numel()
            #print(f'{correct},{total}')
    return correct.item()/total


                            ########### FID calculation ############


def get_activations_stat_orig(datasets, n_samples,ipt_net,ipt_dims, n_gpu,dali):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    curr_n_samples = 0
    ipt_net.eval()
    pred_arr = np.empty((n_samples, ipt_dims))
    #for i, data in enumerate(datasets, 0):
    i = 0
    loader_iter = iter(datasets)
    while True:
        try:
            data = next(loader_iter)
        except StopIteration:
            loader_iter = iter(datasets)
            data = next(datasets)
        if dali:
            imgs = data[0]["data"]
            y_real_c = data[0]["label"].squeeze().long()
        else:
            (imgs,y_real_c) = data
        # print('orig',i)
        start = i * imgs.size(0)
        end = start + imgs.size(0)
        if imgs.size(2) != 299 or imgs.size(3) != 299:
            imgs = F.interpolate(input=imgs,size=(299, 299), mode='bilinear',align_corners=False)
        if n_gpu>0:
            imgs = imgs.cuda()
        pred = ipt_net(imgs)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        # print(start,end,batch_size)
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(imgs.size(0), -1)
        curr_n_samples += imgs.size(0)
        if curr_n_samples>= n_samples:
            break
        i = i + 1
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu,sigma

def get_activations_stat_gen(netG,z_dim,n_classes,Z_dist,Z_params,ipt_net,total_itrs,batch_size, ipt_dims,n_gpu):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    ipt_net.eval()
    n_used_imgs = total_itrs * batch_size
    pred_arr = np.empty((n_used_imgs, ipt_dims))
    for i in range(total_itrs):
        # print('gen',i)
        Z, C_int, C_vec = sample_ZCs(batch_size,z_dim,n_classes,Z_dist,Z_params,n_gpu)
        start = i * batch_size
        end = start + batch_size
        imgs = netG(Z,C_vec)
        if imgs.size(2) != 299 or imgs.size(3) != 299:
            #imgs = imgs.data.mul_(0.5).add_(0.5).mul_(255).clamp_(0,255).round_().div_(255).mul_(2).sub_(1)
            imgs = F.interpolate(input=imgs,size=(299, 299), mode='bilinear',align_corners=False)
        pred = ipt_net(imgs)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu,sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def frechet_inception_distance(netG,ipt_net,z_dim,n_classes,Z_dist,Z_params,n_samples,batch_size, m2,s2,ipt_dims, n_gpu):
    """Calculates the FID of two paths"""
    total_itrs = int(n_samples/batch_size)
    m1, s1 = get_activations_stat_gen(netG,z_dim,n_classes,Z_dist,Z_params,ipt_net,total_itrs,batch_size, ipt_dims,n_gpu)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
