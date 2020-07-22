# credits:
#   https://github.com/pfnet-research/sngan_projection/blob/master/source/inception/inception_score_tf.py
#   https://github.com/bioinf-jku/TTUR/blob/master/fid.py
#   https://github.com/openai/improved-gan/pull/40/commits/0a09faccb45088695228bbf50435ee71e94eb2ce (bug fixing)
import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
from scipy import linalg
import math
import sys
tf.logging.set_verbosity(tf.logging.ERROR)
MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None
last_layer = None

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15


def inception_forward(images, layer):
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    bs = 100
    images = images.transpose(0, 2, 3, 1)
    with tf.Session(config=config) as sess:
        preds = []
        n_batches = int(math.ceil(float(len(images)) / float(bs)))
        for i in range(n_batches):
            #sys.stdout.write(".")
            #sys.stdout.flush()
            inp = images[(i * bs):min((i + 1) * bs, len(images))]
            pred = sess.run(layer, {'InputTensor:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
    return preds


def get_mean_and_cov(images):
    before_preds = inception_forward(images, last_layer)
    m = np.mean(before_preds, 0)
    cov = np.cov(before_preds, rowvar=False)
    return m, cov


def get_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        #msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        #warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean



# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
    preds = inception_forward(images, softmax)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
    global softmax
    global last_layer
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                    name='InputTensor')
        _ = tf.import_graph_def(graph_def, name='',
                                input_map={'ExpandDims:0':input_tensor})
    # Works with an arbitrary minibatch size.
    with tf.Session(config=config) as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                #o._shape = tf.TensorShape(new_shape)
                o.set_shape(new_shape)
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        last_layer = tf.squeeze(pool3, [1, 2])
        logits = tf.matmul(last_layer, w) 
        softmax = tf.nn.softmax(logits)


if softmax is None or last_layer is None:
    _init_inception()
