import argparse
import os
import json
import re
from unittest import result
import numpy as np
from tqdm import tqdm

import SimpleITK as sitk
import scipy
import skimage.io
import skimage.exposure
from skimage import measure, filters, morphology
import concurrent.futures
import time
import matplotlib
import matplotlib.pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('-d', '--dataset', type=str, default=None,
                    help='Specifies a data config')
parser.add_argument('-v', '--val_subset', type=str, default=None)
parser.add_argument('--batch', type=int, default=4, help='Size of minibatch')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--paired', action='store_true')
parser.add_argument('--data_args', type=str, default=None)
parser.add_argument('--net_args', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import network
import data_util.liver
import data_util.brain


def main():
    if args.checkpoint is None:
        print('Checkpoint must be specified!')
        return
    if ':' in args.checkpoint:
        args.checkpoint, steps = args.checkpoint.split(':')
        steps = int(steps)
    else:
        steps = None
    args.checkpoint = find_checkpoint_step(args.checkpoint, steps)
    # print(args.checkpoint)
    model_dir = os.path.dirname(args.checkpoint)
    try:
        with open(os.path.join(model_dir, 'args.json'), 'r') as f:
            model_args = json.load(f)
        # print(model_args)
    except Exception as e:
        print(e)
        model_args = {}

    if args.dataset is None:
        args.dataset = model_args['dataset']
    if args.data_args is None:
        args.data_args = model_args['data_args']

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = model_args['base_network']
    Framework.net_args['n_cascades'] = model_args['n_cascades']
    Framework.net_args['rep'] = args.rep
    Framework.net_args.update(eval('dict({})'.format(model_args['net_args'])))
    if args.net_args is not None:
        Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type')
    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get(
        'segmentation_class_value', None), fast_reconstruction=args.fast_reconstruction, validation=True)
    print('Graph built')

    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    ds = Dataset(args.dataset, batch_size=args.batch, paired=args.paired, **
                 eval('dict({})'.format(args.data_args)))

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES))
    checkpoint = args.checkpoint
    saver.restore(sess, checkpoint)
    tflearn.is_training(False, session=sess)

    val_subsets = [data_util.liver.Split.VALID]
    if args.val_subset is not None:
        val_subsets = args.val_subset.split(',')

    tflearn.is_training(False, session=sess)
    # keys = ['jaccs', 'dices', 'landmark_dists', 'jacobian_det', 
    # 'real_flow', 'image_fixed', 'warped_moving', 'warped_seg_moving']
    keys = ['dices', 'landmark_dists', 'jacobian_det', 'warped_moving', 'warped_seg_moving', 'real_flow']

    if not os.path.exists('evaluate'):
        os.mkdir('evaluate')
    path_prefix = os.path.join('evaluate', short_name(checkpoint))
    if args.rep > 1:
        path_prefix = path_prefix + '-rep' + str(args.rep)
    if args.name is not None:
        path_prefix = path_prefix + '-' + args.name
    for val_subset in val_subsets:
        if args.val_subset is not None:
            output_fname = path_prefix + '-' + str(val_subset) + '.txt'
        else:
            output_fname = path_prefix + '.txt'
        with open(output_fname, 'w') as fo:
            print("Validation subset {}".format(val_subset))
            gen = ds.generator(val_subset, loop=False)
            results = framework.validate(sess, gen, keys=keys, summary=False, show_tqdm=True)
            
            # for key, val in results.items():
            #     print(key)

            for i in range(len(results['dices'])):
                print(results['id1'][i], results['id2'][i], np.mean(results['dices'][i]), 
                    np.mean(results['jacobian_det'][i]), np.mean(results['landmark_dists'][i]), file=fo)
                
                model_name = model_args['base_network'] + '-' + str(model_args['n_cascades']) + '-liver'
                link = './evaluate/main_dataset/' + model_name 
                # link = './evaluate/main_dataset/' + model_name + '/108H'
                # if not os.path.exists(link):
                #     os.mkdir(link)
                # # link = './evaluate/main_dataset/' + model_name + '/' + results['id1'][i] + '_' + results['id2'][i]
                # link = './evaluate/main_dataset/' + model_name + '/108H/' + results['id1'][i] + '_' + results['id2'][i]
                # if not os.path.exists(link):
                #     os.mkdir(link)
                              
                im_flow = results['real_flow'][i]
                # # print(im_flow.shape)
                sitk.WriteImage(sitk.GetImageFromArray(im_flow), link + '/flow.nii.gz')
                # im_flow = RenderFlow(im_flow)
                # skimage.io.imsave(link + '/flow_' + results['id1'][i] + '_' + results['id2'][i] + '.png', im_flow)

                warped_img = results['warped_moving'][i]
                # fixed_img = results['image_fixed'][i]
                seg_fixed = results['seg1'][i]
                seg_moving = results['seg2'][i]
                img_moving = results['img2'][i]
                img_fixed = results['img1'][i]
                seg_warped = results['warped_seg_moving'][i]

                
                sitk.WriteImage(sitk.GetImageFromArray(img_moving), link + '/img_moving.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(img_fixed), link + '/img_fixed.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(seg_moving), link + '/seg_moving.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(seg_fixed), link + '/seg_fixed.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(seg_warped), link + '/seg_warped.nii.gz')
                # sitk.WriteImage(sitk.GetImageFromArray(fixed_img[:,:,:,0]), link + '/fixed.nii.gz')
                sitk.WriteImage(sitk.GetImageFromArray(warped_img[:,:,:,0]), link + '/warped.nii.gz')
                # sitk.WriteImage(sitk.GetImageFromArray(warped_img[:,:,:,0]), 'warped_ld1_130pv132pv.nii.gz')
            
            print('Summary', file=fo)
            dices = results['dices']
            landmarks = results['landmark_dists']
            # jaccs, dices, landmarks = results['jaccs'], results['dices'], results['landmark_dists']
            jacobian_det = results['jacobian_det']
            print("Dice score: {} ({})".format(np.mean(dices), np.std(
                np.mean(dices, axis=-1))), file=fo)
            # print("Jacc score: {} ({})".format(np.mean(jaccs), np.std(
            #     np.mean(jaccs, axis=-1))), file=fo)
            print("Landmark distance: {} ({})".format(np.mean(landmarks), np.std(
                np.mean(landmarks, axis=-1))), file=fo)
            print("Jacobian determinant: {} ({})".format(np.mean(
                jacobian_det), np.std(jacobian_det)), file=fo)


def RenderFlow(flow, coef = 15, channel = (0, 1, 2), thresh = 1):
    flow = flow[:, :, 64]
    im_flow = np.stack([flow[:, :, c] for c in channel], axis = -1)
    #im_flow = 0.5 + im_flow / coef
    im_flow = np.abs(im_flow)
    im_flow = np.exp(-im_flow / coef)
    im_flow = im_flow * thresh
    #im_flow = 1 - im_flow / 20
    return im_flow

def show_image(imgs, fname=None, cmap='gray', norm=False, vmin=0, vmax=1, transpose='z', origin='lower'):
    if len(imgs.shape) == 3:
        if not norm:
            if np.max(imgs) < 5:
                imgs = imgs * 255.0
            imgs = np.array(imgs, dtype=np.uint8)
    if transpose == 'z':
        if len(imgs.shape) == 3:
            imgs = np.transpose(imgs, (2, 0, 1))
        else:
            imgs = np.transpose(imgs, (2, 0, 1, 3))
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for i, ax in zip(range(0, imgs.shape[0], imgs.shape[0] // 16), axes):
        if len(imgs.shape) == 4:
            ax.imshow(imgs[i], aspect='equal', origin=origin)
        elif norm:
            ax.imshow(imgs[i], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax, aspect='equal', origin=origin)
        else:
            ax.imshow(imgs[i], cmap=plt.get_cmap(cmap), norm = matplotlib.colors.NoNorm(), aspect='equal', origin=origin)
    if fname:
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig

def short_name(checkpoint):
    cpath, steps = os.path.split(checkpoint)
    _, exp = os.path.split(cpath)
    return exp + '-' + steps


def find_checkpoint_step(checkpoint_path, target_steps=None):
    pattern = re.compile(r'model-(\d+).index')
    checkpoints = []
    for f in os.listdir(checkpoint_path):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            checkpoints.append((-steps if target_steps is None else abs(
                target_steps - steps), os.path.join(checkpoint_path, f.replace('.index', ''))))
    return min(checkpoints, key=lambda x: x[0])[1]


if __name__ == '__main__':
    main()
