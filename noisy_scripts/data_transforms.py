# encoding: utf-8
import os
import os.path as osp
import shutil
import math
import random
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

def array2img(x):
    return (x*255.0).astype('uint8')

def img2array(x):
    return x.astype('float32')/255.0

def save_seq(seq, seq_dir):
    if osp.exists(seq_dir):
        shutil.rmtree(seq_dir)
    if not osp.exists(seq_dir):
        os.makedirs(seq_dir)
    for i in range(seq.shape[0]):
        save_name = osp.join(seq_dir, '{:0>3d}.png'.format(i))
        cv2.imwrite(save_name, array2img(seq[i, :, :]))

def merge_seq(seq, row=6, col=6):
    frames_index = np.arange(seq.shape[0])
    im_h = seq.shape[1]
    im_w = seq.shape[2]
    num_per_im = row*col
    if len(frames_index) < num_per_im:
        selected_frames_index = sorted(np.random.choice(frames_index, num_per_im, replace=True))
    else:
        selected_frames_index = sorted(np.random.choice(frames_index, num_per_im, replace=False))
    im_merged = np.zeros((im_h*row, im_w*col))
    for i in range(len(selected_frames_index)):
        im = seq[selected_frames_index[i], :, :]
        y = int(i/col)
        x = i%col
        im_merged[y*im_h:(y+1)*im_h, x*im_w:(x+1)*im_w] = im
    im_merged = array2img(im_merged)
    return im_merged

def pad_seq(seq, pad_size):
    return np.pad(seq, ([0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]]), mode='constant')

def cut_img(img, T_H, T_W):
    # print("before cut_img: ", img.shape, np.min(img), np.max(img), T_H, T_W, img.dtype)
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right].astype('uint8')
    # print("after cut_img: ", img.shape, np.min(img), np.max(img), T_H, T_W, img.dtype)
    return img

class RandomErasing(object):
    def __init__(self, prob=0.5, sl=0.02, sh=0.1, r1=0.3, per_frame=False):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                for attempt in range(100):
                    seq_size = seq.shape
                    area = seq_size[1] * seq_size[2]
        
                    target_area = random.uniform(self.sl, self.sh) * area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)
        
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
        
                    if w < seq_size[2] and h < seq_size[1]:
                        x1 = random.randint(0, seq_size[1] - h)
                        y1 = random.randint(0, seq_size[2] - w)
                        seq[:, x1:x1+h, y1:y1+w] = 0.
                        return seq
            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)

class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree
    
    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            img_h = seq.shape[1]
            img_w = seq.shape[2]
            angle = random.uniform(-self.degree, self.degree)
            seq = array2img(seq)
            seq = [Image.fromarray(seq[tmp, :, :], mode='L').rotate(angle) for tmp in range(seq.shape[0])]
            seq = [cut_img(np.asarray(tmp), img_h, img_w) for tmp in seq]
            return img2array(np.stack(seq))

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[:, :, ::-1]

class RandomPadCrop(object):
    def __init__(self, prob=0.5, pad_size=(4, 0), per_frame=False):
        self.prob = prob
        self.pad_size = pad_size
        self.per_frame = per_frame
    
    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                _, dh, dw = seq.shape
                seq = pad_seq(seq, self.pad_size)
                _, sh, sw = seq.shape
                bh, lw, th, rw = self.get_params((sh, sw), (dh, dw))
                seq = seq[:, bh:th, lw:rw]
                seq = array2img(seq)
                seq = [cut_img(seq[tmp, :, :], dh, dw) for tmp in range(seq.shape[0])]
                return img2array(np.stack(seq))
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, axis=0)

    def get_params(self, src_size, dst_size):
        sh, sw = src_size
        dh, dw = dst_size
        if sh == dh and sw == dw:
            return 0, 0, dh, dw

        i = random.randint(0, sh - dh)
        j = random.randint(0, sw - dw)
        return i, j, i+dh, j+dw

def build_data_transforms(random_erasing=False, random_rotate=False, \
        random_horizontal_flip=False, random_pad_crop=False, resolution=64, random_seed=2019):
    np.random.seed(random_seed)
    random.seed(random_seed)
    print("random_seed={} for build_data_transforms".format(random_seed))

    object_list = []
    if random_pad_crop:
        object_list.append(RandomPadCrop(prob=0.5, pad_size=(8*int(resolution/64), 0), per_frame=False))
    if random_rotate:
        object_list.append(RandomRotate(prob=0.5, degree=10))
    if random_erasing:
        object_list.append(RandomErasing(prob=0.5, sl=0.02, sh=0.05, r1=0.3, per_frame=False))
    if random_horizontal_flip:
        object_list.append(RandomHorizontalFlip(prob=0.5))

    transform = T.Compose(object_list)
    return transform

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt
    SEED = 2020
    np.random.seed(SEED)
    random.seed(SEED)
    
    merge_imgs = {}
    
    example_pkl = './example_data/015.pkl'
    seq_in = pickle.load(open(example_pkl, 'rb'))
    resolution = seq_in.shape[1]
    cut_padding = 10*int(resolution/64)
    seq_in = seq_in[:, :, cut_padding:-cut_padding]
    seq_in = img2array(seq_in)
    save_seq(seq_in, seq_dir='./example_data/raw_seq')
    merge_imgs.update({'raw':merge_seq(seq_in)})
    print(seq_in.shape, np.min(seq_in), np.max(seq_in), seq_in.dtype)

    transform = build_data_transforms(random_pad_crop=True, resolution=resolution)
    seq_out = transform(seq_in.copy())
    save_seq(seq_out, seq_dir='./example_data/pad_crop_seq')
    seq_merge = merge_seq(seq_out)
    merge_imgs.update({'pad_crop':merge_seq(seq_out)})
    print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

    transform = build_data_transforms(random_rotate=True)
    seq_out = transform(seq_in.copy())
    save_seq(seq_out, seq_dir='./example_data/rotate_seq')
    seq_merge = merge_seq(seq_out)
    merge_imgs.update({'rotate':merge_seq(seq_out)})
    print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

    transform = build_data_transforms(random_erasing=True)
    seq_out = transform(seq_in.copy())
    save_seq(seq_out, seq_dir='./example_data/erasing_seq')
    seq_merge = merge_seq(seq_out)
    merge_imgs.update({'erasing':merge_seq(seq_out)})
    print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

    transform = build_data_transforms(random_horizontal_flip=True)
    seq_out = transform(seq_in.copy())
    save_seq(seq_out, seq_dir='./example_data/horizontal_flip_seq')
    seq_merge = merge_seq(seq_out)
    merge_imgs.update({'horizontal_flip':merge_seq(seq_out)})
    print(seq_out.shape, np.min(seq_out), np.max(seq_out), seq_out.dtype)

    rows = 1
    columns = len(merge_imgs)
    fig = plt.figure()
    merge_imgs_keys = list(merge_imgs.keys())
    for i in range(1, rows*columns+1):
        ax = fig.add_subplot(rows, columns, i)
        key = merge_imgs_keys[i-1]
        ax.set_title(key)
        plt.imshow(merge_imgs[key], cmap = plt.get_cmap('gray'))
    plt.show()





