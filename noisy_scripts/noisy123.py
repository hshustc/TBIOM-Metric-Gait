# encoding: utf-8
import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
from data_transforms import *

SEED = 2020
np.random.seed(SEED)
random.seed(SEED)

src_root_path = '/data_3/housaihui/Dataset/'
des_root_path = '/data_2/housaihui/done_metric/noisy123data/'

dataset_list = ['casia_b/silhouettes_cut_pkl', 'casia_b/silhouettes_cut128_pkl', 
                'OUMVLP/silhouettes_cut_pkl', 'OUMVLP/silhouettes_cut128_pkl', 
                'MHG/silhouettes_cut_pkl', 'MHG/silhouettes_cut128_pkl']

for dataset in dataset_list:
    src_root_dir = osp.join(src_root_path, dataset)
    des_root_dir = osp.join(des_root_path, dataset)
    if '128' in dataset:
        resolution = 128
    else:
        resolution = 64
    print('dataset={}, resolution={}'.format(dataset, resolution))
    data_transform = build_data_transforms(random_rotate=True, random_erasing=True, random_pad_crop=True, \
                                        resolution=resolution, random_seed=SEED)

    seq_cnt = 0
    id_list = sorted(os.listdir(src_root_dir))
    for _id in id_list:
        id_dir = os.path.join(src_root_dir, _id)
        type_list = sorted(os.listdir(id_dir))
        for _type in type_list:
            type_dir = os.path.join(src_root_dir, _id, _type)
            view_list = sorted(os.listdir(type_dir))
            for _view in view_list:
                seq_cnt = seq_cnt + 1
                view_dir = os.path.join(src_root_dir, _id, _type, _view)
                src_pkl = os.path.join(view_dir, '{}.pkl'.format(_view))
                src_seq = pickle.load(open(src_pkl, 'rb'))
                if seq_cnt % 1000 == 0:
                    print(src_pkl, src_seq.shape, np.min(src_seq), np.max(src_seq))

                des_view_dir = view_dir.replace(src_root_dir, des_root_dir)
                if not osp.exists(des_view_dir):
                    os.makedirs(des_view_dir)
                des_pkl = src_pkl.replace(src_root_dir, des_root_dir)
                des_seq = array2img(data_transform(img2array(src_seq)))
                if seq_cnt % 1000 == 0:
                    print(src_pkl, src_seq.shape, np.min(src_seq), np.max(src_seq))

                # merge_imgs = {}
                # merge_imgs.update({'src':merge_seq(img2array(src_seq))})
                # merge_imgs.update({'des':merge_seq(img2array(des_seq))})
                # rows = 1
                # columns = len(merge_imgs)
                # fig = plt.figure()
                # merge_imgs_keys = list(merge_imgs.keys())
                # for i in range(1, rows*columns+1):
                #     ax = fig.add_subplot(rows, columns, i)
                #     key = merge_imgs_keys[i-1]
                #     ax.set_title(key)
                #     plt.imshow(merge_imgs[key], cmap = plt.get_cmap('gray'))
                # plt.show()

                with open(des_pkl, 'wb') as f:
                    pickle.dump(des_seq, f)

