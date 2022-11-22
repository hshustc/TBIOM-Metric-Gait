import os
import os.path as osp
import pickle
import argparse
from evaluator import evaluation
from print_metric import print_metric

def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str, help='gpu id')
parser.add_argument('--dataset', default='CASIA-B', type=str, help='dataset name')
parser.add_argument('--eval_feature_pkl', default='', type=str, help='path to eval feature pkl')
parser.add_argument('--euc_or_cos_dist', default='euc', choices=['cos', 'euc'], type=str, help='euclidean or cosine distance')
parser.add_argument('--part_dim', default=-1, type=int, help='part dimension')
parser.add_argument('--part_average', default=False, type=boolean_string, help='part average')
parser.add_argument('--exclude_idt_view', default=True, type=boolean_string, help='excluding identical-view cases')
parser.add_argument('--remove_no_gallery', default=False, type=boolean_string, help='remove probe having no gallery')
args = parser.parse_args()
print('#######################################')
print("Args:", args)
print('#######################################')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.remove_no_gallery:
    print("The Seqs that have NO GALLERY are *REMOVED*.")
else:
    print("The Seqs that have NO GALLERY are *INCLUDED*.")
print('#######################################')
assert(osp.exists(args.eval_feature_pkl))
print("{} *EXISTS*".format(args.eval_feature_pkl))
eval_data = pickle.load(open(args.eval_feature_pkl, 'rb'))
print('#######################################')


print("******************************************************PROTOCOL_0******************************************************")
# protocol_0
probe_type_dict = {'CASIA_B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                  'OUMVLP': [['00']],
                  'MHG':[['M_nm_01', 'M_nm_02','L_nm_01', 'L_nm_02'], \
                            ['M_bg_01', 'M_bg_02','L_bg_01', 'L_bg_02'], \
                                ['M_cl_01', 'M_cl_02','L_cl_01', 'L_cl_02']]
                }
gallery_type_dict = {'CASIA_B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                    'OUMVLP': [['01']],
                    'MHG':[['H_nm_01', 'H_nm_02']]
                }
ACC, mAP = evaluation(args.dataset.replace('-', '_'), eval_data, probe_type_dict, gallery_type_dict, args.euc_or_cos_dist, 
                args.part_dim, args.part_average, args.exclude_idt_view, args.remove_no_gallery, cross_view_gallery=False)
print_metric(ACC, metric_name='ACC@Rank1')
print_metric(mAP, metric_name='mAP')
print("******************************************************PROTOCOL_0******************************************************")


print("******************************************************PROTOCOL_1******************************************************")
# protocol_2
probe_type_dict = {'CASIA_B': [['nm-01'], ['bg-01'], ['cl-01']],
                  'OUMVLP': [['00']],
                  'MHG':[['M_nm_01', 'M_nm_02','L_nm_01', 'L_nm_02'], \
                            ['M_bg_01', 'M_bg_02','L_bg_01', 'L_bg_02'], \
                                ['M_cl_01', 'M_cl_02','L_cl_01', 'L_cl_02']]
                }
gallery_type_dict = {'CASIA_B': [['nm-02', 'bg-02', 'cl-02']],
                    'OUMVLP': [['01']],
                    'MHG':[['H_nm_01', 'H_nm_02', 'H_bg_01', 'H_bg_02', 'H_cl_01', 'H_cl_02']]
                }
ACC, mAP = evaluation(args.dataset.replace('-', '_'), eval_data, probe_type_dict, gallery_type_dict, args.euc_or_cos_dist, 
                args.part_dim, args.part_average, args.exclude_idt_view, args.remove_no_gallery, cross_view_gallery=True)
print_metric(ACC, metric_name='ACC@Rank1')
print_metric(mAP, metric_name='mAP')
print("******************************************************PROTOCOL_1******************************************************")
