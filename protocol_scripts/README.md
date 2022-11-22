## Example Usage

GaitSet on CASIA-B is taken as an example and the feature pkl is available at [Google Drive](https://drive.google.com/file/d/1wFF9RyX-xTe5PwVmJLhbC08BWs0OoW4b/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1kK9oF-9vOntvD8b9gAJX0A?pwd=uv4j).
```
python -u main.py \
--dataset CASIA-B --euc_or_cos_dist euc --part_dim 256 --part_average False \
--exclude_idt_view True --remove_no_gallery False \
--eval_feature_pkl CASIA-B_testset_GaitSet_CASIA-B_73_False_256_0.2_128_full_30-80000-eval_feature.pkl \
2>&1 | tee GaitSet_CASIA_B.log
```