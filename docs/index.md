# A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition
[[Paper]](https://ieeexplore.ieee.org/document/9928336) [[Github]](https://github.com/hshustc/TBIOM-Metric-Gait) [[Project Page]](https://hshustc.github.io/TBIOM-Metric-Gait/)

## Update
- The code has been released, and the dataset will be released as soon as possible. (2022-11-22)
- I am recently engaged in preparing the CVPR submission and taking care of my baby (^_^). After the CVPR deadline (2022-11-18), I will clean up and release the code in conjunction with OpenGait ([https://github.com/ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait)). The MHG dataset as well as the application form will be released simultaneously. (2022-10-20)
- **This paper receives a timely review and is accepted by IEEE Transactions on Biometrics, Identity and Behavior. We sincerely thank the editors and reviewers for your time and efforts spent on our manuscript**. (2022-10-20)


## Abstract

Recently the methods based on silhouettes achieve significant improvement for gait recognition. The performance, e.g., 96.4% on the largest OUMVLP, indicates that a promising gait system is around the corner. However, we argue that the observation is not true. Firstly, we find that there exists a non-negligible gap of gait evaluation between academic research and practical applications. To validate the assumption, we conduct a comprehensive study on the evaluation for silhouette-based gait recognition and provide new insights into the limitations of the current methods. Our key findings include: (a) The current evaluation protocol is excessively simplified and ignores a lot of hard cases. (b) The current methods are sensitive to the noise caused by rotation and occlusion. Secondly, we observe that the data scarcity largely hinders the development of gait recognition and some crucial covariates (e.g., camera heights) are not thoroughly investigated. To address the issue, we propose a new dataset called Multi-Height Gait (MHG). It collects 200 subjects of normal walking, walking with bags and walking in different clothes. Particularly, it collects the sequences recorded by the cameras at different heights. We hope this work would inspire more advanced research for gait recognition. The project page is available at [https://hshustc.github.io/TBIOM-Metric-Gait/](https://hshustc.github.io/TBIOM-Metric-Gait/).

## Experiments
###The performance comparison using different settings for evaluation protocol and silhouette noise
![](comparison.png)


## Citation
Please cite the following paper if you find this useful in your research:
```
@Article{hou2022gait,
  Title                    = {A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition},
  Author                   = {Hou, Saihui and Fan, Chao and Cao, Chunshui and Liu, Xu and Huang, Yongzhen},
  Journal                  = {IEEE Transactions on Biometrics, Identity and Behavior},
  Year                     = {2022}
}
```

## Contact
This page is maintained by [Saihui Hou](https://hshustc.github.io/). If you have any question, please send the email to [hshvim@live.com](mailto:hshvim@live.com).



