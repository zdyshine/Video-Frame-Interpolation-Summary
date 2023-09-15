# Video-Frame-Interpolation-Summary
Video Frame Interpolation Summary 2020~2023
# 视频结果
https://space.bilibili.com/350913028/channel/seriesdetail?sid=409673      
（仅用于论文的对比效果展示）    
# 增加功能  
1. 结果保存为无损yuv格式视频：inferencre_video_yuv.py  
2. 增加分离声音和合成声音脚本（to do）    

# 说明
1.代码运行环境：阿里云 V100 16GB，主要考虑推理时间，性能指标及显存占用。    
2.推理代码基于：https://github.com/hzwer/arXiv2020-RIFE 欢迎大家去源码star。    
3.上面代码只给出了推理代码，model等文件，可去对应源码获取。    
4.后续有机会会继续更新，如有误，可联系我进行修正。  

# 参考网站
1.https://paperswithcode.com/sota/video-frame-interpolation-on-ucf101-1。    
2.https://paperswithcode.com/sota/video-frame-interpolation-on-vimeo90k。 

# 性能（2倍插帧）--- 2023年7月新增
数据集：Vimeo90K:
| index | method  | paper | code | PSNR | SSIM | Algorithm | Traindata |
| :----:| :---- | :----: | :----: | :----: | :----: | :----: | :----: |
| 1 | EMA-VFI(CVPR 2023) | [paper](https://arxiv.org/pdf/2303.00440v2.pdf) | [code](https://github.com/mcg-nju/ema-vfi ) | 35.48 | 0.9701 | 混合CNN和Transformer架构 | Vimeo90K |
| 2 | DQBC(IJCAI 2023) | [paper](https://arxiv.org/pdf/2304.13596.pdf) | [code](https://github.com/kinoud/DQBC) | 35.44 | 0.9700 | 基于CNN的SynthNet合成 | Vimeo90K |
| 3 | AMT(CVPR 2023) | [paper](https://arxiv.org/pdf/2304.09790.pdf) | [code](https://github.com/mcg-nku/amt) | 35.45 | 0.9700 | 混合CNN和Transformer架构 | Vimeo90K |
| 4 | VFIformer(CVPR 2022) | [paper](https://arxiv.org/pdf/2205.07230.pdf) | [code](https://github.com/dvlab-research/VFIformer) | 35.43 | 0.9700 | Transformer架构 | Vimeo90K |
| 5 | UPR-Net (CVPR 2023) | [paper](https://arxiv.org/pdf/2211.03456.pdf) | [code](https://github.com/srcn-ivl/upr-net) | 35.47 | 0.9700 | 光流-轻量-指标高 | Vimeo90K(51312 triplets) |
| 6 | BiFormer (CVPR 2023) |[paper](https://arxiv.org/pdf/2304.02225.pdf) | [code](https://github.com/junheum/biformer) | - | - | 双向Transformer-4K帧插 | X4K1000FPS |
| 7 | IFRNet (CVPR 2022) | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kong_IFRNet_Intermediate_Feature_Refine_Network_for_Efficient_Frame_Interpolation_CVPR_2022_paper.pdf) | [code](https://github.com/ltkong218/IFRNet) | 35.42 | 0.9698 | conv-轻量 | Vimeo90K |
| 8 | LDMVFI (arXiv 2023-03) | [paper](https://arxiv.org/pdf/2303.09508.pdf) | [code](https://github.com/danier97/LDMVFI) | 32.186 | - | 扩散模型 | Vimeo90k(64612 frame)+BVI-DVC(17600 frame) |
| 9 | MA-GCSPA (arXiv 2023) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Exploring_Motion_Ambiguity_and_Alignment_for_High-Quality_Video_Frame_Interpolation_CVPR_2023_paper.pdf) | [code](https://github.com/redrock303/CVPR23-MA-GCSPA) | 35.43 | - | conv | Vimeo90k |
| 10 | VFI_Adapter (arXiv 2023-06) | [paper](https://arxiv.org/pdf/2306.13933.pdf) | [code](https://github.com/haoningwu3639/VFI_Adapter)| - | - | 提高VFI性能 | Vimeo90k |
| 11 | FILM (ECCV 2022) | [paper](https://arxiv.org/pdf/2202.04901.pdf) | [code](https://github.com/google-research/frame-interpolation) | 35.87 | 0.968 | 大场景运动 | Vimeo90k |
| 12 |  (CVPR2023) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Plack_Frame_Interpolation_Transformer_and_Uncertainty_Guidance_CVPR_2023_paper.pdf) | None | 36.34 | 0.9814 | a novel transformer-based | Vimeo90k |
| 13 |  (CVPR2023) | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Range-Nullspace_Video_Frame_Interpolation_With_Focalized_Motion_Estimation_CVPR_2023_paper.html) | None | 36.33 | 0.975 | a novel frame renderer | Vimeo90k |
| 14 |  VFIFT (Arxiv 2023-07) | [paper](https://arxiv.org/pdf/2307.16144.pdf) | None | 36.43 | 0.9813 | Flow Transformer | Vimeo90k |
| 15 |  WaveletVFI (IEEE TIP) | [paper](https://arxiv.org/pdf/2309.03508.pdf) | [code](https://github.com/ltkong218/WaveletVFI) | 35.58 | 0.978 | WaveletVFI | Vimeo90k |

# 性能（2倍插帧）
数据集：UCF101:  image size:256x256，image numbers: 379，主要考虑推理时间(ms)，性能指标及显存占用。    
| index | method  | infer time | memory | PSNR | SSIM |
| :----:| :---- | :----: | :----: | :----: | :----: |
| 1 | DAIN(CVPR2019) | ~0.1736s | - | - | - |
| 2 | EQVI(AIM2020) | ~0.669s | - | - | - |
| 3 | RIFE(arXiv2020) | ~1.538s | 1348MiB | 35.243 | 0.96833 |
| 4 | CAIN(AAAI2021) | ~0.1884s | 1996MiB | 34.9580 | 0.96794 |
| 5 | FLAVR(CVPR2021) | ~0.0897s | 2594MiB | 34.970 | 0.96802 |
| 6 | RRIN(ICASSP 2020) | ~0.2931s | 2656MiB | 32.6678 | 0.966584 |
| 7 | AdaCoF(CVPR 2020) | ~0.2055s| 1948MiB | 35.165 | 0.96797 |
| 8 | CDFI(CVPR2021) | 14482M | 3638MiB | 35.208 | 0.96739 |
| 9 | EDSC(CVPR2021) | 1257M | 1832MiB | 35.168 | 0.96793 |


1080p的的视频片段，共625帧，推理时间为整个程序的运行时间。     
| index | method | memory | time | machine |
| :----:| :---- | :----: | :----: | :----: |
| 1 | DAIN | - | - | - |
| 2 | EQVI | - | - | - |
| 3 | RIFE | 6042M | 37s | V100 |
| 4 | CAIN | 4126M | 73s | V100 |
| 5 | FLAVR | 11638M | 2107s | V100 |
| 6 | RRIN | 8832M | 166s | V100 |
| 7 | AdaCoF | 15280M | 77s | V100 |
| 8 | CDFI | 14482M | 508s | V100 |
| 9 | EDSC | 1257M | 97s | V100 |
| 10 | BMBC | 19887M | ~78min | 3090 |      
| 11 | AMBE | 16247M | ~900s | 3090 |      
   
# 源码
1.DAIN (Depth-Aware Video Frame Interpolation)    
   paper:https://arxiv.org/pdf/1904.00830.pdf    
   github:https://github.com/baowenbo/DAIN    
   使用深度感知的流投影层来估计作为双向流加权组合的中间流。    
2.EQVI(Enhanced Quadratic Video Interpolation)     
   paper:https://arxiv.org/pdf/2009.04642.pdf    
   github: https://github.com/lyh-18/EQVI    
3.RIFE v2.4 - Real Time Video Interpolation      
   paper:https://arxiv.org/pdf/2011.06294.pdf    
   github:https://github.com/hzwer/arXiv2020-RIFE    
4.CAIN(Channel Attention Is All You Need for Video Frame Interpolation)    
   paper:https://aaai.org/ojs/index.php/AAAI/article/view/6693/6547    
   github: https://github.com/myungsub/CAIN    
  一种高效的无光流估计方法，使用PixelShuffle算子和channel attention来隐式捕捉运动信息。    
5.FLAVR （Flow-Agnostic Video Representations for Fast Frame Interpolation (不依赖光流)    
   paper:https://arxiv.org/pdf/2012.08512.pdf    
   github:https://github.com/tarun005/FLAVR    
6.RRIN(Video Frame Interpolation via Residue Refinement) (不依赖光流)    
   paper:   https://ieeexplore.ieee.org/document/9053987/    
   github:https://github.com/HopLee6/RRIN    
7.AdaCoF(Adaptive Collaboration of Flows for Video Frame Interpolation)    
   paper:https://arxiv.org/pdf/1907.10244.pdf    
   github:https://github.com/HyeongminLEE/AdaCoF-pytorch    
8.CDFI(Compression-Driven Network Design for Frame Interpolation)    
   paper:https://arxiv.org/pdf/2103.10559.pdf    
   github:https://github.com/tding1/CDFI    
9.EDSC( Multiple Video Frame Interpolation via Enhanced Deformable Separable Convolution)    
   paper:https://arxiv.org/pdf/2006.08070.pdf    
   github:https://github.com/Xianhang/EDSC-pytorch    
    提出DSepConv[6]利用可变形可分卷积扩展基于核的方法，并进一步提出EDSC执行多次插补。    
10.UTI-VFI(Video Frame Interpolation without Temporal Priors)    
   paper:https://github.com/yjzhang96/UTI-VFI/raw/master/paper/nips_camera_ready.pdf    
   github:https://github.com/yjzhang96/UTI-VFI   
11.BMBC(Bilateral Motion Estimation with Bilateral Cost Volume for Video Interpolation)    
   paper:https://arxiv.org/abs/2007.12622    
   github:https://github.com/JunHeum/BMBC   
12.ABME(Asymmetric Bilateral Motion Estimation for Video Frame Interpolation)    
   paper:https://arxiv.org/abs/2108.06815    
   github:https://github.com/JunHeum/ABME  
# Dataset
UCF101: Download UCF101 dataset    
Vimeo90K: Download Vimeo90K dataset    
MiddleBury: Download MiddleBury OTHER dataset    
HD: Download HD dataset    
# 致谢
感谢各位大佬的开源，希望大佬们论文多多，工作顺利。如有侵犯权益，可联系我进行修改和删除。    
