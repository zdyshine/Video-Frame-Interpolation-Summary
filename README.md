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
数据集：UCF101:  image size:256x256，image numbers: 379，主要考虑推理时间(ms)，性能指标及显存占用。 
| index | method  | infer time | memory | PSNR | SSIM | Algorithm | Traindata |
| :----:| :---- | :----: | :----: | :----: | :----: | :----: | :----: |
| 1 | EMA-VFI(CVPR 2023) | - | - | 35.48 | 0.9701 | 混合CNN和Transformer架构 | Vimeo90K |
| 2 | DQBC(IJCAI 2023) | - | - | 35.44 | 0.9700 | 基于CNN的SynthNet合成 | Vimeo90K |
| 3 | AMT(CVPR 2023) | - | - | 35.45 | 0.9700 | 混合CNN和Transformer架构 | Vimeo90K |
| 4 | VFIformer(CVPR 2022) | - | - | 35.43 | 0.9700 | Transformer架构 | Vimeo90K |
| 5 | UPR-Net (CVPR 2023) | - | - | 35.47 | 0.9700 | 光流-轻量-指标高 | Vimeo90K(51312 triplets) |
| 6 | BiFormer (CVPR 2023) | - | - | - | - | 双向Transformer-4K帧插 | X4K1000FPS |
| 7 | IFRNet (CVPR 2022) | - | - | 35.42 | 0.9698 | conv-轻量 | Vimeo90K |
| 8 | LDMVFI (arXiv 2023) | - | - | 32.186 | - | 扩散模型 | Vimeo90k(64612 frame)+BVI-DVC(17600 frame) |
| 9 | MA-GCSPA (arXiv 2023) | - | - | 35.43 | - | conv | Vimeo90k |

Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation
# 论文及源码 --- 2023年7月新统计
###  1. <a name='2023年7月更新'></a>Video-Frame-Interpolation
#### 1.1 
* EMA-VFI(CVPR 2023)：
  * paper：https://arxiv.org/pdf/2303.00440v2.pdf  
  * code：https://github.com/mcg-nju/ema-vfi  
  * 简介：提出了一个新的模块，通过统一操作明确地提取运动和外观信息。具体而言，重新思考了帧间注意力的信息处理过程，并将其注意力图用于外观特征增强和运动信息提取。此外，为了实现高效的VFI,提出的模块可以无缝地集成到混合CNN和Transformer架构中。这种混合流水线可以减轻帧间注意力的计算复杂性，同时保留详细的低级结构信息。

* DQBC (IJCAI 2023)：
  * paper：https://arxiv.org/pdf/2304.13596.pdf
  * code：https://github.com/kinoud/DQBC
  * 简介：提出了密集查询双边相关性(DQBC),它消除了感受野依赖问题，因此更适合小而快速移动的对象。使用DQBC生成的运动场通过上下文特征进一步细化和上采样。在固定运动场之后，一个基于CNN的SynthNet合成最终插值帧。

* AMT (CVPR 2023)：
  * paper：https://arxiv.org/pdf/2304.09790.pdf 
  * code：https://github.com/mcg-nku/amt
  * 简介：提出了一种新的视频帧插值网络架构——All-Pairs Multi-Field Transforms (AMT)。该架构基于两个关键设计。首先，为所有像素对构建双向相关体积，并使用预测的双边流来检索相关性以更新流和插值内容特征。其次，从一组更新的粗粒流中推导出多个细粒度的流动场，以便分别对输入帧执行向后扭曲操作。

* VFIformer (CVPR 2022)：
  * paper：https://arxiv.org/pdf/2205.07230.pdf 
  * code：https://github.com/dvlab-research/VFIformer
  * 简介：出了一种新颖的框架，该框架利用Transformer模型来建模视频帧之间的长距离像素相关性。此外，网络还配备了一种新的跨尺度窗口式注意力机制，其中跨尺度窗口相互作用。这种设计有效地扩大了感受野并聚合了多尺度信息。
    
* UPR-Net (CVPR 2023)：
  * paper：https://arxiv.org/pdf/2211.03456.pdf 
  * code：https://github.com/srcn-ivl/upr-net
  * 简介：提出了一种新颖的统一金字塔循环网络(UPR-Net),用于帧插值。在灵活的金字塔框架中，UPR-Net利用轻量级的循环模块进行双向流估计和中间帧合成。在每个金字塔层中，它利用估计的双向流生成前向变形表示以进行帧合成；在整个金字塔层次中，它使光流和中间帧的迭代细化成为可能。特别是，我们证明我们的迭代合成策略可以显著提高大运动情况下帧插值的鲁棒性。尽管我们的基础版本UPR-Net非常轻量级(仅1.7M参数),但它在各种基准测试上均取得了出色的性能。

* BiFormer (CVPR 2023)：
  * paper：https://arxiv.org/pdf/2304.02225.pdf 
  * code：https://github.com/junheum/biformer
  * 简介：提出了一种基于双向Transformer(BiFormer)的新颖4K视频帧插值器，它执行三个步骤：全局运动估计、局部运动细化和帧合成。首先，在全局运动估计中，预测对称双边运动场的粗尺度。为此，提出了第一个基于Transformer的双边运动估计器——BiFormer。其次，使用块级双边成本体积(BBCVs)高效地细化全局运动场。最后，使用细化的运动场对输入帧进行扭曲并将它们混合以合成中间帧

* IFRNet (CVPR 2022)：
  * paper：https://openaccess.thecvf.com/content/CVPR2022/papers/Kong_IFRNet_Intermediate_Feature_Refine_Network_for_Efficient_Frame_Interpolation_CVPR_2022_paper.pdf
  * code：https://github.com/ltkong218/IFRNet
  * 简介：将分离的光流估计和上下文特征细化合并到一个单一的编码器-解码器基础的IFRNet中，以实现紧凑性和快速推理，使这两个关键元素能够相互受益。

* LDMVFI (Arxiv 2023)：
  * paper：https://arxiv.org/pdf/2303.09508.pdf
  * code：https://github.com/danier97/LDMVFI
  * 简介：VFI问题表述为条件生成问题，从生成的角度来处理VFI问题。作为第一次使用潜在扩散模型解决VFI问题。

* MA-GCSPA (CVPR 2023)：
  * paper：https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Exploring_Motion_Ambiguity_and_Alignment_for_High-Quality_Video_Frame_Interpolation_CVPR_2023_paper.pdf
  * code：https://github.com/redrock303/CVPR23-MA-GCSPA
  * 简介：提出放松对重建一个尽可能接近GT的中间帧的要求。基于假设，即插值内容应与给定帧中的对应部分保持相似的结构，开发了一种纹理一致性损失(TCL),设计了一个简单、高效且强大的指导跨尺度金字塔对齐(GCSPA)模块，其中充分利用了多尺度信息。


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
