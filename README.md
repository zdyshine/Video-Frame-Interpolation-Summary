# Video-Frame-Interpolation-Summary
Video Frame Interpolation Summary 2020~2021

# 增加功能  
1. 结果保存为无损yuv格式视频  


# 说明
1.代码运行环境：阿里云 V100 16GB，主要考虑推理时间，性能指标及显存占用。    
2.推理代码基于：https://github.com/hzwer/arXiv2020-RIFE 欢迎大家去源码star。    
3.上面代码只给出了推理代码，model等文件，可去对应源码获取。    
4.后续有机会会继续更新，如有误，可联系我进行修正。  
# 性能
数据集：UCF101:  image size:256x256，image numbers: 379，主要考虑推理时间(ms)，性能指标及显存占用。    
![image](https://github.com/zdyshine/Video-Frame-Interpolation-Summary/blob/main/UCF101.png)    
1080p的的视频片段，共625帧，推理时间为整个程序的运行时间。     
![image](https://github.com/zdyshine/Video-Frame-Interpolation-Summary/blob/main/1080p_video.png)    

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
# Dataset
UCF101: Download UCF101 dataset    
Vimeo90K: Download Vimeo90K dataset    
MiddleBury: Download MiddleBury OTHER dataset    
HD: Download HD dataset    
# 致谢
感谢各位大佬的开源，希望大佬们论文多多，工作顺利。如有侵犯权益，可联系我进行修改和删除。    
