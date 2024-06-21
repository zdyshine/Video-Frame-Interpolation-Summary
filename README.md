# Video-Frame-Interpolation-Summary
Video Frame Interpolation Summary 2024    
[Summary2023](https://github.com/zdyshine/Video-Frame-Interpolation-Summary/blob/main/2023_before.md)

#### update
2024-01-11：新增结合ffmpeg的视频推理代码[basic_infer](https://github.com/zdyshine/Video-Frame-Interpolation-Summary/tree/main/basic_infer)    
如有其他想测试模型或视频，可以在issue中联系    
2024-01-11：新增结合推理演示视频(1080P) https://space.bilibili.com/350913028    

# 参考网站
1.https://paperswithcode.com/sota/video-frame-interpolation-on-ucf101-1     
2.https://paperswithcode.com/sota/video-frame-interpolation-on-vimeo90k     
3.https://paperswithcode.com/sota/video-frame-interpolation-on-x4k1000fps    
4.https://github.com/AIVFI/Video-Frame-Interpolation-Rankings-and-Video-Deblurring-Rankings     
5.https://sniklaus.com/vfibench(Benchmarking Video Frame Interpolation)    

# List
数据集：Vimeo90K:
| index | method  | paper | code | PSNR | SSIM | Algorithm | Traindata | Arbitrary |
| :----:| :---- | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 1 | MA-VFI(Arxiv 2402) | [paper](https://arxiv.org/pdf/2402.02892.pdf) | [code](None) | 35.96 | 0.980 | Motion-Aware  | Vimeo90K | True |
| 2 | VIDIM (Arxiv 2404) | [paper](https://arxiv.org/pdf/2404.01203.pdf) | [code](https://vidim-interpolation.github.io/) | - | - | generate video  | mixture: WebVid + . | True |
| 3 | LADDER(Arxiv 2404) | [paper](https://arxiv.org/pdf/2404.11108.pdf) | [code](None) | 36.65 | 0.981 | 光流,更高效  | Vimeo90K | False/结构可改 |
| 4 | SGM-VFI(Arxiv 2404) | [paper](https://arxiv.org/pdf/2404.06913.pdf) | [code](https://github.com/MCG-NJU/SGM-VFI) | - | - |  large motion | Vimeo90K/X4K1000FPS | False/结构可改 |
| 5 | MADIFF(Arxiv 2404) | [paper](https://arxiv.org/pdf/2404.13534.pdf) | [code](None) | - | - |  a novel diffusion framework  | Vimeo90K |- |
| 6 | PerVFI(Arxiv 2404) | [paper](https://arxiv.org/pdf/2404.06692.pdf) | [code](https://github.com/mulns/PerVFI) | - | - |  降低模糊和伪影  | Vimeo90K |- |
| 7 | ConsecutiveBrownianBridge(Arxiv 2405) | [paper](https://arxiv.org/pdf/2405.05953) | [code](https://github.com/ZonglinL/ConsecutiveBrownianBridge) | - | - |  Brownian Bridge  | Vimeo90K |True |
| 8 | ToonCrafter(Arxiv 2405) | [paper](https://arxiv.org/abs/2405.17933) | [code](https://github.com/ToonCrafter/ToonCrafter) | - | - |  动漫插帧  | - |True |
| 9 | ZeroSmooth(Arxiv 2406) | [paper](https://arxiv.org/pdf/2406.00908) | [code](https://github.com/ssyang2020/ZeroSmooth) | - | - |  Training-free Diffuser  | - |True |
| 10 | Swin-VFI(Arxiv 2406) | [paper](https://arxiv.org/pdf/2406.11371) | [code](None) | 38.04 | 0.981 |  Swin-Transverter的多阶段多尺度网络  | Vimeo90K | False |
| 11 | LAVIB(Arxiv 2406) | [paper](https://arxiv.org/pdf/2406.09754) | [code](https://github.com/alexandrosstergiou/LAVIB) | - | - |  提出一个新的数据集  | - | - |

 
# Dataset
UCF101: Download UCF101 dataset    
Vimeo90K: Download Vimeo90K dataset    
MiddleBury: Download MiddleBury OTHER dataset    
HD: Download HD dataset    
# 致谢
感谢各位大佬的开源，希望大佬们论文多多，工作顺利。如有侵犯权益，可联系我进行修改和删除。    
