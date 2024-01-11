# basic_infer

## 快速集成开源模型并进行统一标准的测试

#### 使用方式
通用推理:   
inference_vfi.py中 def make_inference(I0, I1, embt, exp) 不同，导致需要对特定函数进行额外定义  
~~~
CUDA_VISIBLE_DEVICES=0 python inference_animesr_vfi.py -i ./casesc.mp4 -o ./results -n VFI -s 1 --expname IFRNet --num_process_per_gpu 1 --suffix 50fpsv1_half --half
一般需要指定:
--expname IFRNet: 指定使用的模型
--half 是否半精度
~~~
#### 文件说明
| index | Arbitrary |
| :----:| :----: |
| archs | 文件中添加网络定义代码 |
| images | 文件中添加测试文件 |
| utils | 一些处理相关代码 |
| check_archs.py | 加载图片，测试模型是否正确  |
| define_load_model.py | 定义模型加载 |
| ffmpeg_utils.py | ffmpeg相关预定义 |
| inference_config.py | 预配置文件 |
| inference_ema_vfi.py | EMA-VFI的推理代码 |
| inference_vfi.py | 通用推理代码 |      
| checkpoints | 放预训练模型文件 |      
  
#### 已集成list:
[EMA-VFI](https://github.com/mcg-nju/ema-vfi)    
[WaveletVFI](https://github.com/ltkong218/WaveletVFI)    
[IFRNet](https://github.com/ltkong218/IFRNet)    
#### 感谢开源，侵权可删
