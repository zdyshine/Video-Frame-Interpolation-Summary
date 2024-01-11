# enhance_infer

## 快速集成开源模型并进行统一标准的测试
## 实现图片和视频的快速修复增强
#### 使用方式
archs文件中添加网络定义代码    
mkdir checkpoints 文件夹放置对应网络的预训练模型    
defint_network中添加调用模型和加载预训练模型代码    
inference.py进行图片推理        
inference_ffmpeg.py加载视频推理    
run.py python调用ffmpeg, ffmpeg混合滤镜，指定显示宽高比，指定视频时段    

#### 已集成list:
https://github.com/xinntao/Real-ESRGAN    
https://github.com/sunny2109/SAFMN    
https://github.com/Zj-BinXia/DiffIR    
#### 感谢开源，侵权可删
