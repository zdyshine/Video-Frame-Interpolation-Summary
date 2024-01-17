from decord import VideoReader
from decord import cpu, gpu

video_path = "N:/liuh/output_600s.mp4"
vr = VideoReader(video_path, ctx=cpu(0))
frame_length = len(vr)

# 逐帧
img = vr.next().asnumpy()

# 跳帧
vr.seek(0)
for i in range(frame_length//50):
    vr.seek(i*50)
    vr.seek_accurate()
    img = vr.next().asnumpy()

#指定帧
frame_index_list = [7, 62, 117, 172, 227, 275]
frames = vr.get_batch(frame_index_list).asnumpy()
# decord 指定帧块
# av 指定帧块
