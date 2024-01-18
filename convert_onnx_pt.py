import os
import cv2
import glob
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.onnx

@torch.no_grad()
def convert_onnx(model, output_folder, is_dynamic_batches=False):
    output_name = os.path.join(output_folder, 'test.onnx')
    dynamic_params = None
    if is_dynamic_batches:
        dynamic_params = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    # Export the model
    torch.onnx.export(model,  # model being run
                      fake_x,  # model input (or a tuple for multiple inputs)
                      output_name,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes=dynamic_params)

    print('Model has been converted to ONNX')

@torch.no_grad()
def convert_pt(model, output_folder):
    output_name = os.path.join(output_folder, 'test.pt')

    traced_module = torch.jit.trace(model, fake_x)
    traced_module.save(output_name)
    print('Model has been converted to pt')


def test_onnx(onnx_model, input_path, save_path):
    # for GPU inference
    # ort_session = ort.InferenceSession(onnx_model, ['CUDAExecutionProvider'])

    ort_session = ort.InferenceSession(onnx_model)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(input_path, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]

        print(f'Testing......idx: {idx}, img: {imgname}')

        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        if img.size != (960, 640):
            img = cv2.resize(img, (960, 640), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, HWC -> CHW
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)

        output = ort_session.run(None, {"input": img})

        # save image
        print('Saving!')
        output = np.squeeze(output[0], axis=0)
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

        output = (output.clip(0, 1) * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_SAFMN.png'), output)


if __name__ == "__main__":
    target_H, target_W = 480, 848
    simplifier = True

    from archs_mg.downv2_arch import DownNetNFV2
    model = DownNetNFV2(target_size=(target_H, target_W)).eval().cuda()

    # from archs_mg.naflka_gpu_arch import NAFLKAGPUNet
    # model = NAFLKAGPUNet().eval().cuda()

    fake_x = torch.rand(1, 3, 1080, 1920, requires_grad=False).cuda()

    # pretrained_model = 'experiments/pretrained_models/SAFMN_L_Real_LSDIR_x2.pth'
    # model.load_state_dict(torch.load(pretrained_model)['params'], strict=True)

    # ###################Onnx export#################
    output_folder = 'convert_onnx_pt'
    os.makedirs(output_folder, exist_ok=True)

    convert_onnx(model, output_folder)
    convert_pt(model, output_folder)
    if simplifier:
        cmd = 'onnxsim' + f' "{output_folder}/test.onnx"' + f' "{output_folder}/test_sim.onnx"'
        os.system(cmd)

    # ###################Test the converted model #################
    # onnx_model = 'convert/SAFMN_640_960_x2.onnx'
    # input_path = 'datasets/real_test'
    # save_path = 'results/onnx_results'
    # test_onnx(onnx_model, input_path, save_path)