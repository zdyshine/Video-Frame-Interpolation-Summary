import argparse
import numpy as np
import torch
import torch.nn as nn
import os, cv2
from tqdm import tqdm
from collections import OrderedDict
# from IQA_pytorch import SSIM
import vimdo90k_dataset
from RRIN_arch import RRIN
import logging
import random
import utils
####################################################
NET_NAME = 'RRIN'

if not os.path.exists('./log/'):
    os.makedirs('./log/')

log_file_name = './log/' + NET_NAME + '_log.txt'
logging.basicConfig(level=logging.INFO, format='%(asctime)s  -  %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_file_name, filemode='a')


show_dir = './valimg_show/' + NET_NAME
os.makedirs(show_dir, exist_ok=True)
####################################################

def get_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        print('param_group',param_group['lr'])
        param_group['lr'] = lr

def load_network(load_path, network, strict=True):
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)

def main(args):
    logging.info(args)
    # set random seed
    print('========>Random Seed:', args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # make dirs
    save_path = os.path.join(args.save_path, NET_NAME)
    os.makedirs(save_path, exist_ok=True)
    # set device
    device = torch.device('cuda')
    # set model
    net = RRIN()

    if args.pretrained:
        print('=========> Load From: ', args.pretrained)
        load_network(args.pretrained, net)
    print_network(net)
    net = nn.DataParallel(net)
    net.to(device)
    # set dataload
    train_data = vimdo90k_dataset.MultiFramesDataset(args)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_data = vimdo90k_dataset.Middlebury_other(args)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=0.0)
    loss_criterion = nn.L1Loss().cuda()
    # loss_ssim = SSIMLoss(data_range=255.).cuda()
    # ssim_criterion = SSIM(channels=3).cuda()

    iters = 0
    best_psnr = -1.0
    learning_rate = args.lr
    print('=========> Start Train ...')
    for epoch in range(args.epochs):
        train_loss = 0
        train_loss_l1 = 0
        train_loss_ssim = 0
        net.train()
        torch.set_grad_enabled(True)

        # adjust lr
        if epoch % args.lr_decay == 0 and epoch != 0:
            learning_rate = learning_rate / 2.0
            if learning_rate < args.min_lr:
                learning_rate = args.min_lr
            get_learning_rate(optimizer, lr=learning_rate)

        for i, tensor in tqdm(enumerate(train_loader)):
            iters += 1
            # # adjust lr
            # if iters % args.lr_decay_iters == 0 and iters != 0:
            #     learning_rate = learning_rate / 5.0
            #     if learning_rate < args.min_lr:
            #         learning_rate = args.min_lr
            #     get_learning_rate(optimizer, lr=learning_rate)

            tensor0, tensor1, tensor2 = tensor[:, 0, ...].to(device), tensor[:, 1, ...].to(device), tensor[:, 2, ...].to(device)
            out_tensor = net(tensor0, tensor2)
            loss_l1 = loss_criterion(out_tensor, tensor1)
            # loss_l1 = l1_criterion(out_tensor * 255.0, tensor1 * 255.0)
            loss = loss_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_l1 += loss_l1.item()
            train_loss_ssim += 0 # loss_s.item()
            # if iters % 5 == 0:
            if iters % (len(train_loader) // 10) == 0:
                log_info = 'epoch:{}, Total iter:{}, iter:{}|{}, TLoss:{:.4f}, L1Loss:{:.4f}, SLoss:{}. Lr:{}'.format(
                    epoch, iters, i, len(train_loader), train_loss / (i + 1), train_loss_l1 / (i + 1),train_loss_ssim / (i + 1), learning_rate
                )
                print(log_info)
                logging.info(log_info)

        # val and save
        avg_psnr = validate(val_loader, net, epoch)
        is_best = avg_psnr > best_psnr
        best_psnr = max(avg_psnr, best_psnr)
        save_checkpoint(epoch, net, avg_psnr)
        if is_best:
            save_best_checkpoint(net)

def validate(val_loader, net, epoch):
    net.eval()
    avg_psnr = 0
    img_show = []
    with torch.no_grad():
        for i, tensor in enumerate(val_loader):
            tensor0, tensor1, tensor2 = tensor[:, 0, ...].cuda(), tensor[:, 1, ...].cuda(), tensor[:, 2, ...].cuda()
            out_tensor = net(tensor0, tensor2)
            img_show.append(out_tensor)
            output = utils.tensor2img(out_tensor.float().cpu().squeeze(0))
            target = utils.tensor2img(tensor1)
            pnsr_value = utils.compute_psnr(target, output)
            avg_psnr += pnsr_value

    # img_show = torch.stack(img_show, dim=1)
    save_dir = show_dir + '/{}'.format(epoch)
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(len(img_show)):
        img = utils.tensor2img(img_show[idx])
        cv2.imwrite(save_dir + '/{}.png'.format(idx), img[:, :, ::-1])

    print("===> Valid. psnr: {:.4f}".format(avg_psnr / len(val_loader)))
    # log_str = "===> Valid. psnr: {:.4f}".format(avg_psnr / len(val_loader))
    #logging.info(log_str)
    return avg_psnr / len(val_loader)

def save_checkpoint(iteration, model, psnr=1):
    model_folder = os.path.join('model', NET_NAME) #"model/PTMR/"
    os.makedirs(model_folder, exist_ok=True)
    model_out_path = model_folder + "/{}_{}_{:.3f}.pth".format(NET_NAME, iteration, psnr)
    # model_out_path = model_folder + "/{}_{}.pth".format(NET_NAME, iteration)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)

def save_best_checkpoint(model):
    model_folder = os.path.join('model', NET_NAME)
    os.makedirs(model_folder, exist_ok=True)
    model_out_path = model_folder + "/best_model.pth"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)
    logging.info('Total number of parameters: %d' % num_params)

if __name__== '__main__':
    parser = argparse.ArgumentParser(description="Video Completion")

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='RRIN')

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')

    # Directory Setting
    parser.add_argument('--train', type=str, default='/dataset/vimeo90k/vimeo_septuplet')
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")
    parser.add_argument('--test_input', type=str, default='/dataset/test/middlebury_others/input')
    parser.add_argument('--gt', type=str, default='/dataset/test/middlebury_others/gt')

    # Learning Options
    parser.add_argument('--epochs', type=int, default=110, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--loss', type=str, default='l1', help='loss function configuration')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=30, help='learning rate decay per N epochs')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'),
                        help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    main(args)
