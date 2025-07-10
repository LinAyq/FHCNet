import os
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.model import Net
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
file = open("log/****.txt", "a")
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
np.random.seed(3407)
torch.backends.cudnn.benchmark = True

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    # size_rates = [0.75, 1, 1.25]
    size_rates = [1]
    loss_recordh4, loss_recordh3, loss_recordh2 = AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            g4, g3, g2 = model(images)
            # ---- loss function ----
            lossh4 = structure_loss(g4, gts)
            lossh3 = structure_loss(g3, gts)
            lossh2 = structure_loss(g2, gts)

            loss = lossh4 + lossh3 + lossh2  # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_recordh4.update(lossh4.data, opt.batchsize)
                loss_recordh3.update(lossh3.data, opt.batchsize)
                loss_recordh2.update(lossh2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-h4: {:0.4f}, lateral-h3: {:0.4f}, lateral-h2: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_recordh4.avg, loss_recordh3.avg,
                         loss_recordh2.avg))

            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-h4: {:0.4f}, lateral-h3: {:0.4f}, lateral-h2: {:0.4f}]\n'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                              loss_recordh4.avg, loss_recordh3.avg,
                              loss_recordh2.avg))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 1 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'Net-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'Net-%d.pth'% epoch)
        file.write('[Saving Snapshot:]' + save_path + 'Net-%d.pth' % epoch + '\n')

def test(model, epoch, opt):
    global best_mae,best_epoch
    test_data_path = '*****/TestDataset/CAMO' # your test data path
    save_path = '*****/snapshots/{}/'.format(opt.train_save) # your snapshots save path

    model.eval()

    image_root = '{}/Imgs/'.format(test_data_path)
    gt_root = '{}/GT/'.format(test_data_path)
    test_loader = test_dataset(image_root, gt_root, opt.trainsize)
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, _, _ = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum+=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae = mae_sum/test_loader.size

        if epoch==1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        print('Epoch: {} MAE: {} #### bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        file.write('Epoch: {} MAE: {} #### bestMAE: {} bestEpoch: {}\n'.format(epoch,mae,best_mae,best_epoch))


best_mae=1
best_epoch=0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=384, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='*******/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='*******')
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()
    pre_path = 'swin_base_patch4_window12_384_22k.pth'
    model.load_pre(pre_path)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)
        test(model,epoch,opt)

    file.close()