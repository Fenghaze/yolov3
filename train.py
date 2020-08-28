#!/usr/bin/python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import dataset
import torch.optim as optim
import argparse
from Darknet import Darknet
from utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter

use_cuda = True if torch.cuda.is_available() else False
device = 'cuda' if use_cuda else 'cpu'

FLAGS = None
conf_thresh = 0.25
nms_thresh = 0.4
iou_thresh = 0.5
eps = 1e-5


# def test(batch_idx):
#
#     print("Tesing...")
#
#     def truth_length(truths):
#         for i in range(50):
#             if truths[i][1]==0:
#                 return i
#         return 50
#
#     total = 0.0
#     proposals = 0.0
#     correct = 0.0
#
#     global model
#     model.eval()
#     net_w = model.width
#     net_h = model.height
#     nC = int(model.num_classes)
#     with torch.no_grad():
#         try:
#             with tqdm(test_loader) as t:
#                 for imgs, labels, org_w, org_h in t:
#                     imgs = imgs.to(device)
#                     labels = labels.to(device)
#
#                     output = model(imgs)
#                     all_boxes = get_all_boxes(output, (net_w, net_h), conf_thresh, nC)
#
#                     # for every single image
#                     for i in range(len(all_boxes)):
#                         boxes = all_boxes[i]
#                         correct_yolo_boxes(boxes, org_w[i], org_h[i], model.width, model.height)
#                         boxes = np.array(nms(boxes, nms_thresh=nms_thresh))
#
#                         num_pred = len(boxes)
#                         if num_pred == 0:
#                             continue
#                         truths = labels[i].view(-1, 5)
#                         num_gts = truth_length(truths)
#                         total += num_gts
#                         proposals += (boxes[: 4] > 0).sum()
#
#                         for k in range(num_gts):
#                             gt_box = torch.FloatTensor([truths[k][1], truths[k][2],
#                                                         truths[k][3], truths[k][4], 1.0, 1.0, truths[k][0]])
#                             gt_box = gt_box.repeat(num_pred, 1).t()
#                             pred_box = torch.FloatTensor(boxes).t()
#                             best_iou, best_j = torch.max(cal_ious(gt_box, pred_box), 0)
#                             if best_iou > iou_thresh and pred_box[6][best_j] == gt_box[6][0]:
#                                 correct += 1
#         except KeyboardInterrupt:
#             t.close()
#             raise
#         t.close()
#     precision = 1.0 * correct / (proposals + eps)
#     recall = 1.0 * correct / (total + eps)
#     fscore = 2.0 * precision * recall / (precision + recall)
#     print('batch:{} precision:{:2f}, recall:{:2f}, fscore:{:2f}'.format(batch_idx, precision, recall, fscore))
#     save_logging('precision:{:2f}, recall:{:2f}, fscore:{:2f}'.format(precision, recall, fscore))
#     return correct, fscore


def main():
    global loss_layers
    global test_loader
    global model
    data_options = read_data_file(FLAGS.data)
    net_options = parse_cfg(FLAGS.config)[0]

    train_dir = data_options['train']
    test_dir = data_options['valid']
    names = data_options['names']

    batch_size = int(net_options['batch'])
    learning_rate = float(net_options['learning_rate'])
    hue = float(net_options['hue'])
    hue = float(net_options['hue'])
    exposure = float(net_options['exposure'])
    saturation = float(net_options['saturation'])
    momentum = float(net_options['momentum'])

    epochs = 50

    model = Darknet(FLAGS.config)
    torch.manual_seed(0)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = data_options['gpus']
        torch.cuda.manual_seed(0)

    model = model.to(device)
    model.load_weights(weightfile=FLAGS.weight)
    loss_layers = model.loss_layers
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_data = dataset.YoloDataset(train_dir, (model.width, model.height),
                                     transform=transforms.ToTensor(), train=True)
    test_data = dataset.YoloDataset(test_dir, (model.width, model.height),
                                    transform=transforms.ToTensor(), train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    writer = SummaryWriter('runs')
    for epoch in range(epochs):
        model.train()
        print('\n\nStarting epoch %d / %d' % (epoch + 1, epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        org_loss = []
        loss1 = 0
        loss2 =0
        loss3 = 0
        total_loss = 0
        for idx, (images, labels) in enumerate(train_loader):
            # print(idx, images.shape, labels.shape)
            images = images.to(device)  # [n, 3, 416, 416]
            labels = labels.to(device)  # [n, 250]
            optimizer.zero_grad()
            output = model(images)  # output[0]:[n, 3*25, 13, 13]; output[1]:[n, 3*25, 26, 26]; output[2]:[n, 3*25, 52, 52]
            for i, l in enumerate(loss_layers):
                l.seen += labels.data.size(0)
                ol = l(output[i]['output'], labels)
                if i ==0:
                    loss1 = ol
                elif i==1:
                    loss2 = ol
                else:
                    loss3 = ol
            #sum(org_loss).backward()
            loss = loss1+loss2+loss3
            loss.backward()
            total_loss += loss
            optimizer.step()
            if (idx + 1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d]average_loss: %.4f, current_lr: %f'
                      % (epoch + 1, epochs, idx + 1, len(train_loader), total_loss / (idx + 1),
                         learning_rate))
            # if (idx + 1) % 250 == 0:
            #     model.save_weights('models/batch_{}.weights'.format(idx))
            #     print('Model saved.')
            #     # test(idx)
        writer.add_scalar('train_loss', total_loss/len(train_loader), epoch)
        model.save_weights('models_scratch/epoch_{}.weights'.format(epoch+1))
        print('Epoch_{:d} model saved.'.format(epoch + 1))
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',
                        type=str, default='data/voc.data', help='data description info.')
    parser.add_argument('--config', '-c',
                        type=str, default='data/yolo_v3.cfg', help='cfg file.')
    parser.add_argument('--weight', '-w',
                        type=str, default='../yolov3_pth/yolov3.weights', help='yolov3 weight file.')
    FLAGS = parser.parse_args()
    main()
