from __future__ import print_function
from scipy.stats import mode
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
import torch.nn.functional as F
import os
import time
# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=24, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num_layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='save', metavar='B',
                    help='board dir')
parser.add_argument('--train_path', type=str, default='/home/ly3090/wym/data/train/ImaMod_share_picture1300', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='/home/ly3090/MCD_DA/train_test', metavar='B',
                    help='directory of target train datasets')
parser.add_argument('--test_path', type=str, default='/home/ly3090/wym/data/test', metavar='B',
                    help='directory of target test datasets')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
train_path = args.train_path    #./data/test
val_path = args.val_path    #./data/train
test_path = args.test_path
num_k = args.num_k  #how many steps to repeat the generator update
num_layer = args.num_layer #how many layers for classifier
batch_size = args.batch_size 
# save_path = args.save+'_'+str(args.num_k) #save_4

data_transforms = {
    train_path: transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    test_path: transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

}
dsetsTest = {y: datasets.ImageFolder(os.path.join(y), data_transforms[y]) for y in [train_path,test_path]} 
# print(dsets[train_path])
dset_sizes = {y: len(dsetsTest[y]) for y in [train_path, test_path]}
dset_classes = dsetsTest[train_path].classes
print ('classes'+str(dset_classes))
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_loader = CVDataLoader()
opt= args
# test_loader.initialize(dsets[train_path],dsets[val_path],batch_size,shuffle=True)#打乱测试数据
test_loader.initialize(dsetsTest[train_path],dsetsTest[test_path],batch_size,shuffle=False)#不打乱测试数据
dataset_test = test_loader.load_data()

#for x in test_loader.data_loader_B.batch_sampler.__iter__():
#    print(x)
print(list(dataset_test.data_loader_B.batch_sampler.__iter__())[0])
#print(test_loader.data_loader_B.batch_sampler.__iter__())
#print("dsetsTest[train_path]:\n", dsetsTest[test_path].imgs[21624])
def test():
    test_loss = 0
    correct = 0
    correct2 = 0
    size = 0
    val = 0
    total_class_num = 30
    test_class_num = 30
    class_idx = [0 for x in range(total_class_num)]
    class_target1_idx = [0 for x in range(test_class_num)] #F1 pred idx  
    class_target2_idx = [0 for x in range(test_class_num)]
    model3D_class_target1_idx = [0 for x in range(test_class_num)] #F1 pred idx  
    model3D_class_target2_idx = [0 for x in range(test_class_num)]
    model3D_wrong_classfication = [[0 for i in range(30)] for j in range(30)]
    
    
    for batch_idx, data in enumerate(dataset_test):
        if (batch_idx+1)*batch_size > 21624:#21624#测试文件有21624张图片,舍弃最后一个不足的batch
            break
        print("batch_idx", batch_idx)
        print(get_imgdir(dsetsTest[test_path], list(dataset_test.data_loader_B.batch_sampler.__iter__()), batch_idx))
        if args.cuda:
            data2  = data['T']
            target2 = data['T_label']
            if val:     #?不懂这个是什么意思
                data2  = data['S']
                target2 = data['S_label']
            data2, target2 = data2.cuda(), target2.cuda()   #.cuda 移入GPU的数据形式 
        data1, target1 = Variable(data2), Variable(target2)
        for idx in target1:
            class_idx[idx] += 1     #统计三十个类别的图片数量

def get_imgdir(dataset: 'class datasets, for example datasets.ImageFolder',
                batch_sampler_list: 'the sampler list of batch',
                batch_idx: 'idx of batch') -> list:
    #该函数是dataset进行读取某个batch的图片路径
    # 第二个参数是dataloader里的batch_sampler给出的batch idx列表 如：list(dataset_test.data_loader_B.batch_sampler.__iter__()))
    img_list = dataset.imgs
    batch_idx_list = batch_sampler_list[batch_idx]
    return [img_list[index][0] for index in batch_idx_list]

#def get_imgdir(dataset: 'class datasets, for example datasets.ImageFolder',
#                batch_idx: 'the idx of batch') -> list:
#    #该函数是对顺序未经过打乱的dataset进行读取某个batch的图片路径
#    img_list = dataset.imgs
#    return [x[0] for x in img_list[batch_idx*batch_size: (batch_idx+1)*batch_size]]

#print(get_imgdir(dsetsTest[test_path],0))
test()