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
print('res_train_main_D.py 不带有stepD的训练过程,作为base来比较cor_loss效果,统计错误分类的类别,2022/03/01')
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
parser.add_argument('--resnet', type=str, default='101', metavar='B',
                    help='which resnet 18,50,101,152,200')

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
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path,val_path]}
dsets_2 ={z: datasets.ImageFolder(os.path.join(z), data_transforms[z]) for z in [train_path,val_path]} 
# print(dsets[train_path])
dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
print ('classes'+str(dset_classes))
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path],dsets[val_path],batch_size,shuffle=True)#打乱数据
dataset = train_loader.load_data()
#return {'S': A, 'S_label': A_paths,
#        'T': B, 'T_label': B_paths}

train_2_loader = CVDataLoader()
train_2_loader.initialize(dsets[train_path],dsets[val_path],batch_size,shuffle=False)#F1,F2训练corloss 不打乱数据
dataset_2 = train_2_loader.load_data()


test_loader = CVDataLoader()
opt= args
# test_loader.initialize(dsets[train_path],dsets[val_path],batch_size,shuffle=True)#打乱测试数据
test_loader.initialize(dsetsTest[train_path],dsetsTest[test_path],batch_size,shuffle=False)#不打乱测试数据
dataset_test = test_loader.load_data()
option = 'resnet'+args.resnet   #resnet101
G = ResBase(option)
F1 = ResClassifier(num_layer=num_layer)
F2 = ResClassifier(num_layer=num_layer)
F1.apply(weights_init)
F2.apply(weights_init)
lr = args.lr

if args.cuda:
    G.cuda()
    F1.cuda()
    F2.cuda()
if args.optimizer == 'momentum':
    optimizer_g = optim.SGD(list(G.features.parameters()), lr=args.lr,weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F1.parameters())+list(F2.parameters()),momentum=0.9,lr=args.lr,weight_decay=0.0005)
elif args.optimizer == 'adam':
    optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr,weight_decay=0.0005)
    optimizer_f = optim.Adam(list(F1.parameters())+list(F2.parameters()), lr=args.lr,weight_decay=0.0005)
else:
    optimizer_g = optim.Adadelta(G.features.parameters(),lr=args.lr,weight_decay=0.0005)
    optimizer_f = optim.Adadelta(list(F1.parameters())+list(F2.parameters()),lr=args.lr,weight_decay=0.0005)    
    
def train(num_epoch):
    criterion = nn.CrossEntropyLoss().cuda()
    for i in range(1):
        cweight = 0
        # cweight = (i+7)*cweight#0.01 - 0.1
        # if cweight == 0.06:
        #     continue
        print('===================================================================================')   
        print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
        print('cor_loss 权重为:',cweight)  
        for ep in range(num_epoch):
            print("----------------------------------------------------------------")
            print('ep = ',ep)
            G.train()
            F1.train()
            F2.train()
            for batch_idx, (data,data_2) in enumerate(zip(dataset,dataset_2)):
                # print('data = ',data)
                # print('batch_idx = ',batch_idx)
                if (batch_idx+1) * batch_size > 39000:#39000:#train文件共1300x30=39000个图片,舍弃最后不足一个batch的数据
                    break                              #30个类39000个图片
                if args.cuda:
                    data1 = data['S']
                    target1 = data['S_label']
                    # print('data1 = ',data1)
                    # print('S_label =', target1)
                    data1_1 = data_2['S']
                    target1_1 = data_2['S_label']

                    data2  = data['T']
                    target2 = data['T_label']
                    data2_1 = data_2['T']
                    target2_1 = data_2['T_label'] 
                    # print('data2 = ',data2)
                    # print('T_label =',target2)
                    data1, target1 = data1.cuda(), target1.cuda()
                    data2, target2 = data2.cuda(), target2.cuda()
                    data1_1, target1_1 = data1_1.cuda(), target1_1.cuda()
                    data2_1,target2_1 = data2_1.cuda(),target2_1.cuda()
                # when pretraining network source only
                eta = 1.0
                data = Variable(torch.cat((data1,data2),0))#torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现
                data_2 = Variable(torch.cat((data1_1,data2_1),0))
                # print('data = ',data)
                target1 = Variable(target1)
                target1_1 = Variable(target1_1)#未打乱的数据data2的源域标签
                # Step A train all networks to minimize loss on source
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                output = G(data)
                output1 = F1(output)
                output2 = F2(output)
                # print('output = ',output)
                # print('output1 = ',output1)

                output_s1 = output1[:batch_size,:]  #F1对于源域的输出
                output_s2 = output2[:batch_size,:]  #F2对于源域的输出
                output_t1 = output1[batch_size:,:]  #F1对于目标域的输出
                output_t2 = output2[batch_size:,:]  #F2对于目标域的输出
                # print('output_s1 = ',output_s1)
                # print('output_t1 = ',output_t1)
                output_t1 = F.softmax(output_t1,dim=1)
                output_t2 = F.softmax(output_t2,dim=1)
                # print('After softmax,output_t1 = ',output_t1)
                entropy_loss = - torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
                #.mean():dim=0,按列求平均值，返回的形状是（1，列数);dim=1,按行求平均值，返回的形状是（行数，1）
                #torch.log是以自然数e为底的对数函数。
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

                loss1 = criterion(output_s1, target1)   #output_s1是源域的预测,target1 = data['S_label']
                loss2 = criterion(output_s2, target1) #损失是与‘S_label’比较
                all_loss = loss1 + loss2 + 0.01 * entropy_loss
                all_loss.backward()
                optimizer_g.step()
                optimizer_f.step()

                #Step B train classifier to maximize discrepancy
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                output = G(data)    #data = Variable(torch.cat((data1,data2),0)
                output1 = F1(output)
                output2 = F2(output)
                output_s1 = output1[:batch_size,:]
                output_s2 = output2[:batch_size,:]
                output_t1 = output1[batch_size:,:]
                output_t2 = output2[batch_size:,:]
                output_t1 = F.softmax(output_t1,dim=1)#https://blog.csdn.net/will_ye/article/details/104994504 dim=1:按列
                output_t2 = F.softmax(output_t2,dim=1)
                loss1 = criterion(output_s1, target1)
                loss2 = criterion(output_s2, target1)

                entropy_loss = - torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))
                loss_dis = torch.mean(torch.abs(output_t1-output_t2))
                F_loss = loss1 + loss2 - eta*loss_dis  + 0.01 * entropy_loss
                F_loss.backward()
                optimizer_f.step()

                # Step C train genrator to minimize discrepancy
                for i in range(num_k):
                    optimizer_g.zero_grad()
                    output = G(data)
                    output1 = F1(output)
                    output2 = F2(output)

                    output_s1 = output1[:batch_size,:]
                    output_s2 = output2[:batch_size,:]
                    output_t1 = output1[batch_size:,:]
                    output_t2 = output2[batch_size:,:]

                    loss1 = criterion(output_s1, target1)
                    loss2 = criterion(output_s2, target1)
                    output_t1 = F.softmax(output_t1,dim=1)
                    output_t2 = F.softmax(output_t2,dim=1)
                    loss_dis = torch.mean(torch.abs(output_t1-output_t2))
                    entropy_loss = -torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
                    entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

                    loss_dis.backward()
                    optimizer_g.step()
                
                #step D
                # {
                # optimizer_f.zero_grad()
                # output_2 = G(data_2)
                # output1_2 = F1(output_2)
                # output2_2 = F2(output_2)
                
                # output_s1 = output1_2[:batch_size,:]
                # output_s2 = output2_2[:batch_size,:]
                # output_t1 = output1_2[batch_size:,:]
                # output_t2 = output2_2[batch_size:,:]

                # loss1 = criterion(output_s1, target1_1)
                # loss2 = criterion(output_s2, target1_1)
                # output1_2 = F.softmax(output_t1,dim=1)
                # output2_2 = F.softmax(output_t2,dim=1) 
                # cor_loss = torch.mean(torch.std(output1_2[:12],axis=0)+torch.std(output1_2[12:],axis=0)+torch.std(output2_2[:12],axis=0)+torch.std(output2_2[12:],axis=0))
                # cor_loss=cweight*cor_loss
                # entropy_loss = -torch.mean(torch.log(torch.mean(output1_2,0)+1e-6))
                # entropy_loss -= torch.mean(torch.log(torch.mean(output2_2,0)+1e-6))
                
                # cor_loss.backward()
                # optimizer_f.step()
                # }

                # print('output1_2[:12]',output1_2[:12])
                # print('torch.std(output1_2,axis=0)+torch.std(output2_2,axis=0)=',torch.std(output1_2,axis=0)+torch.std(output2_2,axis=0))
                # print('batch_idx={},  cor_loss={:.6f}'.format(batch_idx,cor_loss))


                # if batch_idx % 200 == 0: #每100个batch输出一下
                #     print('Train Ep: {} [{}/{} ({:.0f}%)] cor_loss:{:.6f}  Loss1: {:.6f}  Loss2: {:.6f}  Dis: {:.6f} Entropy: {:.6f}'.format(
                #     ep, batch_idx * 24, 39000,
                #     100. * batch_idx * 24/ 39000,cor_loss/cweight,loss1.item(),loss2.item(),loss_dis.item(),entropy_loss.item()))
                
                if batch_idx % 200 == 0: #每100个batch输出一下
                    print('Train Ep: {} [{}/{} ({:.0f}%)]  Loss1: {:.6f}  Loss2: {:.6f}  Dis: {:.6f} Entropy: {:.6f}'.format(
                    ep, batch_idx * 24, 39000,
                    100. * batch_idx * 24/ 39000,loss1.item(),loss2.item(),loss_dis.item(),entropy_loss.item()))

                if batch_idx == 0 and ep >1:    #迭代两次之后开始测试
                    test(ep)
                    G.train()
                    F1.train()
                    F2.train()

def test(epoch):
    G.eval()
    F1.eval()
    F2.eval()
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
        if args.cuda:
            data2  = data['T']
            target2 = data['T_label']
            if val:     #?不懂这个是什么意思
                data2  = data['S']
                target2 = data['S_label']
            data2, target2 = data2.cuda(), target2.cuda()   #.cuda 移入GPU的数据形式 
        data1, target1 = Variable(data2), Variable(target2)
        for idx in target1:
            class_idx[idx] +=