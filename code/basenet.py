import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

#from resnet200 import Res200
#from resnext import ResNeXt
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Function
class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    def forward(self, x):
        return x.view_as(x)
    def backward(self, grad_output):
        return (grad_output*-self.lambd)
def grad_reverse(x,lambd=1.0):
    return GradReverse(lambd)(x)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)
 

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types                                                                                                                                                   
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm                                                                                                                                                            
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt()+self.eps
        x/=norm.expand_as(x)
        out = self.weight.unsqueeze(0).expand_as(x) * x
        return out
class BaseNet(nn.Module):
    #Model VGG
    def __init__(self):
        super(BaseNet, self).__init__()
        model_ft = models.vgg16(pretrained=True)
        mod = list(model_ft.features.children())
        self.features = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        self.classifier = nn.Sequential(*mod)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        return x
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        model_ft = models.alexnet(pretrained=True)
        mod = list(model_ft.features.children())
        self.features = model_ft.features#nn.Sequential(*mod)        
        print(self.features[0])
        #mod = list(model_ft.classifier.children())
        #mod.pop()

        #self.classifier = nn.Sequential(*mod)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),9216)
        #x = self.classifier(x)

        return x
class AlexNet_office(nn.Module):
    def __init__(self):
        super(AlexNet_office, self).__init__()
        model_ft = models.alexnet(pretrained=True)
        mod = list(model_ft.features.children())
        self.features = model_ft.features#nn.Sequential(*mod)        
        mod = list(model_ft.classifier.children())
        mod.pop()
        print(mod)
        self.classifier = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),9216)
        x = self.classifier(x)
        #x = F.dropout(F.relu(self.top(x)),training=self.training)

        return x
class AlexMiddle_office(nn.Module):
    def __init__(self):
        super(AlexMiddle_office, self).__init__()
        self.top = nn.Linear(4096,256)        
    def forward(self, x):
        x = F.dropout(F.relu(self.top(x)),training=self.training)
        return x


class AlexClassifier(nn.Module):
    # Classifier for VGG
    def __init__(self, num_classes=12):
        super(AlexClassifier, self).__init__()
        mod = []
        mod.append(nn.Dropout())
        mod.append(nn.Linear(4096,256))
        #mod.append(nn.BatchNorm1d(256,affine=True))
        mod.append(nn.ReLU())
        #mod.append(nn.Linear(256,256))
        mod.append(nn.Dropout())
        #mod.append(nn.ReLU())
        mod.append(nn.Dropout())
        #self.top = nn.Linear(256,256)        
        mod.append(nn.Linear(256,31))
        self.classifier = nn.Sequential(*mod)
    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x,reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x



class Classifier(nn.Module):
    # Classifier for VGG
    def __init__(self, num_classes=12):
        super(Classifier, self).__init__()
        model_ft = models.alexnet(pretrained=False)
        mod = list(model_ft.classifier.children())
        mod.pop()
        mod.append(nn.Linear(4096,num_classes))
        self.classifier = nn.Sequential(*mod)

    def forward(self, x):

        x = self.classifier(x)
        return x

class ClassifierMMD(nn.Module):
    def __init__(self, num_classes=12):
        super(ClassifierMMD, self).__init__()
        model_ft = models.vgg16(pretrained=True)
        mod = list(model_ft.classifier.children())
        mod.pop()
        self.classifier1 = nn.Sequential(*mod)
        self.classifier2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1000,affine=True),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            )
        self.last = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.classifier1(x)
        x1 = self.classifier2(x)
        x2 = self.classifier3(x1)
        x3 = self.last(x2)
        return x3,x2,x1
class ResBase(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        mod = list(model_ft.children())
        mod.pop()
        #self.model_ft =model_ft
        self.features = nn.Sequential(*mod)
    def forward(self, x):

        x = self.features(x)
        
        x = x.view(x.size(0), self.dim)
        return x
class ResBasePlus(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResBasePlus, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        mod = list(model_ft.children())
        mod.pop()
        #self.model_ft =model_ft
        self.layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 1000),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1000,affine=True),
            nn.Dropout(),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(*mod)
    def forward(self, x):

        x = self.features(x)        
        x = x.view(x.size(0), self.dim)
        x = self.layer(x)
        return x

class ResNet_all(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResNet_all, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        #mod = list(model_ft.children())
        #mod.pop()
        #self.model_ft =model_ft
        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        self.pool = model_ft.avgpool
        self.fc = nn.Linear(2048,12)
    def forward(self, x,layer_return = False,input_mask=False,mask=None,mask2=None):
        if input_mask:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = mask*self.layer1(x)
            fm2 = mask2*self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            return x#,fm1
        else:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = self.layer1(x)
            fm2 = self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            if layer_return:
                return x,fm1,fm2
            else:
                return x

class Mask_Generator(nn.Module):
    def __init__(self):
        super(Mask_Generator, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1,stride=1,padding=0)
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=1,stride=1,padding=0)
        self.bn1_2 = nn.BatchNorm2d(256)
        self.conv2_2 = nn.Conv2d(256, 512, kernel_size=1,stride=1,padding=0)

    def forward(self, x,x2):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.sigmoid(self.conv2(x))
        x2 = F.relu(self.bn1_2(self.conv1_2(x2)))
        x2 = F.sigmoid(self.conv2_2(x2))
        return x,x2


class ResMiddle_office(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResMiddle_office, self).__init__()
        self.dim = 2048
        layers = []
        layers.append(nn.Linear(self.dim,256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout())
        self.bottleneck = nn.Sequential(*layers)
        #self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = self.bottleneck(x)
        return x


class ResBase_office(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResBase_office, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        #mod = list(model_ft.children())
        #mod.pop()
        #self.model_ft =model_ft

        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool

        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        self.pool = model_ft.avgpool
        #self.bottleneck = nn.Sequential(*layers)
        #self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.pool(self.layer4(fm3))
        x = fm4.view(fm4.size(0), self.dim)
        #x = self.bottleneck(x)
        return x
class ResBase_D(nn.Module):
    def __init__(self,option = 'resnet18',pret=True):
        super(ResBase_D, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        #mod = list(model_ft.children())
        #mod.pop()
        #self.model_ft =model_ft
        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.drop0 = nn.Dropout2d()
        self.layer1 = model_ft.layer1
        self.drop1 = nn.Dropout2d()
        self.layer2 = model_ft.layer2
        self.drop2 = nn.Dropout2d()
        self.layer3 = model_ft.layer3
        self.drop3 = nn.Dropout2d()
        self.layer4 = model_ft.layer4
        self.drop4 = nn.Dropout2d()
        self.pool = model_ft.avgpool
        #self.features = nn.Sequential(*mod)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.drop0(x)
        conv_x = x
        x = self.maxpool(x)
        fm1 = self.layer1(x)
        x = self.drop1(x)
        fm2 = self.layer2(fm1)
        x = self.drop2(x)
        fm3 = self.layer3(fm2)
        x = self.drop3(x)
        fm4 = self.pool(self.drop4(self.layer4(fm3)))
        x = fm4.view(fm4.size(0), self.dim)
        return x

class ResBasePararrel(nn.Module):
    def __init__(self,option = 'resnet18',pret=True,gpu_ids=[]):
        super(ResBasePararrel, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        if option == 'resnet200':
            model_ft = Res200()
        if option == 'resnetnext':
            model_ft = ResNeXt(layer_num=101)
        mod = list(model_ft.children())
 