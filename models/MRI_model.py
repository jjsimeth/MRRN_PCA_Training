import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from math import floor,isnan
#from options.seg_options import SegmentationOptions
from options.train_options import TrainOptions

import torch.nn.functional as F

import torch.nn as nn

#opt = SegmentationOptions().parse()
opt2 = TrainOptions().parse()
num_organ = 1
imsize = opt2.fineSize #448

from torch.nn.modules.loss import _Loss
#from sklearn.utils.class_weight import compute_class_weight

class JDTLoss(_Loss):
    def __init__(self,
                 mIoUD=1.0,
                 mIoUI=0.0,
                 mIoUC=0.0,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 smooth=1e-3,
                 threshold=0.01,
                 norm=1,
                 log_loss=False,
                 ignore_index=None,
                 class_weights=None,
                 active_classes_mode_hard="PRESENT",
                 active_classes_mode_soft="ALL"):
        """
        Arguments:
            mIoUD (float): The weight of the loss to optimize mIoUD.
            mIoUI (float): The weight of the loss to optimize mIoUI.
            mIoUC (float): The weight of the loss to optimize mIoUC.
            alpha (float): The coefficient of false positives in the Tversky loss.
            beta (float): The coefficient of false negatives in the Tversky loss.
            gamma (float): When `gamma` > 1, the loss focuses more on
                less accurate predictions that have been misclassified.
            smooth (float): A floating number to avoid `NaN` error.
            threshold (float): The threshold to select active classes.
            norm (int): The norm to compute the cardinality.
            log_loss (bool): Compute the log loss or not.
            ignore_index (int | None): The class index to be ignored.
            class_weights (list[float] | None): The weight of each class.
                If it is `list[float]`, its size should be equal to the number of classes.
            active_classes_mode_hard (str): The mode to compute
                active classes when training with hard labels.
            active_classes_mode_soft (str): The mode to compute
                active classes when training with hard labels.

        Comments:
            Jaccard: `alpha`  = 1.0, `beta`  = 1.0
            Dice:    `alpha`  = 0.5, `beta`  = 0.5
            Tversky: `alpha` >= 0.0, `beta` >= 0.0
        """
        super().__init__()

        assert mIoUD >= 0 and mIoUI >= 0 and mIoUC >= 0 and \
               alpha >= 0 and beta >= 0 and gamma >= 1 and \
               smooth >= 0 and threshold >= 0
        assert isinstance(norm, int) and norm > 0
        assert ignore_index == None or isinstance(ignore_index, int)
        assert class_weights == None or all((isinstance(w, float)) for w in class_weights)
        assert active_classes_mode_hard in ["ALL", "PRESENT"]
        assert active_classes_mode_soft in ["ALL", "PRESENT"]

        self.mIoUD = mIoUD
        self.mIoUI = mIoUI
        self.mIoUC = mIoUC
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.threshold = threshold
        self.norm = norm
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        if class_weights == None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.tensor(class_weights)
        self.active_classes_mode_hard = active_classes_mode_hard
        self.active_classes_mode_soft = active_classes_mode_soft


    def forward(self, logits, label, keep_mask=None):
        """
        Arguments:
            logits (torch.Tensor): Its shape should be (B, C, D1, D2, ...).
            label (torch.Tensor):
                If it is hard label, its shape should be (B, D1, D2, ...).
                If it is soft label, its shape should be (B, C, D1, D2, ...).
            keep_mask (torch.Tensor | None):
                If it is `torch.Tensor`,
                    its shape should be (B, D1, D2, ...) and
                    its dtype should be `torch.bool`.
        """
        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long

        logits = logits.view(batch_size, num_classes, -1)
        prob = logits.log_softmax(dim=1).exp()

        if keep_mask != None:
            assert keep_mask.dtype == torch.bool
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)
        elif self.ignore_index != None and hard_label:
            keep_mask = label != self.ignore_index
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)

        if hard_label:
            label = torch.clamp(label, 0, num_classes - 1).view(batch_size, -1)
            label = F.one_hot(label, num_classes=num_classes).permute(0, 2, 1).float()
            active_classes_mode = self.active_classes_mode_hard
        else:
            label = label.view(batch_size, num_classes, -1)
            active_classes_mode = self.active_classes_mode_soft

        loss = self.forward_loss(prob, label, keep_mask, active_classes_mode)

        return loss


    def forward_loss(self, prob, label, keep_mask, active_classes_mode):
        if keep_mask != None:
            prob = prob * keep_mask
            label = label * keep_mask

        prob_card = torch.norm(prob, p=self.norm, dim=2)
        label_card = torch.norm(label, p=self.norm, dim=2)
        diff_card = torch.norm(prob - label, p=self.norm, dim=2)

        if self.norm > 1:
            prob_card = torch.pow(prob_card, exponent=self.norm)
            label_card = torch.pow(label_card, exponent=self.norm)
            diff_card = torch.pow(diff_card, exponent=self.norm)

        tp = (prob_card + label_card - diff_card) / 2
        fp = prob_card - tp
        fn = label_card - tp

        loss = 0
        batch_size, num_classes = prob.shape[:2]
        if self.mIoUD > 0:
            active_classes = self.compute_active_classes(label, active_classes_mode, num_classes, (0, 2))
            loss_mIoUD = self.forward_loss_mIoUD(tp, fp, fn, active_classes)
            loss += self.mIoUD * loss_mIoUD

        if self.mIoUI > 0 or self.mIoUC > 0:
            active_classes = self.compute_active_classes(label, active_classes_mode, (batch_size, num_classes), (2, ))
            loss_mIoUI, loss_mIoUC = self.forward_loss_mIoUIC(tp, fp, fn, active_classes)
            loss += self.mIoUI * loss_mIoUI + self.mIoUC * loss_mIoUC

        return loss


    def compute_active_classes(self, label, active_classes_mode, shape, dim):
        if active_classes_mode == "ALL":
            mask = torch.ones(shape, dtype=torch.bool)
        elif active_classes_mode == "PRESENT":
            mask = torch.amax(label, dim) > self.threshold

        active_classes = torch.zeros(shape, dtype=torch.bool, device=label.device)
        active_classes[mask] = 1

        return active_classes


    def forward_loss_mIoUD(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(tp)

        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            loss_mIoUD = -torch.log(tversky)
        else:
            loss_mIoUD = 1.0 - tversky

        if self.gamma > 1:
            loss_mIoUD **= self.gamma

        if self.class_weights != None:
            loss_mIoUD *= self.class_weights

        loss_mIoUD = loss_mIoUD[active_classes]
        loss_mIoUD = torch.mean(loss_mIoUD)

        return loss_mIoUD


    def forward_loss_mIoUIC(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(tp), 0. * torch.sum(tp)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            loss_matrix = -torch.log(tversky)
        else:
            loss_matrix = 1.0 - tversky

        if self.gamma > 1:
            loss_matrix **= self.gamma

        if self.class_weights != None:
            class_weights = self.class_weights.unsqueeze(0).expand_as(loss_matrix)
            loss_matrix *= class_weights

        loss_matrix *= active_classes
        loss_mIoUI = self.reduce(loss_matrix, active_classes, 1)
        loss_mIoUC = self.reduce(loss_matrix, active_classes, 0)

        return loss_mIoUI, loss_mIoUC


    def reduce(self, loss_matrix, active_classes, dim):
        active_sum = torch.sum(active_classes, dim)
        active_dim = active_sum > 0
        loss = torch.sum(loss_matrix, dim)
        loss = loss[active_dim] / active_sum[active_dim]
        loss = torch.mean(loss)

        return loss



class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.ones = torch.eye(depth).to(device)

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


# class SoftDiceLoss(nn.Module):
#     def __init__(self, n_classes=2):
#         super(SoftDiceLoss, self).__init__()
#         self.one_hot_encoder = One_Hot(n_classes).forward
#         self.n_classes = n_classes

#     def forward(self, input, target):
#         smooth = 0.0001
#         #print(np.shape(input))
#         batch_size = input.size(0)
#         target.double()
#         # valid_mask = target.ne(-1)
#         # target = target.masked_fill_(~valid_mask, 0)
        
#         input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
#         #target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
#         #target = F.softmax(target, dim=1).view(batch_size, self.n_classes, -1)
#         target=torch.cat((1-target,target),1)
#         target=target.contiguous().view(batch_size, self.n_classes, -1)

#         inter = torch.sum(input * target , 2) + smooth
#         union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

#         score = torch.sum(2.0 * inter / union)
#         score = 1.0 - score / (float(batch_size) * float(self.n_classes))

#         return score

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=2,DSC_smoothing=0.001):
        super(SoftDiceLoss, self).__init__()
        #self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.DSC_smoothing=DSC_smoothing

    def forward(self, input, target):
        smooth = self.DSC_smoothing #0.0001
        #print(np.shape(input))
        batch_size = input.size(0)
        target.double()
        # valid_mask = target.ne(-1)
        # target = target.masked_fill_(~valid_mask, 0)
        
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        #target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        #target = F.softmax(target, dim=1).view(batch_size, self.n_classes, -1)
        target=torch.cat((1-target,target),1)
        target=target.contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target , 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class DiceSemimetricLoss(nn.Module):
    def __init__(self,num_organ=1):
        super(DiceSemimetricLoss, self).__init__()

        self.num_organ=num_organ
    def forward(self, pred_stage1, organ_target):
        """
        :param pred_stage1: (B, 9,  256, 256)
        :param pred_stage2: (B, 9, 256, 256)
        :param target: (B, 256, 256)
        :return: Dice
        
        """
        batch_size = pred_stage1.size(0)
        num_organ=self.num_organ
        pred_stage1 = F.softmax(pred_stage1, dim=1).view(batch_size,num_organ+1, -1)
        organ_target = organ_target.double().view(batch_size,num_organ, -1)
        smooth=1e-5
        
        
        # loss
        dice_stage1 = 0.0

        organ_index=1
        inter=abs(pred_stage1[:, organ_index,:]).sum()-abs(organ_target[:, organ_index,:]).sum()+(abs(pred_stage1[:, organ_index, :]-organ_target[:, organ_index, :])).sum()
        union=abs(pred_stage1[:, organ_index,:]).sum()+abs(organ_target[:, organ_index,:]).sum()
        
        # inter=(2*pred_stage1[:, organ_index,:] * organ_target[:, organ_index ,:]).sum()
        # union=(2*pred_stage1[:, organ_index, :] * organ_target[:, organ_index , :]).sum()+torch.sum(abs(pred_stage1[:, organ_index, :]-organ_target[:, organ_index, :]),1).sum()
        print(inter)
        print(union)
        
        dice_stage1+=(inter+smooth)/(union+smooth)
        # print('organ %i: %f ' %(organ_index,torch.sum((inter+smooth)/ (union+smooth))))
            
        dice_stage1 /= (num_organ)
        dice = dice_stage1 

        # 
        return (1 - dice).mean()











# class DiceSemimetricLoss(nn.Module):
#     def __init__(self,num_organ=1):
#         super(DiceSemimetricLoss, self).__init__()

#         self.num_organ=num_organ
#     def forward(self, input, target):
#         smooth = 1e-5
#         #print(np.shape(input))
#         batch_size = input.size(0)
        
#         input = F.softmax(input, dim=1).view(batch_size, self.num_organ+1, -1)
#         input=input[:,self.num_organ,:]

#         target=target.contiguous().view(batch_size, self.num_organ, -1)

#         dice_stage1=0.0
#         # inter=(input * target).sum(dim=1).sum(dim=1)
#         # union=(input * target).sum(dim=1).sum(dim=1)+torch.sum(abs(input-target),1).sum(dim=1)
       

#         inter=abs(input).sum()+abs(target).sum()-(abs(input-target)).sum()
#         union=abs(input).sum()+abs(target).sum()

#         # print(inter)
#         # print(union)
        
#         dice = torch.sum((inter+smooth)/ (union+smooth))

#         # 
#         return (1 - dice).mean()

class DiceLoss_test(nn.Module):
    def __init__(self,num_organ=1):
        super(DiceLoss_test, self).__init__()

        self.num_organ=num_organ
    def forward(self, pred_stage1, target):
        """
        :param pred_stage1: (B, 9,  256, 256)
        :param pred_stage2: (B, 9, 256, 256)
        :param target: (B, 256, 256)
        :return: Dice
        """
        pred_stage1 = F.softmax(pred_stage1, dim=1)
        num_organ=self.num_organ
        # 
        #[b,12,256,256]
        organ_target = torch.zeros((target.size(0), num_organ, imsize, imsize))
        #print (organ_target.size())
        #[0-11] 
        for organ_index in range(0, num_organ ):
            #print (organ_index)
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            #print (organ_target[:, organ_index, :, :].size())
            #print (temp_target.size())
            organ_target[:, organ_index, :, :] = torch.squeeze(temp_target)
            # organ_target: (B, 8,  128, 128)

        organ_target = organ_target.cuda()

        # loss
        dice_stage1 = 0.0

        for organ_index in range(0, num_organ ):
            dice_stage1 += 2 * (pred_stage1[:, organ_index, :, :] * organ_target[:, organ_index , :, :]).sum(dim=1).sum(
                dim=1) / (pred_stage1[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) +
                          organ_target[:, organ_index, :, :].pow(2).sum(dim=1).sum(dim=1) + 1e-5)

        
        dice_stage1 /= num_organ


        # 
        dice = dice_stage1 

        # 
        return (1 - dice).mean()

import numpy as np

class MRRN_Segmentor(BaseModel):
    def name(self):
        return 'MRRN_Segmentor'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain=opt.isTrain
        
        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size) # input A
        if(opt.model_type == 'classifier'):
           self.input_A_y = self.Tensor(nb, opt.output_nc, 1, 1)
        else:
           self.input_A_y = self.Tensor(nb, opt.output_nc, size, size) # input B
           
           
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
           
           
        self.input_A=self.input_A.to(device)
        self.input_A_y=self.input_A_y.to(device)
        if(opt.model_type == 'multi'):
          #self.input_A_z = self.Tensor(3,1)
          ## going to treat this as a pixel-wise output within the mask -- makes sense if it is regarding the pCR because the whole tumor part tumor may respond/ or for subtype
          self.input_A_z = self.Tensor(nb, opt.output_nc, size, size)
          self.input_A_z = self.input_A_z.to(device)

        self.test_A = self.Tensor(nb, opt.output_nc, size, size) # input B
        self.num_organ=1 #6#9+3 #/->6 ->8 

        self.hdicetest=DiceLoss_test()
        self.dicetest=SoftDiceLoss()
        self.ce_loss=nn.CrossEntropyLoss(weight=torch.tensor([0.001, 0.999],dtype=torch.float).to(device),label_smoothing=0.01)
        self.DML_loss=JDTLoss(alpha = 0.5, beta = 0.5)#DiceSemimetricLoss()
        #self.DML_loss=JDTLoss()
        #MRRN
        self.netSeg_A=networks.get_Incre_MRRN_deepsup(opt.nchannels,1,opt.init_type,self.gpu_ids, opt.deeplayer)
        #flops, params = get_model_complexity_info(self.netSeg_A, (256, 256), as_strings=True, print_per_layer_stat=True)
        #print ('params is ',params)
        #print ('flops is ',flops)
        #self.criterion = nn.CrossEntropyLoss()

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            if self.isTrain:
                self.load_network(self.netSeg_A,'Seg_A',which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.optimizer_Seg_A = torch.optim.AdamW(self.netSeg_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)
            if opt.optimizer == 'SGD':
                self.optimizer_Seg_A = torch.optim.SGD(self.netSeg_A.parameters(), lr=opt.lr, momentum=0.99)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_Seg_A)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        if self.isTrain:
            networks.print_network(self.netSeg_A)
        print('-----------------------------------------------')

    def set_test_input(self,input):
        input_A1=input[0]
        self.test_A,self.test_A_y=torch.split(input_A1, input_A1.size(0), dim=1)    
        
    def cross_entropy_2D(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        input = input.float()
        if 1 < 0:
           loss = nn.CrossEntropyLoss(F.log_softmax(input), target.long())
        else:
           log_p = F.log_softmax(input, dim=1)
           log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
           target = target.view(target.numel())
           target = target.long()
           loss = F.nll_loss(log_p, target, weight=weight, size_average=True)
        #if size_average:
        #    loss /= float(target.numel())
        return loss

    def cross_entropy_1D(self, input, target, weight=None, size_average=True):
        input = input.float()
        if 1 < 0:
           loss = nn.CrossEntropyLoss(F.log_softmax(input), target.long())
        else:
           log_p = F.log_softmax(input, dim=1)
           target = target.view(target.numel())
           target = target.long()
           loss = F.nll_loss(log_p, target, weight=weight, size_average=True)
        return loss


    def dice_loss(self,input, target,x):

        ##USE for CrossEntrophy
        CE_loss=0
        CE_ohm_loss=0
        DML_loss=0
        
        Soft_dsc_loss=0
        dice_test=0
        hdice_test=0
        dice_ce=0
        CE_loss = 0   
        if(self.opt.loss == 'dice_ce'):
           dice_ce = 1
        if(self.opt.loss == 'ce'):
           CE_loss = 1
        if(self.opt.loss == 'DML'):
           DML_loss = 1
        if(self.opt.loss == 'soft_dsc'):
           Soft_dsc_loss = 1
        if(self.opt.loss == 'dice'):
           dice_test = 1
        if(self.opt.loss == 'hdice'):
           hdice_test = 1
        if(self.opt.model_type == 'classification'):
           CE_loss = 1
           dice_ce = 0

        if CE_loss:    
            #print(input.size())
            # n, c, h, w = input.size()
            # input=input.float()
            # log_p = F.log_softmax(input,dim=1)
            # log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            
            # target=torch.stack((1-target,target),3)
            # target = target.view(-1, c)
            # #target=target.long()
            # loss = F.nll_loss(log_p, target, weight=None, size_average=True)
            #size_average=False
            #if size_average:
            #    loss /= float(target.numel())
            input=input.float()

            target=torch.stack((1.0-target,target),1).float()
            loss = self.ce_loss(input,target)
            
        elif Soft_dsc_loss:       
            loss=self.soft_dice_loss(input,target)                
        elif CE_ohm_loss:
            loss=self.CrossEntropy2d_Ohem(input,target)
        elif dice_test:
            loss=self.dicetest(input, target)    
        elif hdice_test:
            loss=self.hdicetest(input, target)    
        elif DML_loss:
            
            
            
            
            #loss=self.DML_loss(F.softmax(input,dim=1)[:,1,:], target)
            
            
            input=input.double()

            target=torch.stack((1.0-target,target),1).double()
            
            loss1=self.DML_loss(input, target)
            loss2 = self.ce_loss(input,target)
            loss=loss1*0.75+loss2*0.25
            #loss=loss1
            
        else: #dice_ce
           
            # loss1=self.dicetest(input.double(), target.double())  
            # input=input.double()

            # target=torch.stack((1.0-target,target),1).double()
            # loss2 = self.ce_loss(input,target)
            # loss=loss1*0.75+loss2*0.25
            # # n, c, h, w = input.size()
            # # input=input.float()
            # # log_p = F.log_softmax(input,dim=1)
            # # log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            # # target=torch.stack((1-target,target),3)
            # # target = target.view(-1, c)
            # # #target=target.long()
            # # loss2 = F.nll_loss(log_p, target, weight=None, size_average=True)
        
            
            #loss=0.75*loss1+0.25*loss2
            loss1=self.dicetest(input, target)   
            # n, c, h, w = input.size()
            # input=input.float()
            # log_p = F.log_softmax(input,dim=1)
            # log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            # target = target.view(target.numel())
            # target=target.long()
            # loss2 = F.nll_loss(log_p, target, weight=None, size_average=True)
            target=torch.stack((1.0-target,target),1).double()
            loss2=self.ce_loss(input,target)
            
            # #print("DICE/OHEM %0.4f/" % loss1,"%0.4f" % loss2)
            
            
            loss=0.75*loss1+0.25*loss2
            #loss=loss1
            
        return loss

    def get_curr_lr(self):
        self.cur_lr=self.optimizer_Seg_A.param_groups[0]['lr'] 

        return self.cur_lr
    
    def cal_dice_loss(self,pred_stage1, target):
        """
        :param pred_stage1: (B, 9,  256, 256)
        :param pred_stage2: (B, 9, 256, 256)
        :param target: (B, 256, 256)
        :return: Dice
        : HV note: These parameters are a relic -- don't make any sense
        """
        num_organ=pred_stage1.size(1)-1
        
        organ_target = torch.zeros((target.size(0), num_organ+1, imsize, imsize))  # 8+1
        pred_stage1=F.softmax(pred_stage1,dim=1)
        

        for organ_index in range(num_organ + 1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index,  :, :] = temp_target.reshape(temp_target.shape[0], imsize, imsize)

        organ_target = organ_target.cuda()
            # loss
        dice_0=0

        dice_stage1 = 0.0   
        smooth = 1.
        
        for organ_index in  range(num_organ + 1):
            pred_tep=pred_stage1[:, organ_index,  :, :] 
            target_tep=organ_target[:, organ_index,  :, :]
            
            pred_tep=pred_tep.contiguous().view(-1)
            target_tep=target_tep.contiguous().view(-1)
            intersection_tp = (pred_tep * target_tep).sum()
            dice_tp=(2. * intersection_tp + smooth)/(pred_tep.sum() + target_tep.sum() + smooth)
            
            if organ_index==0:
                dice_0=dice_tp
            
        return dice_0

    def set_test_input(self,input):
        self.test_A=input#torch.split(input_A1, input_A1.size(0), dim=1)    
     # def set_test_input(self,input,input_label):
     #    self.test_A,self.test_A_y=input, input_label#torch.split(input_A1, input_A1.size(0), dim=1)    
            

    def set_input(self, input):
        #AtoB = self.opt.which_direction == 'AtoB'  ## this is a misleading name -- a relic of using cycleGAN
        input_A1=input#[0]
        #input_A1=input_A1.view(-1,2,512,512)
        if(self.opt.model_type == 'multi'):
           input_A1=input_A1.view(-1,3,imsize,imsize)
           input_A11, input_A12, input_A13 = torch.split(input_A1.size(1)//3, dim=1)
           self.input_A.resize_(input_A11.size()).copy_(input_A11)
           self.input_A_y.resize_(input_A12.size()).copy_(input_A12)
           self.input_A_z.resize_(input_A13.size()).copy_(input_A13)
        else:
           input_A1=input_A1.view(-1,2,imsize,imsize)
           input_A11,input_A12=torch.split(input_A1, input_A1.size(1)//2, dim=1)
           self.input_A.resize_(input_A11.size()).copy_(input_A11)
           self.input_A_y.resize_(input_A12.size()).copy_(input_A12)
           
        self.image_paths = 'test'#input['A_paths' if AtoB else 'B_paths']

    def set_input_multi(self, input_x, input_y, input_z):
        #AtoB = self.opt.which_direction == 'AtoB'
        
        input_A11, input_A12, input_A13 = input_x, input_y, input_z
        self.input_A.resize_(input_A11.size()).copy_(input_A11)
        self.input_A_y.resize_(input_A12.size()).copy_(input_A12)
        self.input_A_z.resize_(input_A13.size()).copy_(input_A13)
        self.image_paths = 'test'#input['A_paths' if AtoB else 'B_paths']
        
    def set_input_sep(self, input_x,input_y):
        #AtoB = self.opt.which_direction == 'AtoB'

        input_A11,input_A12=input_x,input_y#torch.split(input_A1, input_A1.size(1)//2, dim=1)

        self.input_A.resize_(input_A11.size()).copy_(input_A11)


        self.input_A_y.resize_(input_A12.size()).copy_(input_A12)

        self.image_paths = 'test'#input['A_paths' if AtoB else 'B_paths']
    
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_A_y=Variable(self.input_A_y)
        if(self.opt.model_type == 'multi'):
           self.class_A=Variable(self.input_A_z)
        
    
    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        self.seg_A = self.netSeg_A(real_A).data

    def net_Classtest_image(self):
        
        self.test_A=self.test_A.cuda()
        self.test_A_y=self.test_A_y.cuda()
        test_img=self.test_A
        test_img = test_img.float()

        A_class = self.netSeg_A(test_img)
        A_class = torch.argmax(A_class, dim=1)
        A_class = A_class.view(1,1)
        A_class_out = A_class.data
        
        return self.test_A.cpu().float().numpy(),A_class_out.cpu().float().numpy()
    

    def net_Segtest_image(self):

        self.test_A=self.test_A.cuda()
        #self.test_A_y=self.test_A_y.cuda()
        test_img=self.test_A
        if(self.opt.model_type == 'deep'):
           _,A_AB_seg=self.netSeg_A(test_img)
        else:
           A_AB_seg=self.netSeg_A(test_img)
           
        #loss=self.dice_loss(A_AB_seg,self.test_A_y, test_img)
        
        A_AB_seg=F.softmax(A_AB_seg, dim=1)
        A_AB_seg=A_AB_seg[:,1,:,:]

        
        A_AB_seg=A_AB_seg.view(1,1, imsize, imsize)
        A_AB_seg_out=A_AB_seg.data

        A_AB_seg_out=A_AB_seg_out.view(1,1, imsize, imsize)

        #A_y_out=self.test_A_y.data
        #self.test_A_y=self.test_A_y.cuda()

        test_A_data=self.test_A.data
        A_AB_seg=A_AB_seg.data
        #A_y=self.test_A_y.data
        test_A_data,d999=self.tensor2im_jj(test_A_data)

        A_AB_seg=util.tensor2im_scaled(A_AB_seg)
        #A_y=util.tensor2im_scaled(A_y)

        test_A_data=test_A_data[:,imsize:imsize*2,:]

        A_AB_seg=A_AB_seg#[:,256:512,:]
        #A_y=A_y#[:,256:512,:]

        
        #image_numpy_all=np.concatenate((test_A_data,A_y,),axis=1)
        #image_numpy_all=np.concatenate((image_numpy_all,A_AB_seg,),axis=1)


        #return loss,self.test_A.cpu().float().numpy(),A_AB_seg_out.cpu().float().numpy(),A_y_out.cpu().float().numpy(),image_numpy_all
        #return self.test_A.cpu().float().numpy(),A_AB_seg_out.cpu().float().numpy(),image_numpy_all
        return self.test_A.cpu().float().numpy(),A_AB_seg_out.cpu().float().numpy()
    
    def get_image_paths(self):
        return self.image_paths
    
    def cal_seg_loss (self,netSeg,pred,gt):
        img=pred
        lmd = 1
        if(self.opt.model_type == 'deep'):
           out1, self.pred = netSeg(pred)
           seg_loss = lmd*(self.dice_loss(self.pred, gt, img)*self.opt.out_wt + (1. - self.opt.out_wt)*self.dice_loss(out1, gt, img))
        else:  
           #print('Input Shape Unet: ', self.pred.shape())
           self.pred=netSeg(pred)
           seg_loss=lmd*self.dice_loss(self.pred,gt,img)
        return seg_loss


    def backward_Seg_A(self):
        gt_A=self.real_A_y # gt 
        img_A=self.real_A # gt
        
        seg_loss = self.cal_seg_loss(self.netSeg_A, img_A, gt_A)
        if(self.opt.model_type == 'deep'):
           _,out = self.netSeg_A(img_A)
           d0 = self.cal_dice_loss(out, gt_A)
        else:
           d0 = self.cal_dice_loss(self.netSeg_A(img_A), gt_A)
        self.d0 = d0.item()
        self.seg_loss = seg_loss
        seg_loss.backward()

    def load_MR_seg_A(self, weight):
        self.load_network(self.netSeg_A,'Seg_A',weight)

def optimize_parameters(self, accumulate=False):
    """
    Calculate losses, gradients, and update network weights.
    
    Parameters:
        accumulate (bool): If True, accumulate gradients instead of updating weights immediately
    """
    # forward
    self.forward()
    
    # Only zero gradients if we're not accumulating
    if not accumulate:
        self.optimizer_Seg_A.zero_grad()

    self.backward_Seg_A()
    
    # Only update weights if we're not accumulating
    if not accumulate:
        self.optimizer_Seg_A.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([('Seg_loss',  self.seg_loss), ('d0', self.d0)])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        real_Ay=util.tensor2im_scaled(self.input_A_y)
        if(self.opt.model_type == 'deep'):
          _,pred_A = self.netSeg_A(self.input_A)
        else:
          pred_A=self.netSeg_A(self.input_A)
        pred_A=F.softmax(pred_A, dim=1)

        pred_A=torch.argmax(pred_A, dim=1)
        pred_A=pred_A.view(self.input_A.size()[0],1,imsize, imsize)
        pred_A=pred_A.data

        seg_A=util.tensor2im_scaled(pred_A) #

        ret_visuals = OrderedDict([('real_A', real_A),('real_A_GT_seg',real_Ay),('real_A_seg', seg_A)])
        return ret_visuals

    def get_current_seg(self):
        ret_visuals = OrderedDict([('d0', self.d0),])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netSeg_A, 'Seg_A', label, self.gpu_ids)
    
    def tensor2im_jj(self,image_tensor):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy_tep=image_numpy
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        elif (image_numpy.shape[0] != 3):
            k=floor(image_numpy.shape[0]/(2))
            image_numpy = np.tile(image_numpy[k,:,:], (3, 1, 1))

        self.test_A_tep = self.test_A[0].cpu().float().numpy()
        if self.test_A_tep.shape[0] == 1:
            self.test_A_tep = np.tile(self.test_A_tep, (3, 1, 1))
        elif (self.test_A_tep.shape[0] != 3):
            k=floor(self.test_A_tep.shape[0]/(2))
            self.test_A_tep = np.tile(self.test_A_tep[k,:,:], (3, 1, 1))
            
        image_numpy_all=np.concatenate((self.test_A_tep,image_numpy,),axis=2)
        image_numpy_all = (np.transpose(image_numpy_all, (1, 2, 0)) + 1) / 2.0 * 255.0    

        return image_numpy_all.astype(np.uint8),image_numpy_tep

    def tensor2im_jj_3(self,image_tensor):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy_tep=image_numpy
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        self.test_A_tep = self.test_A[0].cpu().float().numpy()
        if self.test_A_tep.shape[0] == 1:
            self.test_A_tep = np.tile(self.test_A_tep, (3, 1, 1))

        image_numpy_all=np.concatenate((self.test_A_tep,image_numpy,),axis=2)
        image_numpy_all = (np.transpose(image_numpy_all, (1, 2, 0)) + 1) / 2.0 * 255.0        
        

        return image_numpy_all.astype(np.uint8),image_numpy_tep
