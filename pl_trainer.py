from model import *
from torch.optim.sgd import SGD
import pytorch_lightning as pl
from config import *
# import torch.optim.adam import Adam
from losses.seg_loss import OHMECE
from utils.seg_metric import SegmentationMetric
from model.bisenet_custom import MobileBisenet
from torch.optim.lr_scheduler import OneCycleLR
import torch

class BisenetTrainer(pl.LightningModule):
    def __init__(self) -> None:
        super(BisenetTrainer,self).__init__()
        self.save_hyperparameters()
        self.model = MobileBisenet()
        self.creterion1 = OHMECE()
        self.creterion2 = OHMECE()
    #def forward(self,x):
    #    predict = self.model(x,False)
    #    return predict
    
    def configure_optimizers(self):
        optimizer = SGD( self.model.parameters(),lr = INIT_LR ,momentum= MOMENTUM,nesterov=True)
        
        scheduler = OneCycleLR(optimizer, 
                               max_lr = 0.03,
                               epochs=EPOCHS,
                               steps_per_epoch=2000,
                               pct_start=0.01, 
                               div_factor=100, 
                               final_div_factor=10)
        return {"optimizer": optimizer,
                "lr_scheduler" :
                    {"scheduler":scheduler,
                    "interval": "step"}
                    }
        # optimizer = torch.optim.Adam(self.model.parameters())
        # return optimizer
    def training_step(self,batch,batch_idx):
        img,mask,names = batch
        #print(self.model.requires_grad)
        logits1,logits2,logits3,edge_out = self.model(img)

        loss_1,loss_2,loss_3,loss_edge = None,None,None,None
        loss_1 = self.creterion1(logits=logits1,labels=mask)
        if INTER_C4:
            loss_2 = 0.68 * self.creterion1(logits = logits2, labels = mask)
        if INTER_C5:
            loss_3 = 0.32 * self.creterion1(logits = logits3, labels = mask)
        if EDGE_OUT:
            loss_edge = self.creterion2(edge_out,mask)
        loss = loss_1 + loss_2 + loss_3 + loss_edge
        
        self.log("loss",loss)
        self.log("loss_1",loss_1)
        self.log("loss_2",loss_2)
        self.log("loss_3",loss_3)
        self.log("loss_edge",loss_edge)

        num_class = N_CLS
        train_pred = logits1.argmax(dim=1, keepdim=True)
        mn,mc,mh,mw = mask.shape
        mious = torch.sum(train_pred == mask)/(mn*mc*mw*mh)
        
        metric = SegmentationMetric(num_class) # 3表示有3个分类，有几个分类就填几
        metric.addBatch(train_pred.detach().cpu().numpy(), mask.detach().cpu().numpy())
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        #mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        self.log("pa",pa)
        # self.log("cpa",cpa)
        # self.log("mpa",mpa)
        self.log("mIoU",mIoU)       
        self.log("train_mious",mious)    
        # print('pa is : %f' % pa)
        # print('cpa is :') # 列表
        # print(cpa)
        # print('mpa is : %f' % mpa)
        # print('mIoU is : %f' % mIoU)
        return loss
    def validation_step(self,batch,batch_idx):
        
        model_eval = self.model.eval()
        img,mask,names = batch
        logits = model_eval(img,False)

        num_class = N_CLS
        eval_pred = logits.argmax(dim=1, keepdim=True)
        metric = SegmentationMetric(num_class) # 3表示有3个分类，有几个分类就填几
        metric.addBatch(eval_pred.detach().cpu().numpy(), mask.detach().cpu().numpy())
        eva_pa = metric.pixelAccuracy()
        eval_cpa = metric.classPixelAccuracy()
        eval_mIoU = metric.meanIntersectionOverUnion()
        loss = self.creterion1(logits,mask)
        mn,mc,mh,mw = mask.shape
        eval_miou = torch.sum(eval_pred == mask)/(mn*mc*mw*mh)

        
        self.log("val_loss",loss,sync_dist=True)
        self.log("eva_pa",eva_pa,sync_dist=True)
        self.log("eval_mIoU",eval_mIoU,sync_dist=True) 
        self.log("eval_mious",eval_miou,sync_dist=True,prog_bar=True) 
        return loss
        
        
        


            
        
        
        
    
        
        
        