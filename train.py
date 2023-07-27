from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data.semantic_data_pl import  SegmentationDataModule
from config import *
from pl_trainer import BisenetTrainer
from pytorch_lightning import loggers
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator",default = 'gpu')
    parser.add_argument("--devices",default = 2 )
    parser.add_argument("--precision",default = 32 )
    args = parser.parse_args()
    # pl data module
    train_data_list = TRAIN_DATA_DIRS
    val_data_list = VAL_DATA_DIRS
    dm = SegmentationDataModule(train_data_list,val_data_list)

    # pl model
    system = BisenetTrainer()
       
    # pl logger
    logger = loggers.TensorBoardLogger(save_dir="ckpts")
    #logger = loggers.WandbLogger()
    #logger = loggers.CometLogger()
    #logger = loggers.MLFlowLogger()
    #logger = loggers.NeptuneLogger()
    
    # stop early
    es = EarlyStopping(monitor = "val_loss")
    
    
    # save checkpoints
    ckpt_dir = "ckpts/{}/version_{:d}".format("bisenet_pl",logger.version)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename="{epoch}-{val_loss:.4f}-{eval_mious:.4f}",
                                          monitor="eval_mious",
                                          #monitor="mIoU",
                                          mode="max",
                                          save_last=True,
                                          save_weights_only=True,
                                          save_top_k=3)
    # restore from previous checkpoints 
    if PRETRAINED_MODEL is not None:
        print("load pre-trained model from {}".format(PRETRAINED_MODEL))
        system.load_from_checkpoint(PRETRAINED_MODEL,strict=False)
    
    # set up trainer
    if HALF_PRECISION:
        trainer = Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=EPOCHS,
            num_sanity_val_steps=4,
            callbacks=[checkpoint_callback],
            logger=logger,
            benchmark=True,
            precision=args.precision
        )
    else:
        trainer = Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=EPOCHS,
            num_sanity_val_steps=4,
            callbacks=[checkpoint_callback],
            logger=logger,
            benchmark=True,  
            strategy="ddp_find_unused_parameters_false"
        )
        
    
    trainer.fit(system,dm)

    

    
    
    