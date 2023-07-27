from utils import seg_transform  


#path list contain two folder :img/*.jpg and mask/*.png
TRAIN_DATA_DIRS = [
"/path/to/train/folder/"


]
VAL_DATA_DIRS = [
"/path/to/test/folder/"
]



N_CLS = number_class
CLS_NAMES = ['first_cate','second_cate','third_cate',..."n_th_cate"]

BATCH_SIZE = 80
NUM_WORKERS = 4

EDGE_OUT = True
INTER_C4 = True
INTER_C5 = True
INNER_CHANNEL = 16

INIT_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

EPOCHS = 300
WARMUP_EPOCH = 5
WARM_UP_FACTOR = 0.01
COSINE_DECAY_RATE = 0.1

# bk_ground:0; ceramic:1; wood:2;
SAVE_NAME = "model_name"

SAMPLE = {
    2: 600,
    4: 1000
}
EMA = True
DEVICE = 0
PRETRAINED_MODEL = None
HALF_PRECISION = False
