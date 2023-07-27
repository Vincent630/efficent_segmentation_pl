import math
import random
import cv2 as cv
import numpy as np
from copy import deepcopy
import torchvision.transforms as transforms
from PIL import Image
import torch

cv.setNumThreads(0)

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, img, mask):
        tensors = []
        # put it from HWC to CHW format
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        img = torch.from_numpy(img).float()/255
        mask = torch.from_numpy(mask).float()
        return img, mask


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
 
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(8, 10, 15)):
    # def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
 
    def __call__(self, img,mask):

        if random.uniform(0, 1) > self.probability:
            return img,mask
        else:
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w <  img.shape[1] and h <  img.shape[0]:
                x1 = random.randint(0,  img.shape[0] - h)
                y1 = random.randint(0,  img.shape[1] - w)

                img[ x1:x1 + h, y1:y1 + w,0] = self.mean[0]
                img[ x1:x1 + h, y1:y1 + w,1] = self.mean[1]
                img[ x1:x1 + h, y1:y1 + w,2] = self.mean[2]
                if len( mask.shape) == 2:
                     mask[x1:x1 + h, y1:y1 + w] = 0
                return img,mask
            else:
                return img,mask
        
        

class Identity(object):
    def __init__(self, **kwargs):
        kwargs['p'] = 1.0
        super(Identity, self).__init__(**kwargs)

    def __call__(self,img,mask):    
        return img,mask

class Resize(object):
    def __init__(self, w = 448, h = 256):
        self.w = w
        self.h = h
    def __call__(self,img,mask):          
        img = cv.resize( img, (self.w, self.h), interpolation=cv.INTER_NEAREST)
        mask = cv.resize( mask, (self.w, self.h), interpolation=cv.INTER_NEAREST)
        return img,mask

class RandNoise(object):
    def __init__(self,probability=0.5):
        self.probability = probability
        
    @staticmethod
    def aug(img):
        mu = 0
        sigma = np.random.uniform(0, 5)
        ret_img = img + np.random.normal(mu, sigma, img.shape)
        ret_img = ret_img.clip(0., 255.).astype(np.uint8)
        return ret_img
        
    def __call__(self,img,mask): 
        if random.uniform(0, 1) > self.probability :
            img = self.aug(img)

        return img,mask

class RandBlur(object):
    def __init__(self,probability=0.5):
        self.probability = probability
        
    @staticmethod
    def gaussian_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img

    @staticmethod
    def median_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.medianBlur(img, kernel_size, 0)
        return img

    @staticmethod
    def blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.blur(img, (kernel_size, kernel_size))
        return img

    def aug(self, img: np.ndarray) -> np.ndarray:
        aug_blur = np.random.choice([self.gaussian_blur, self.median_blur, self.blur])
        img = aug_blur(img)
        return img

    def __call__(self,img,mask):  
        if random.uniform(0, 1) > self.probability:
            img = self. aug(img)
        return img,mask


class RandHSV(object):

    def __init__(self, hgain=0.4, sgain=0.8, vgain=0.36,probability = 0.5):
        self.probability = probability
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def aug(self, img):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        img = img.astype(np.uint8)
        hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
        ret_img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        return ret_img

    def __call__(self,img,mask): 
        if random.uniform(0, 1) > self.probability:
            img = self.aug( img)
        return img,mask

class ColorJitter(object):
    def __init__(self, 
                 brightness=(0.8, 1.2), 
                 contrast=(0.8, 1.2), 
                 saturation=(0.8, 1.2),
                 hue = (-0.3, 0.3),
                 probability = 0.5):
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.3, 0.3)
        self.probability = probability
        self.color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)

    def __call__(self, img,mask) :
        img = Image.fromarray(np.uint8(img))
        if random.uniform(0,1) > self.probability :
            img = self.color_aug(img)
        img = np.array(img)
        return img,mask

class RandPerspective(object):
    def __init__(self,
                 target_size=None,
                 degree=(-5, 5),
                 translate=0.1,
                 scale=(0.8, 1.2),
                 shear=0,
                 perspective=0.0,
                 probability = 0.5):

        assert isinstance(target_size, tuple) or target_size is None
        assert isinstance(degree, tuple)
        assert isinstance(scale, tuple)
        self.target_size = target_size
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.probability = probability

    def get_transform_matrix(self, img):
        if self.target_size is not None:
            width, height = self.target_size
        else:
            height, width = img.shape[:2]

        matrix_c = np.eye(3)
        matrix_c[0, 2] = -img.shape[1] / 2
        matrix_c[1, 2] = -img.shape[0] / 2

        matrix_p = np.eye(3)
        matrix_p[2, 0] = random.uniform(-self.perspective, self.perspective)
        matrix_p[2, 1] = random.uniform(-self.perspective, self.perspective)

        matrix_r = np.eye(3)
        angle = np.random.uniform(self.degree[0], self.degree[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        matrix_r[:2] = cv.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

        matrix_t = np.eye(3)
        matrix_t[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        matrix_t[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        matrix_s = np.eye(3)
        matrix_s[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        matrix_s[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        return matrix_t @ matrix_s @ matrix_r @ matrix_p @ matrix_c, width, height, scale

    def aug(self,img,mask):
        transform_matrix, width, height, scale = self.get_transform_matrix( img)
        if self.perspective:
            img = cv.warpPerspective( img,
                                              transform_matrix,
                                              dsize=(width, height),
                                              borderValue=(0, 0, 0))
        else:  # affine
            img = cv.warpAffine( img,
                                         transform_matrix[:2],
                                         dsize=(width, height),
                                         borderValue=(0, 0, 0))
        if self.perspective:
            mask = cv.warpPerspective( mask,
                                                transform_matrix,
                                                dsize=(width, height),
                                                flags=cv.INTER_NEAREST,
                                                borderValue=(0, 0, 0))
        else:  # affine
            mask = cv.warpAffine( mask,
                                           transform_matrix[:2],
                                           dsize=(width, height),
                                           flags=cv.INTER_NEAREST,
                                           borderValue=(0, 0, 0))
        return img,mask
    
    def __call__(self,img,mask):
        if random.uniform(0, 1) > self.probability:
            img ,mask = self.aug(img,mask)
        return img,mask

class LRFlip(object):
    def __init__(self,probability = 0.5):
        self.probability = probability

    @staticmethod
    def aug(img: np.ndarray) -> np.ndarray:
        img = np.fliplr(img)
        return img

    def __call__(self,img,mask): 
        _, w =  img.shape[:2]
        if random.uniform(0,1) > self.probability:
            img = self.aug( img)
            mask = self.aug(mask)
        return img,mask
    
class RandomCrop(object):
    def __init__(self, probability = 0.5):
        self.probability  = probability
        
    def aug(self,img,mask): 
        scale = np.random.uniform(0.3, 1)
        img_h, img_w =  img.shape[0],  img.shape[1] 
        label_h, label_w =  mask.shape[0],  mask.shape[1] 
        if (img_h!=label_h or img_w!=label_w):
            print("mask shape unequal img: ",img_h," ",img_w,"mask: ",label_h," ",label_w)
        height, width = int(img_h*scale), int(img_w*scale)
        x = np.random.randint(0, img_w-width)
        y = np.random.randint(0, img_h-height)
        img =  img[y:y+height, x:x+width]       
        mask  =  mask[y:y+height, x:x+width]    
        return img,mask
    
    def __call__(self,img,mask): 
        if random.uniform(0,1) > self.probability:
            img,mask = self.aug(img,mask)
        return img,mask


# class RandomFlip(object):
# up_down flip break ground-background constrain
#     def __call__(self,img,mask): 
#         _, w =  img.shape[:2]
#         flip_code = random.randint(-1, 1)
#         img = cv.flip(img,flip_code)
#         mask = cv.flip(mask,flip_code)
#         return img,mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self,img,mask):
        for transform in self.transforms:
            img,mask = transform(img,mask)
        return img,mask


if "__main__" == __name__:
    img_path= r"D:\7cls_test\test\img\2c30d5d2b4915bee2c53a34924b8a2a_f2a0b112a88341c6.jpg"
    mask_path = r"D:\7cls_test\test\mask_2cls_material\2c30d5d2b4915bee2c53a34924b8a2a_f2a0b112a88341c6.png"
    img = cv.imread(img_path)
    mask = cv.imread(mask_path,cv.IMREAD_GRAYSCALE)
    transform = [RandBlur(),
                 ColorJitter(),
                 RandomCrop(),
                 RandNoise(),
                 RandHSV(),
                 RandomErasing(),
                 LRFlip(),
                 RandPerspective(),
                 Resize(448,256) ]
    # transform = RandomErasing()
    compose_transforms = Compose(transform)
    trans_img,trans_maks = compose_transforms(img,mask)
    # trans_img,trans_maks = transform(img,mask)
    # cv.imwrite("trans_img.jpg",trans_img)
    # cv.imwrite("trans_maks.png",trans_maks*60)
    
    