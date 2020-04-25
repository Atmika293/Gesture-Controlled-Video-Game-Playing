import os
from os.path import isdir, exists, abspath, join
import random
import numpy as np

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2

from torchvision import transforms
import torch

from scipy.interpolate import UnivariateSpline

class WarmingFilter:
    def __init__(self):
        pass      

    def create_LUT(self, x, y):
        spline = UnivariateSpline(x, y)
        return spline(range(256))

    def generate_LUTs(self):
        x = [0, 64, 128, 192, 255]

        y = [0, random.randint(70, 100), random.randint(140, 170), random.randint(210, 240), 255]
        warming_lut = self.create_LUT(x, y)

        y = [0, random.randint(30, 60), random.randint(80, 110), random.randint(120, 150), 192]
        cooling_lut = self.create_LUT(x, y)

        return warming_lut, cooling_lut

    def __call__(self, image):
        b, g, r = cv2.split(image)
        warming_lut, cooling_lut = self.generate_LUTs()
        r = np.clip(cv2.LUT(r, warming_lut), 0, 255).astype(np.uint8)
        b = np.clip(cv2.LUT(b, cooling_lut), 0, 255).astype(np.uint8)

        warm_image = cv2.merge((b, g, r))

        h, s, v = cv2.split(cv2.cvtColor(warm_image, cv2.COLOR_BGR2HSV))
        s = np.clip(cv2.LUT(s, warming_lut), 0, 255).astype(np.uint8)
        warm_image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

        return warm_image

class DataLoader():
    def __init__(self, root_dir, batch_size, input_width=240, input_height=240, mode='train', split=0.1): 
        self.batch_size = batch_size
        self.mode = mode

        self.root_dir = abspath(root_dir)
        self.img_dirs = os.listdir(join(self.root_dir, '_LABELLED_SAMPLES'))
        self.label_dirs = os.listdir(join(self.root_dir, 'masks'))
        
        self.img_dirs.remove('hand_over_face')
        self.label_dirs.remove('hand_over_face')
        self.img_dirs.sort()
        self.label_dirs.sort()
        self.img_dirs.insert(0, 'hand_over_face')
        self.label_dirs.insert(0, 'hand_over_face')

        for img_dir, label_dir in zip(self.img_dirs, self.label_dirs):
            if img_dir != label_dir:
                print(img_dir, label_dir)

        self.img_dirs = [join(join(self.root_dir, '_LABELLED_SAMPLES'), d) for d in self.img_dirs]
        self.label_dirs = [join(join(self.root_dir, 'masks'), d) for d in self.label_dirs]

        self.data_files = []
        self.label_files = []
        for i in range(len(self.img_dirs)):
            for filename in os.listdir(self.img_dirs[i]):
                if filename.endswith('.jpg'):
                    self.data_files.append(join(self.img_dirs[i], filename))
                    self.label_files.append(join(self.label_dirs[i], filename))

        self.shuffle_data()

        eval_size = int(split * len(self.data_files))
        self.eval_files = self.data_files[-eval_size:]
        self.eval_labels = self.label_files[-eval_size:]

        self.data_files = self.data_files[:-eval_size]
        self.label_files = self.label_files[:-eval_size]

        self.test_files = []
        self.test_folder = join(self.root_dir, 'frames')#abspath('../../743_gestures/frames')
        self.test_dirs = os.listdir(self.test_folder)
        self.test_dirs = [join(self.test_folder, d) for d in self.test_dirs]
        for i in range(len(self.test_dirs)):
            for filename in os.listdir(self.test_dirs[i]):
                if filename.endswith('.jpg'):
                    self.test_files.append(join(self.test_dirs[i], filename))

        self.normalize = transforms.ToTensor()

        self.input_width = input_width
        self.input_height = input_height

        self.warming_transform = WarmingFilter()


    def __iter__(self):
        data = []
        labels = []
        if self.mode == 'train':
            data = self.data_files
            labels = self.label_files
        elif self.mode == 'eval':
            data = self.eval_files
            labels = self.eval_labels
        elif self.mode == 'test':
            data = self.test_files

        data_size = len(data)

        if self.mode == 'test':
            input_batch = torch.zeros([1, 3, self.input_height, self.input_width], dtype=torch.float32)
        else:
            input_batch = torch.zeros([self.batch_size, 3, self.input_height, self.input_width], dtype=torch.float32)
            target_batch = torch.zeros([self.batch_size, 1, self.input_height, self.input_width], dtype=torch.float32)        

        if self.mode == 'test':
            current = 0
            while current < data_size:
                data_image_orig = cv2.imread(data[current])
                data_image_orig = cv2.resize(data_image_orig, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                input_batch[0, :, :, :] = self.normalize(data_image_orig)

                yield input_batch, data[current]
                current += 1
        else:
            current = 0
            while current < data_size:
                count = 0
                while count < self.batch_size and current < data_size:
                    # print(data[current])
                    # print(labels[current])
                    data_image_orig = cv2.imread(data[current])
                    label_image_orig = cv2.imread(labels[current], cv2.IMREAD_GRAYSCALE)
                    
                    # Resizing
                    data_image_orig = cv2.resize(data_image_orig, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                    # To crop change to 572 and un comment next line
                    # To not crop 388 (check assignment chart again)
                    #label_image_orig = label_image_orig.resize((388,388)) 
                    label_image_orig = cv2.resize(label_image_orig, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                    _, label_image_orig = cv2.threshold(label_image_orig, 127, 255, cv2.THRESH_BINARY)        
                    
                    ## AUGMENTATION ##
                    # img_size = np.shape(label_image_orig)
                    # segmap = np.zeros(img_size, dtype=np.uint8)
                    # segmap[:] = label_image_orig
                    # segmap = SegmentationMapOnImage(segmap, shape=img_size)
                    segmap = SegmentationMapsOnImage(label_image_orig, shape=np.shape(label_image_orig))
                    
                    # Augementation pipeline
                    # pipeline = iaa.Sometimes(
                    #                     0.7,
                    pipeline =  iaa.OneOf([
                                    iaa.Affine(scale=(0.5, 1.5)),
                                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                                    iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
                                    iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add((0, 50)))),
                                    iaa.ChangeColorTemperature((1100, 10000)),
                                    iaa.GammaContrast((0.5, 2.0))
                                ])
                                    # )
                    
                    if random.random() > 0.3:
                        if random.random() > 0.4:
                            data_image_aug, label_image_aug = pipeline(image = data_image_orig, segmentation_maps=segmap)
                            label_image_aug = label_image_aug.get_arr()
                        else:
                            data_image_aug = self.warming_transform(data_image_orig)
                            label_image_aug = label_image_orig
                    else:
                        data_image_aug = data_image_orig
                        label_image_aug = label_image_orig

                    # data_image_aug = data_image_aug.transpose((2, 0, 1))
                    # label_image_aug = np.expand_dims(label_image_aug.get_arr(), axis=0).astype('uint8')
                    input_batch[count, :, :, :] = self.normalize(data_image_aug)

                    # label_image_aug = label_image_aug.get_arr() // 255
                    # label_image_aug = label_image_aug.astype('uint8')
                    # target_batch[count, :, :] = torch.from_numpy(label_image_aug).long()

                    label_image_aug = np.expand_dims(label_image_aug.astype(np.float32) / 255.0, axis=0)
                    target_batch[count, :, :, :] = torch.from_numpy(label_image_aug)

                    count += 1
                    current += 1
                   
                yield input_batch, target_batch

    def setMode(self, mode):
        self.mode = mode

    def shuffle_data(self):
        train_list = list(zip(self.data_files, self.label_files))
        random.shuffle(train_list)
        train_list = [list(l) for l in list(zip(*train_list))]
        self.data_files[:] = train_list[0][:]
        self.label_files[:] = train_list[1][:]

    def shuffle_eval(self):
        eval_list = list(zip(self.eval_files, self.eval_labels))
        random.shuffle(eval_list)
        eval_list = [list(l) for l in list(zip(*eval_list))]
        self.eval_files[:] = eval_list[0][:]
        self.eval_labels[:] = eval_list[1][:]