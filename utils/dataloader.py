import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
import torch
from PIL import ImageEnhance

random.seed(2021)


# several data augumentation strategies
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label


# def getFlip(self):
#     p = random.randint(0, 1)
#     self.flip = transforms.RandomHorizontalFlip(p)


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)

    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]


        self.images = sorted(self.images)
        self.gts = sorted(self.gts)


        self.filter_files()
        self.size = len(self.images)
        self.kernel = np.ones((5, 5), np.uint8)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.ge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])



        self.kernel_e = np.ones((3, 3), np.uint8)
        self.kernel_t = np.ones((3, 3), np.uint8)
        self.size = len(self.images)

    def getFlip(self):
        p = random.randint(0, 1)
        self.flip = transforms.RandomHorizontalFlip(p)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])


        image, gt = cv_random_flip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.ge_transform(gt)



        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []


        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)

            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)

        self.images = images
        self.gts = gts


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class test_dataset:
    """load test dataset (batchsize=1)"""

    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


# solve dataloader random bug
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


class test_loader_faster(data.Dataset):
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)

    def __getitem__(self, index):
        images = self.rgb_loader(self.images[index])
        images = self.transform(images)

        img_name_list = self.images[index]

        return images, img_name_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
