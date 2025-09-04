#coding=utf-8
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms
from configures import parser
args = parser.parse_args()
import random
import cv2
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp
def Compute_Precision(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fn = np.sum(predict == 1)
    return tp,fn
def Compute_Recall(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(label==1)
    return tp, fp


def nearest_interpolation(lr_feature_high, fake_image):
    scale = fake_image.size(2)//lr_feature_high.size(2)
    batch_size = lr_feature_high.size(0)
    channels = lr_feature_high.size(1)

    tmp_feature = fake_image
    for m in range(batch_size):
        for n in range(channels):
            new_lr_feature_high = lr_feature_high[m][n].unsqueeze(0).unsqueeze(0)
            a = torch.nn.functional.interpolate(new_lr_feature_high, scale_factor = scale,  mode='nearest', align_corners=None)
            tmp_feature[m][n] = a.squeeze()
    return tmp_feature


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class LoadDatasetFromFolder(Dataset):
    def __init__(self, args, hr1_path, lr2_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder, self).__init__()
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.lr2_filenames = [join(lr2_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize= False)
        self.label_transform = get_transform()

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        lr2_img = self.transform(Image.open(self.lr2_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))

        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return hr1_img, lr2_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)

class LoadDatasetFromFolder_CD(Dataset):
    def __init__(self, args, hr1_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder_CD, self).__init__()
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]
        datalist2 = [name for name in os.listdir(lab_path) for item in args.suffix if
                    os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist2 if is_image_file(x)]
        ##########
        self.file_names = datalist
        ###########
        self.transform = get_transform(convert=True, normalize= True)
        self.label_transform = get_transform()

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))
        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 3).squeeze(0)

        return hr1_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)


class LoadDatasetFromFolder_CD_inverse(Dataset):
    def __init__(self, args, hr1_path, hr2_path, lab_path1,lab_path2):
        super(LoadDatasetFromFolder_CD_inverse, self).__init__()
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames1 = [join(lab_path1, x) for x in datalist if is_image_file(x)]
        self.lab_filenames2 = [join(lab_path2, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize= True)
        self.label_transform = get_transform()

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))
        label1 = self.label_transform(Image.open(self.lab_filenames1[index]))
        label1 = make_one_hot(label1.unsqueeze(0).long(), 3).squeeze(0)

        label2 = self.label_transform(Image.open(self.lab_filenames2[index]))
        label2 = make_one_hot(label2.unsqueeze(0).long(), 3).squeeze(0)

        return hr1_img, hr2_img, label1, label2

    def __len__(self):
        return len(self.hr1_filenames)

class LoadDatasetFromFolder_CD_sam(Dataset):
    def __init__(self, args, hr1_path, hr2_path, lab_path, point_num=1, mask_num=5,image_size=256):
        super(LoadDatasetFromFolder_CD_sam, self).__init__()
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                    os.path.splitext(name)[1] == item]
        datalist2 = [name for name in os.listdir(lab_path) for item in args.suffix if
                     os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist2 if is_image_file(x)]
        ##########
        self.file_names = datalist
        ###########
        self.transform = get_transform(convert=True, normalize=True)
        self.label_transform = get_transform()
        self.point_num = point_num
        self.mask_num = mask_num
        self.image_size = image_size

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))
        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 3).squeeze(0)

        h, w, _ = hr1_img.shape
        transforms = train_transforms(self.image_size, h, w)

        boxes_list = []
        point_coords_list, point_labels_list = [], []

        mask_path =self.lab_filenames[index]

        for m in [mask_path]:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            img1 = hr1_img.permute(1, 2, 0).numpy()
            augments = transforms(image=img1, mask=pre_mask)
            image_tensor,mask_tensor = augments['image'], augments['mask'].to(torch.int64)


            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        return hr1_img, hr2_img, label, boxes, point_coords, point_labels

    def __len__(self):
        return len(self.hr1_filenames)


if __name__ == "__main__":
    train_set = LoadDatasetFromFolder_CD_sam(args, args.hr1_train, args.hr2_train, args.lab_train)
    print("Dataset:", len(train_set))
