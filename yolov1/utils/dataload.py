import os
from xml.dom.minidom import parse

import numpy as np
import pandas as pd
from PIL import Image
from pandas import read_csv
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import *
from tqdm import tqdm


def readvocxml(xml_path, image_dir):
    """ 
    Args:
    xml_path:singal xml file's path.
    image_dir:the image's location dir that xml file indicates.
    """
    tree = parse(xml_path)
    rootnode = tree.documentElement
    sizenode = rootnode.getElementsByTagName('size')[0]
    width = int(sizenode.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(sizenode.getElementsByTagName('height')[0].childNodes[0].data)
    depth = int(sizenode.getElementsByTagName('depth')[0].childNodes[0].data)

    name_node = rootnode.getElementsByTagName('filename')[0]
    filename = name_node.childNodes[0].data
    path = image_dir + '\\' + filename

    objects = rootnode.getElementsByTagName('object')
    objects_info = []
    for object in objects:
        label = object.getElementsByTagName('name')[0].childNodes[0].data
        xmin = int(float(object.getElementsByTagName('xmin')[0].childNodes[0].data))
        ymin = int(float(object.getElementsByTagName('ymin')[0].childNodes[0].data))
        xmax = int(float(object.getElementsByTagName('xmax')[0].childNodes[0].data))
        ymax = int(float(object.getElementsByTagName('ymax')[0].childNodes[0].data))
        info = []
        info.append(label)
        info.append(xmin)
        info.append(ymin)
        info.append(xmax)
        info.append(ymax)
        objects_info.append(info)

    return [filename, path, depth, height, width, objects_info]


def convert_bbox2labels(bboxes):
    """
    :param bboxes:(N,5)的bbox信息列表
    :return:(30,7,7)的yolov1格式的label,需要将(cls_index,dx,dy,dw,dh)转换成(cx,cy,dw,dh,confidence,cx,cy,dw,dh,confidence,....)
    tips:(30,7,7) = (info_dim,x,y)
    """
    grid_size = 1 / 7.0
    label = np.zeros(shape=(30, 7, 7))
    # 遍历每一个bbox，把它放入该放的地方
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        # 获取中心点所在-下标-
        grid_x = int(bbox[1] // grid_size)
        grid_y = int(bbox[2] // grid_size)
        x = bbox[1] / grid_size - grid_x
        y = bbox[2] / grid_size - grid_y
        label[0:5, grid_x, grid_y] = np.array([x, y, bbox[3], bbox[4], 1])  # first bbox idx:0-4
        label[5:10, grid_x, grid_y] = np.array([x, y, bbox[3], bbox[4], 1])  # second bbox idx:5-9
        label[10 + int(bbox[0]), grid_x, grid_y] = 1  # class one-hot:10-30
    return label


def class_get(cls_txt):
    """
    遍历xml读取kind
    txt file -> buffer
    """
    with open(cls_txt, 'r') as f:
        content = f.read()
        content = content.split('\n')[:-1]
        return content


def class_generator(xml_dir, cls_txt):
    """
    buffer -> txt file
    """
    classes = []
    for xml_name in tqdm(os.listdir(xml_dir)):
        xml_path = xml_dir + '/' + xml_name
        tree = parse(xml_path)
        rootnode = tree.documentElement
        objects = rootnode.getElementsByTagName('object')
        for object in objects:
            label = object.getElementsByTagName('name')[0].childNodes[0].data
            classes.append(label)
    classes = list(set(classes))
    strs = ""
    for name in classes:
        strs = strs + "{}\n".format(name)
    with open(cls_txt, 'w') as f:
        f.write(strs)


def generator(image_dir, xml_dir, txt_dir, cls_path, refer_path):
    for path in [image_dir, xml_dir, txt_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    if not os.path.exists(cls_path):  # 如果没有classes.txt，就生成一个
        class_generator(xml_dir, cls_path)
    classes = class_get(cls_path)
    info_dict = {"img_name": [], "img_path": [], "object_path": []}
    for xml_name in tqdm(os.listdir(xml_dir)):
        xml_path = xml_dir + '/' + xml_name
        img_name, img_path, _, height, width, objects_info = readvocxml(xml_path, image_dir)
        for i in range(len(objects_info)):
            objects_info[i][0] = classes.index(objects_info[i][0])
            objects_info[i][1] /= width
            objects_info[i][2] /= height
            objects_info[i][3] /= width
            objects_info[i][4] /= height
        npy_path = txt_dir + '/' + img_name[:-4] + '.npy'
        objects_info = np.array(objects_info)
        np.save(npy_path, objects_info)
        info_dict['img_name'].append(img_name)
        info_dict['img_path'].append(img_path)
        info_dict['object_path'].append(npy_path)
    df = pd.DataFrame(info_dict)
    df.to_csv(refer_path, encoding='utf-8')


class VOCDataset(Dataset):
    def __init__(self, image_dir, csv_path, resize):
        super(VOCDataset, self).__init__()
        self.image_dir = image_dir
        self.resize = resize
        self.df = read_csv(csv_path, encoding='utf-8', engine='python')
        # 图像转换器，该转换器会调整图像的大小为resize并将其转换为张量
        self.transformer = Compose([
            Resize(self.resize),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.transformer(Image.open(self.df['img_path'][idx]))
        label = np.load(self.df['object_path'][idx])
        label = convert_bbox2labels(label)

        return image, label


def get_dataloader(image_dir, csv_path, resize, batch_size, num_workders, train_percent=0.9):
    dataset = VOCDataset(image_dir, csv_path, resize)
    num_sample = len(dataset)
    # 训练数据集数量
    num_train = int(train_percent * num_sample)
    # 验证数据集数量
    num_valid = num_sample - num_train
    # 根据数量随机进行数据集分割
    train_ds, valid_ds = random_split(dataset, [num_train, num_valid])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workders, pin_memory=True,
                          persistent_workers=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workders, pin_memory=True,
                          persistent_workers=True)
    return train_dl, valid_dl, len(dataset), len(train_ds), len(valid_ds)
