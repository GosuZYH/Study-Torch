import argparse
from datetime import datetime
from tqdm import tqdm
from utils.show import log_generator
import torch
from prettytable import PrettyTable 
from utils.dataload import get_dataloader
from model import Yolov1
from torch.optim import Adam
from utils.loss import YOLO_loss
from utils.show import class_get
from torchsummary import summary 

voc_image_path = 'D:\\Datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages'
voc_mapping_csv = 'D:\\Datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012_detect\\refer.csv'
voc_cls_idx = 'D:\\Datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012_detect\\classes.txt'
result_save_path = 'D:\\Datasets\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012_detect'

def args_parse():
    parser = argparse.ArgumentParser(description='This is a training parser of a object localization.')
    parser.add_argument( '-t', type=str, default='Pytorch-ObjectDetection-master', help='the theme of your training task' )
    parser.add_argument( '-imaged', type=str, default=voc_image_path, help="The JPEGImage's directory" )
    parser.add_argument( '-infop', type=str, default=voc_mapping_csv, help="The dataset information csv file's path" )
    parser.add_argument( '-clsp', type=str, default=voc_cls_idx, help="The dataset classes' information file's path" )
    parser.add_argument( '-rs', type=tuple, default=(448, 448), help="regular size of images" )
    parser.add_argument( '-nw', type=int, default=6, help="number of workers" )
    parser.add_argument( '-wd', type=str, default=result_save_path, help="the directory of log's saving path" )
    parser.add_argument( '-bs', type=int, default=32, help="batch size" )
    parser.add_argument( '-tp', type=float, default=0.9, help="train percent" )
    parser.add_argument( '-lr', type=float, default=1e-4, help="learning rate" )
    parser.add_argument( '-e', type=int, default=50, help="epoch" )
    return parser.parse_args()


if __name__=='__main__':
    args = args_parse()
# ----------------------------------------------------------------------------------------------------------------------
    # 训练设备信息
    print("Training device information:")
    device_table = ""
    if torch.cuda.is_available():
        device_table = PrettyTable(['number of gpu', 'applied gpu index', 'applied gpu name'], min_table_width=80)
        gpu_num = torch.cuda.device_count()
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name()
        device_table.add_row([str(gpu_num), str(gpu_index), str(gpu_name)])
        print('{}\n'.format(device_table))
    else:
        print("Using cpu......")
        device_table = 'CPU'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------------------------------------------
    # 数据集信息
    print("Use dataset information file:{}\nLoading dataset from path: {}......".format(args.infop, args.imaged))
    train_dl, valid_dl, samples_num, train_num, valid_num = get_dataloader(args.imaged, args.infop, args.rs, args.bs, args.nw, args.tp)
    dataset_table = PrettyTable(['number of samples', 'train number', 'valid number', 'percent',"num_workers"], min_table_width=80)
    dataset_table.add_row([samples_num, train_num, valid_num, args.tp, args.nw])
    print("{}\n".format(dataset_table))
# ----------------------------------------------------------------------------------------------------------------------
    # 训练组件配置
    print("Classes information:")
    classes = class_get(args.clsp)
    classes_table = PrettyTable(classes, min_table_width=60)
    classes_table.add_row(range(len(classes)))
    print("{}\n".format(classes_table))
    print("Train information:")
    model = Yolov1().to(device)   
    summary(model, (3,*args.rs), args.bs)
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    loss_fn = YOLO_loss()  
    train_table = PrettyTable(['theme', 'resize', 'batch size', 'epoch', 'learning rate', 'directory of log'], min_table_width=120)
    train_table.add_row([args.t, args.rs, args.bs, args.e, args.lr, args.wd])
    print('{}\n\n'.format(train_table))
# ----------------------------------------------------------------------------------------------------------------------
    # 开始训练
    losses = [] 
    st = datetime.now()
    for epoch in range(args.e):

        prediction = []
        label = []
        score = []

        model.train()
        train_bar = tqdm(iter(train_dl), ncols=150, colour='red')
        train_loss = 0.
        i = 0
        for train_data in train_bar:
            x_train, y_train = train_data
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            output = model(x_train) 
            loss = loss_fn(output, y_train)
            optimizer.zero_grad()
            # clone().detach()：可以仅仅复制一个tensor的数值而不影响tensor# 原内存和计算图
            train_loss += loss.clone().detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            # 显示每一批次的loss
            train_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(train_dl)))
            train_bar.set_postfix({"train loss": "%.3f" % loss.data})
            i += 1
        train_loss = train_loss / i
        # 最后得到的i是一次迭代中的样本数批数
        losses.append(train_loss)
 
    et = datetime.now()

    log_generator(args.t, et - st, dataset_table, classes_table, device_table, train_table, optimizer, model, args.e,
                  [losses], args.wd)
# ----------------------------------------------------------------------------------------------------------------------
