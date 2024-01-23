from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import math
from data_utils.FG3DDataHandle import dataNormalization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

"""
配置参数：
--normal 
--log_dir pointnet2_cls_msg
"""
def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting [default: 3]')
    # kandy
    parser.add_argument('--DATA_PATH', type=str, default="data/airp_ship/", help='the path of folder')
    parser.add_argument('--num_class', type=int, default=9, help='the number of classes')
    parser.add_argument('--seg_type', type=str, default=" ",
                        help='Is the format of point cloud data separated by commas or spaces')
    parser.add_argument('--uniform', type=bool, default=True, help='IS adopt FPS')
    parser.add_argument('--weight_name', type=str, default="best_model_lihailiang_240115.pth",
                        help='To load the name of weight')
    # kandy
    return parser.parse_args()

def get_subfolder_names(folder_path):
    subfolders_list = []

    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            subfolders_list.append(dir)

    return subfolders_list


def test(model,txtPath):
    mean_correct = []
    # class_acc = np.zeros((num_class,3))
    # all_data = [[0 for j in range(8)] for i in range(8)]
    points,classes = dataNormalization(txtPath)
    points = points.transpose(2, 1)
    points = points.cuda()
    classifier = model.eval()
    pred, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    cls = pred_choice.item()
    cls_name = find_key(classes,cls)
    return cls_name


def main(args,txtPath):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # log_string(args)


    '''MODEL LOADING'''
    num_class = args.num_class
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)

    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()

    checkpoint = torch.load(str(experiment_dir) + ('/checkpoints/' + args.weight_name))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    class_name = test(classifier.eval(),txtPath)
    return class_name

if __name__ == '__main__':
    args = parse_args()
    txtPath = r"test_data/deltawing_000006.txt"
    cls_name = main(args,txtPath)
    print("预测类别:",cls_name)
