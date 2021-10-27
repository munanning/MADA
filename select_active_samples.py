import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# import torchvision.models as models
# import torchvision

# from visdom import Visdom

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from tqdm import tqdm

from data import create_dataset
from utils.utils import get_logger
from models.adaptation_model import CustomModel
from models.utils import normalisation_pooling
from tensorboardX import SummaryWriter

import heapq

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
ncentroids = 10


def Select(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))
    # create dataset
    default_gpu = cfg['model']['default_gpu']
    device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(cfg, writer, logger)  # source_train\ target_train\ source_valid\ target_valid + _loader

    model = CustomModel(cfg, writer, logger)

    epoches = cfg['training']['epoches']

    # We select the active samples from the target dataset
    target_train_loader = datasets.target_train_loader
    selective_dataset = datasets.target_train
    logger.info('target train batchsize is {}'.format(target_train_loader.batch_size))
    print('target train batchsize is {}'.format(target_train_loader.batch_size))

    val_loader = None
    if cfg.get('valset') == 'gta5':
        val_loader = datasets.source_valid_loader
        logger.info('valset is gta5')
        print('valset is gta5')
    else:
        val_loader = datasets.target_valid_loader
        logger.info('valset is cityscapes')
        print('valset is cityscapes')
    logger.info('val batchsize is {}'.format(val_loader.batch_size))
    print('val batchsize is {}'.format(val_loader.batch_size))

    class_features = Class_Features(numbers=19)
    CAU_full = torch.load('./anchors/cluster_centroids_full_{}.pkl'.format(ncentroids))
    CAU_full = CAU_full.reshape(ncentroids, 19, 256)
    class_features.centroids = CAU_full

    cac_list = []

    # begin training
    model.iter = 0

    with torch.no_grad():
        for target_image, target_label, target_img_name in tqdm(datasets.target_train_loader):
            target_image = target_image.to(device)
            if cfg['training'].get('freeze_bn') == True:
                model.freeze_bn_apply()
            if model.PredNet.training:
                model.PredNet.eval()

            _, _, feat_cls, output = model.PredNet_Forward(target_image)

            outputs_softmax = F.softmax(output, dim=1)
            outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
            target_vectors, target_ids = model.calculate_mean_vector(feat_cls, output, outputs_argmax.float())
            single_image_objective_vectors = np.zeros([19, 256])
            for t in range(len(target_ids)):
                single_image_objective_vectors[target_ids[t]] = target_vectors[t].detach().cpu().numpy().squeeze()
            MSE = class_features.calculate_min_mse(single_image_objective_vectors)
            cac_list.append(MSE)
            print(MSE)

    # cac
    remaining_img_ids = selective_dataset.get_remainings()
    lenth = len(remaining_img_ids)
    per = 0.05
    selected_lenth = int(per * lenth)
    selected_index_list = list(map(cac_list.index, heapq.nlargest(selected_lenth, cac_list)))
    selected_index_list.sort()
    selected_img_list = []
    for index in selected_index_list:
        selected_img_list.append(remaining_img_ids[index])
    # file = open(os.path.join('./selection_list', 'stage1_cac_list_%.2f_c%d.txt' % (per, ncentroids)), 'w')
    file = open(os.path.join('./selection_list', 'stage1_cac_list_%.2f.txt' % per), 'w')
    for i in range(len(selected_img_list)):
        img = str(selected_img_list[i])
        x = img.split('/')
        temp = x[-2] + '/' + x[-1]
        file.write(temp + '\n')
    file.close()
    # file = open(os.path.join('./selection_list', 'stage1_cac_index_%.2f_c%d.txt' % (per, ncentroids)), 'w')
    file = open(os.path.join('./selection_list', 'stage1_cac_index_%.2f.txt' % per), 'w')
    for i in range(len(selected_index_list)):
        file.write(str(selected_index_list[i]) + '\n')
    file.close()


class Class_Features:
    def __init__(self, numbers=19):
        self.class_numbers = numbers
        self.tsne_data = 0
        self.pca_data = 0
        # self.class_features = np.zeros((19, 256))
        self.class_features = [[] for i in range(self.class_numbers)]
        self.centroids = np.zeros((10, 19, 256)).astype('float32')
        self.num = np.zeros(numbers)
        self.all_vectors = []
        self.pred_ids = []
        self.ids = []
        self.pred_num = np.zeros(numbers + 1)
        self.labels = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            'ignored', ]
        self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.markers = [".",
                        ",",
                        "o",
                        "v",
                        "^",
                        "<",
                        ">",
                        "1",
                        "2",
                        "3",
                        "4",
                        "8",
                        "p",
                        "P",
                        "*",
                        "h",
                        "H",
                        "+",
                        "x",
                        "|", ]
        return

    def calculate_mean_vector(self, feat_cls, outputs, labels_val):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        labels_expanded = self.process_label(labels_val)
        outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item() == 0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                scale = torch.sum(outputs_pred[n][t]) / labels_val.shape[2] / labels_val.shape[3] * 2
                s = normalisation_pooling()(s, scale)
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_min_mse(self, single_image_objective_vectors):
        loss = []
        for centroid in self.centroids:
            new_loss = np.mean((single_image_objective_vectors - centroid) ** 2)
            loss.append(new_loss)
        min_loss = min(loss)
        min_index = loss.index(min_loss)
        print(min_loss)
        print(min_index)
        return min_loss

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, 20, w, h).cuda()
        id = torch.where(label < 19, label, torch.Tensor([19]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/Select_from_gta_to_city.yml',
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    Select(cfg, writer, logger)
