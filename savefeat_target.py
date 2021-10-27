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
from metrics import runningScore, averageMeter
from loss import get_loss_function
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def Savefeat(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))
    ## create dataset
    default_gpu = cfg['model']['default_gpu']
    device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(cfg, writer, logger)  #source_train\ target_train\ source_valid\ target_valid + _loader

    model = CustomModel(cfg, writer, logger)

    # We set the Cityscapes as the target dataset
    target_train_loader = datasets.target_train_loader
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

    # begin training
    i_iter = 0

    print(len(datasets.source_train))
    full_dataset_objective_vectors = np.zeros([len(datasets.target_train), 19, 256])
    print(full_dataset_objective_vectors.shape)

    with torch.no_grad():
        for target_image, target_label, target_img_name in tqdm(datasets.target_train_loader):
            target_image = target_image.to(device)
            if cfg['training'].get('freeze_bn') == True:
                model.freeze_bn_apply()
            if model.PredNet.training:
                model.PredNet.eval()

            _, _, feat_cls, output = model.PredNet_Forward(target_image)
            # calculate pseudo-labels
            # threshold_arg, cluster_arg = model.metrics.update(feat_cls, output, target_label, model)
            outputs_softmax = F.softmax(output, dim=1)
            outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
            target_vectors, target_ids = model.calculate_mean_vector(feat_cls, output, outputs_argmax.float())
            print(target_vectors)
            print(target_ids)
            single_image_objective_vectors = np.zeros([19, 256])
            for t in range(len(target_ids)):
                single_image_objective_vectors[target_ids[t]] = target_vectors[t].detach().cpu().numpy().squeeze()
                # model.update_objective_SingleVector(ids[t], vectors[t].detach().cpu().numpy(), 'mean')
            print(single_image_objective_vectors)
            full_dataset_objective_vectors[i_iter, :] = single_image_objective_vectors[:]
            print(i_iter)

            i_iter += 1

    torch.save(full_dataset_objective_vectors, 'features/target_full_dataset_objective_vectors.pkl')


class Class_Features:
    def __init__(self, numbers = 19):
        self.class_numbers = numbers
        self.tsne_data = 0
        self.pca_data = 0
        # self.class_features = np.zeros((19, 256))
        self.class_features = [[] for i in range(self.class_numbers)]
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
            'ignored',]
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
            "|",]
        return

    def calculate_mean_vector(self, feat_cls, outputs, labels_val, model):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = model.process_label(outputs_argmax.float())
        # outputs_pred = model.process_pred(outputs_softmax, 0.5)
        # outputs_pred = outputs_argmax[:, 0:19, :, :] * outputs_softmax
        labels_expanded = model.process_label(labels_val)
        outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids
    
    def calculate_mean(self,):
        out = [[] for i in range(self.class_numbers)]
        for i in range(self.class_numbers):
            out[i] = self.class_features[i] / max(self.num[i], 1)
        return out

    def calculate_dis(self, vector, id):
        if isinstance(vector, torch.Tensor): vector = vector.detach().cpu().numpy().squeeze()
        mean = self.calculate_mean()
        dis = []
        for i in range(self.class_numbers):
            dis_vec = [x - y for x,y in zip(mean[i], vector)]
            dis.append(np.linalg.norm(dis_vec, 2))
        return dis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/CAC_from_gta_to_city_target.yml',
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

    Savefeat(cfg, writer, logger)
