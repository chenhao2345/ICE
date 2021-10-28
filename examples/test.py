from __future__ import print_function, absolute_import

import argparse
import os.path as osp
import random
import numpy as np
import sys
import time

from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from ice.utils.logging import Logger
from ice import datasets
from ice import models
from ice.trainers import ImageTrainer
from ice.evaluators import Evaluator, extract_features
from ice.utils.data import IterLoader
from ice.utils.data import transforms as T
from ice.utils.data.sampler import MoreCameraSampler
from ice.utils.data.preprocessor import Preprocessor_mutual, Preprocessor
from ice.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from ice.utils.faiss_rerank import compute_jaccard_distance
from ice.utils.lr_scheduler import WarmupMultiStepLR

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, mutual=False, index=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.GaussianBlur([.1, 2.])], p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.6, mean=[0.485, 0.456, 0.406]),
    ])

    weak_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = MoreCameraSampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor_mutual(train_set, root=dataset.images_dir, transform=train_transformer, mutual=mutual, transform_weak=weak_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)
    model_1_ema.cuda()
    model_1_ema = nn.DataParallel(model_1_ema)
    return model_1_ema


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, 256, args.workers)

    # Create model
    model_1_ema = create_model(args)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)
    checkpoint = load_checkpoint(osp.join(args.logs_dir, args.dataset_target+'_unsupervised', 'model_best.pth.tar'))
    model_1_ema.load_state_dict(checkpoint['state_dict'])
    evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Moco Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters")
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--tau-c', type=float, default=0.5)
    parser.add_argument('--tau-v', type=float, default=0.1)
    parser.add_argument('--scale-kl', type=float, default=0.4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[],
                        help='milestones for the learning rate decay')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    # cluster
    parser.add_argument('--eps', type=float, default=0.55, help="dbscan threshold")
    parser.add_argument('--k1', type=int, default=30,
                        help="k1, default: 30")
    parser.add_argument('--min-samples', type=int, default=4,
                        help="min sample, default: 4")
    end = time.time()
    main()
    print('Time used: {}'.format(time.time()-end))
