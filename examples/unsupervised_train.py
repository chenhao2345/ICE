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
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)

    model_1_ema = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)

    model_1.cuda()
    model_1_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    model_1_ema = nn.DataParallel(model_1_ema)

    if args.init != '':
        initial_weights = load_checkpoint(args.init)
        copy_state_dict(initial_weights['state_dict'], model_1)
        copy_state_dict(initial_weights['state_dict'], model_1_ema)

    return model_1, model_1_ema


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
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, 256, args.workers)

    # Create model
    model_1, model_1_ema = create_model(args)

    # Optimizer
    params = []
    for key, value in model_1.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)

    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
                                     warmup_iters=args.warmup_step)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)

    for epoch in range(args.epochs):

        cluster_loader = get_test_loader(dataset_target, args.height, args.width, 256, args.workers, testset=dataset_target.train)
        dict_f1, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
        cf = torch.stack(list(dict_f1.values()))

        rerank_dist = compute_jaccard_distance(cf, k1=args.k1, k2=6)
        eps = args.eps

        print('eps in cluster: {:.3f}'.format(eps))
        print('Clustering and labeling...')
        cluster = DBSCAN(eps=eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)

        centers = []
        for id in range(num_ids):
            centers.append(torch.mean(cf[labels == id], dim=0))
        centers = torch.stack(centers, dim=0)

        # change pseudo labels
        pseudo_labeled_dataset = []
        pseudo_outlier_dataset = []
        labels_true = []

        cams = []

        for i, ((fname, pid, cid), label) in enumerate(zip(dataset_target.train, labels)):
            labels_true.append(pid)
            cams.append(cid)
            if label == -1:
                pseudo_outlier_dataset.append((fname, label.item(), cid))
            else:
                pseudo_labeled_dataset.append((fname, label.item(), cid))
        cams = np.asarray(cams)

        intra_id_features = []
        intra_id_labels = []
        for cc in np.unique(cams):
            percam_ind = np.where(cams == cc)[0]
            percam_feature = cf[percam_ind].numpy()
            percam_label = labels[percam_ind]
            percam_class_num = len(np.unique(percam_label[percam_label >= 0]))
            percam_id_feature = np.zeros((percam_class_num, percam_feature.shape[1]), dtype=np.float32)
            cnt = 0
            for lbl in np.unique(percam_label):
                if lbl >= 0:
                    ind = np.where(percam_label == lbl)[0]
                    id_feat = np.mean(percam_feature[ind], axis=0)
                    percam_id_feature[cnt, :] = id_feat
                    intra_id_labels.append(lbl)
                    cnt += 1
            percam_id_feature = percam_id_feature / np.linalg.norm(percam_id_feature, axis=1, keepdims=True)
            intra_id_features.append(torch.from_numpy(percam_id_feature))

        print('Epoch {} has {} labeled samples of {} ids and {} unlabeled samples'.
              format(epoch, len(pseudo_labeled_dataset), num_ids, len(pseudo_outlier_dataset)))
        print('Learning Rate:', optimizer.param_groups[0]['lr'])
        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters, trainset=pseudo_labeled_dataset, mutual=True)

        # Trainer
        trainer = ImageTrainer(model_1, model_1_ema, num_cluster=num_ids, alpha=args.alpha,
                                      num_instance=args.num_instances, tau_c=args.tau_c, tau_v=args.tau_v,
                                      scale_kl=args.scale_kl)

        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_target, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_target), centers=centers,
                      intra_id_labels=intra_id_labels, intra_id_features=intra_id_features, cams=cams, all_pseudo_label=labels)

        lr_scheduler.step()

        if (epoch+1)%args.eval_step==0:
            cmc, mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
            is_best = mAP_1 > best_mAP
            best_mAP = max(mAP_1, best_mAP)
            save_checkpoint({
                'state_dict': model_1_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, args.dataset_target+'_unsupervised','checkpoint.pth.tar'))

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
    # init
    parser.add_argument('--init', type=str,
                        default='',
                        metavar='PATH')
    end = time.time()
    main()
    print('Time used: {}'.format(time.time()-end))
