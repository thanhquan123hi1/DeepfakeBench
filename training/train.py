# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.iid_dataset import IIDDataset
from dataset.pair_dataset import pairDataset
from dataset.lrl_dataset import LRLDataset
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')

# [NEW] Thêm tham số weights_path giống test.py
parser.add_argument('--weights_path', type=str, default=None, help='Path to pretrained weights (overrides config)')

args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config.get('cuda', False):
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        from dataset.ff_blend import FFBlendDataset
        from dataset.fwa_blend import FWABlendDataset
        from dataset.sbi_dataset import SBIDataset
        from dataset.lsda_dataset import LSDADataset
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        from dataset.I2G_dataset import I2GDataset
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )
    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler, 
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last = (test_name=='DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer_config = config['optimizer'][opt_name]
        optimizer_config.setdefault('lr', 0.001)
        optimizer_config.setdefault('momentum', 0.9)
        optimizer_config.setdefault('weight_decay', 0.0)
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config['momentum'],
            weight_decay=optimizer_config['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer_config = config['optimizer'][opt_name]
        optimizer_config.setdefault('lr', 0.0003)
        optimizer_config.setdefault('weight_decay', 0.0)
        optimizer_config.setdefault('beta1', 0.9)
        optimizer_config.setdefault('beta2', 0.999)
        optimizer_config.setdefault('eps', 1e-8)
        optimizer_config.setdefault('amsgrad', False)
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            betas=(optimizer_config['beta1'], optimizer_config['beta2']),
            eps=optimizer_config['eps'],
            amsgrad=optimizer_config['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer_config = config['optimizer'][opt_name]
        optimizer_config.setdefault('lr', 0.001)
        optimizer_config.setdefault('momentum', 0.9)
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=optimizer_config['lr'],
            momentum=optimizer_config['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        config.setdefault('lr_eta_min', 0.0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank']=args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    
    # [NEW] Logic ưu tiên: CLI Argument > YAML Config
    if args.weights_path:
        config['pretrained'] = args.weights_path
        
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    config.setdefault('cuda', torch.cuda.is_available())
    config.setdefault('cudnn', False)
    config.setdefault('metric_scoring', 'auc')
    config.setdefault('with_landmark', False)
    config.setdefault('with_mask', False)
    config.setdefault('use_data_augmentation', False)
    config.setdefault('data_aug', {
        'flip_prob': 0.5,
        'rotate_prob': 0.5,
        'rotate_limit': [-10, 10],
        'blur_prob': 0.5,
        'blur_limit': [3, 7],
        'brightness_prob': 0.5,
        'brightness_limit': [-0.1, 0.1],
        'contrast_limit': [-0.1, 0.1],
        'quality_lower': 40,
        'quality_upper': 100,
    })
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],
                config['model_name'] + task_str + '_' + timenow
            )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp']= args.ddp
    
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
        
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    # --- [NEW] LOAD PRETRAINED WEIGHTS (Xử lý thông minh) ---
    if config.get('pretrained') is not None and os.path.exists(config['pretrained']):
        logger.info(f"🔄 Loading pretrained weights from: {config['pretrained']}")
        try:
            # Load checkpoint
            checkpoint = torch.load(config['pretrained'], map_location='cpu')
            
            # Xử lý trường hợp checkpoint lưu cả epoch/optimizer (dict lồng nhau)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Xử lý prefix 'module.' (do DataParallel/DDP)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") 
                new_state_dict[name] = v
            
            # Load với strict=False để hỗ trợ Fine-tuning (bỏ qua các layer không khớp)
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if len(missing_keys) > 0:
                logger.warning(f"⚠️ Missing keys (initialized randomly): {missing_keys[:5]} ... total {len(missing_keys)}")
            if len(unexpected_keys) > 0:
                logger.warning(f"⚠️ Unexpected keys (ignored): {unexpected_keys[:5]} ... total {len(unexpected_keys)}")
                
            logger.info("✅ Successfully loaded pretrained weights!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load pretrained weights: {e}")
    else:
        logger.info("ℹ️ No pretrained weights provided via CLI or YAML. Training from scratch.")
    # --------------------------------------------------------

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    
    if best_metric is not None:
        logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()
