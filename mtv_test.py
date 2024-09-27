import torch
import numpy as np
import os
import random
import time
import argparse
import logging
import math
import shutil

from scipy import io as sio
from scipy import signal
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from archs.Physformer_Pre_Fine import Finetune_Physformer_MTV_GDFF as backbone
from datasets.rppg_datasets import VIPL, UBFC, PURE, BUAA, MMSE, COHFACE
from losses.NPLoss import Neg_Pearson
from losses.CELoss import TorchLossComputer
from utils.utils import AvgrageMeter, cxcorr_align, pearson_correlation_coefficient

def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _init_fn(seed=92):
    np.random.seed(seed)

class MTVFinetuneTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = backbone(frame=args.num_rppg, image_size=args.image_size, patches=(4,4,4), dim=96, ff_dim=144,\
                               num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7).to(self.device)

        ## generate save path
        self.run_date = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        self.save_path = f'{args.save_path}/{self.run_date}'
        
        ## dataloader NOTE: SELECT YOUR DATASET
        # -------------------------------------------------------
        ## VIPL dataset
        if args.dataset == 'VIPL':
            self.train_dataset = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=2)
            self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=2)
        # -------------------------------------------------------
        ## UBFC dataset
        elif args.dataset == 'UBFC':
            self.train_dataset = UBFC(data_dir='/data/chushuyang/UBFC_RAW', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size)
            self.val_dataset_video = UBFC(data_dir='/data/chushuyang/UBFC_RAW', T=-1, train='test', w=args.image_size, h=args.image_size)
        # -------------------------------------------------------
        ## PURE dataset
        elif args.dataset == 'PURE':
            self.train_dataset = PURE(data_dir='/data/chushuyang/pure', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size)
            self.val_dataset_video = PURE(data_dir='/data/chushuyang/pure', T=-1, train='test', w=args.image_size, h=args.image_size)
        # -------------------------------------------------------
        ## BUAA dataset
        elif args.dataset == 'BUAA':
            self.train_dataset = BUAA(data_dir='/data2/chushuyang/BUAA', T=args.num_rppg, train='train')
            self.val_dataset_video = BUAA(data_dir='/data2/chushuyang/BUAA', T=-1, train='test')
        # -------------------------------------------------------
        ## VIPL -> PURE
        elif args.dataset == 'V2P':
            self.train_dataset_1 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=1)
            self.train_dataset_2 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=2)
            self.train_dataset_3 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=3)
            self.train_dataset_4 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=4)
            self.train_dataset_5 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=5)
            self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset_1, self.train_dataset_2, self.train_dataset_3, self.train_dataset_4, self.train_dataset_5])
            self.val_dataset_video = PURE(data_dir='/data/chushuyang/pure', T=-1, train='test', w=args.image_size, h=args.image_size)
        # -------------------------------------------------------
        ## PURE -> VIPL
        elif args.dataset == 'P2V':
            self.train_dataset = PURE(data_dir='/data/chushuyang/pure', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size)
            self.val_dataset_1 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=1)
            self.val_dataset_2 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=2)
            self.val_dataset_3 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=3)
            self.val_dataset_4 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=4)
            self.val_dataset_5 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=5)
            self.val_dataset_video = torch.utils.data.ConcatDataset([self.val_dataset_1, self.val_dataset_2, self.val_dataset_3, self.val_dataset_4, self.val_dataset_5])
        # -------------------------------------------------------
        ## VIPL -> COHFACE
        elif args.dataset == 'V2C':
            self.train_dataset_1 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=1)
            self.train_dataset_2 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=2)
            self.train_dataset_3 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=3)
            self.train_dataset_4 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=4)
            self.train_dataset_5 = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size, fold=5)
            self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset_1, self.train_dataset_2, self.train_dataset_3, self.train_dataset_4, self.train_dataset_5])
            self.val_dataset_video = COHFACE(data_dir='/data/chushuyang/COHFACE/cohface', T=-1, train='test', w=args.image_size, h=args.image_size)
        # -------------------------------------------------------
        ## COHFACE -> VIPL
        elif args.dataset == 'C2V':
            self.train_dataset = COHFACE(data_dir='/data/chushuyang/COHFACE/cohface', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size)
            self.val_dataset_1 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=1)
            self.val_dataset_2 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=2)
            self.val_dataset_3 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=3)
            self.val_dataset_4 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=4)
            self.val_dataset_5 = self.val_dataset_video = VIPL(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', w=args.image_size, h=args.image_size, fold=5)
            self.val_dataset_video = torch.utils.data.ConcatDataset([self.val_dataset_1, self.val_dataset_2, self.val_dataset_3, self.val_dataset_4, self.val_dataset_5])
        # -------------------------------------------------------
        ## PURE -> COHFACE
        elif args.dataset == 'P2C':
            self.train_dataset = PURE(data_dir='/data/chushuyang/pure', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size)
            self.val_dataset_video = COHFACE(data_dir='/data/chushuyang/COHFACE/cohface', T=-1, train='test', w=args.image_size, h=args.image_size)
        # -------------------------------------------------------
        ## COHFACE-> PURE
        elif args.dataset == 'C2P':
            self.train_dataset = COHFACE(data_dir='/data/chushuyang/COHFACE/cohface', T=args.num_rppg, train='train', w=args.image_size, h=args.image_size)
            self.val_dataset_video = PURE(data_dir='/data/chushuyang/pure', T=-1, train='test', w=args.image_size, h=args.image_size)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=_init_fn)
        self.val_dataloader_video = DataLoader(self.val_dataset_video, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=_init_fn)

        ## optimizer
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        self.criterion_Pearson = Neg_Pearson()

        ## constant
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).to(self.device)
        self.best_epoch = 0
        self.best_val_mae = 1000    # mean absolute error
        self.best_val_rmse = 1000   # root mean square error
        self.best_sd = 1000         # standard deviation
        self.best_r = 0             # Pearson’s correlation coefficient
        self.frame_rate = 30        
        # a --> Pearson loss; b --> frequency loss
        self.a_start = 0.1
        self.b_start = 1.0
        self.exp_a = 0.5
        self.exp_b = 5.0

    def load_model(self, model, pretrained_model_path, pretrained_stem_path):
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})  # {}
            # print("local_metadata:" +str(local_metadata))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                # print(str(name)+" "+str(child))
                if child is not None:
                    # print("child is not None")
                    load(child, prefix + name + '.')

        prefix = ''
        ignore_missing = "relative_position_index"
        pre_model = torch.load(pretrained_model_path, map_location=self.device)
        pre_model_stem = torch.load(pretrained_stem_path, map_location=self.device)
        all_keys = list(pre_model.keys())
        all_keys_stem = list(pre_model_stem.keys())
        
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = pre_model[key]
            elif key.startswith('encoder.transformer'):
                new_dict[key[8:]] = pre_model[key]
            else:
                new_dict[key] = pre_model[key]
        for key in all_keys_stem:
            if key.startswith('patch_embedding.'): 
                new_dict[key] = pre_model_stem[key]
        
        state_dict = new_dict

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)  # None
        # print("metadata:"+str(metadata))
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        
        load(model, prefix=prefix)
        # print(model.state_dict())

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            self.logger.debug("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            self.logger.debug("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            self.logger.debug("Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            self.logger.debug('\n'.join(error_msgs))
    
    def prepare_train(self, start_epoch, continue_log):
        if start_epoch != 0:
            self.save_path = self.args.save_path + '/' + continue_log
            self.run_date = continue_log

        self.save_ckpt_path = f'{self.save_path}/ckpt'
        self.save_rppg_path = f'{self.save_path}/rppg'
        if not os.path.exists(self.save_ckpt_path):
            os.makedirs(self.save_ckpt_path)
        if not os.path.exists(self.save_rppg_path):
            os.makedirs(self.save_rppg_path)
        
        logging.basicConfig(filename=f'./logs/{self.args.train_stage}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}.log',\
                            format='%(message)s', filemode='a', level=logging.INFO)
        self.logger = logging.getLogger(f'./logs/{self.args.train_stage}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}')
        self.logger.setLevel(logging.INFO)

        ## save proj_file to save_path
        cur_file = os.getcwd()
        cur_file_name = cur_file.split('/')[-1]
        # shutil.copytree(cur_file, f'{self.save_path}/{cur_file_name}', dirs_exist_ok=True)
        shutil.copytree(cur_file, f'{self.save_path}/{cur_file_name}/5')

        if start_epoch != 0:
            if not os.path.exists(f'{self.save_ckpt_path}/mvt_finetune_{start_epoch - 1}.pth'):
                raise Exception(f'finetune model ckpt file {start_epoch - 1} not found')
            self.model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_finetune_{start_epoch - 1}.pth'))
        else:
            if self.args.add_pretrained:
                self.load_model(self.model, self.args.pretrained_model, self.args.pretrained_stem)
    
        self.model.train()
    
    def draw_rppg_ecg(self, rPPG, ecg, save_path_epoch, train=False, mini_batch=0):
        ## save the results
        b, a = signal.butter(2, [0.67 / 15, 3 / 15], 'bandpass')
        # 使用 lfilter 函数进行滤波
        rPPG_np = rPPG[0].cpu().data.numpy()
        rPPG_np = signal.lfilter(b, a, rPPG_np)
        rPPG[0] = torch.from_numpy(rPPG_np).to(self.device)
        results_rPPG = []
        y1 = rPPG[0].cpu().data.numpy()
        y2 = ecg[0].cpu().data.numpy() 
        results_rPPG.append(y1)
        results_rPPG.append(y2)
        if not train:
            sio.savemat(os.path.join(save_path_epoch, f'test_rPPG.mat'), {'results_rPPG': results_rPPG})
        else:
            sio.savemat(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.mat'), {'results_rPPG': results_rPPG})
        # show the ecg images
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        psd_pred = TorchLossComputer.complex_absolute(rPPG[0], self.frame_rate, self.bpm_range)
        psd_gt = TorchLossComputer.complex_absolute(ecg[0], self.frame_rate, self.bpm_range)
        ax[0].set_title('rPPG')
        ax[0].plot(y1, label='rPPG')
        ax[0].plot(y2, label='ecg')
        ax[0].legend()
        ax[1].set_title('psd')
        ax[1].plot(psd_pred[0].cpu().data.numpy(), label='pred')
        ax[1].plot(psd_gt[0].cpu().data.numpy(), label='gt')
        ax[1].legend()
        if not train:
            fig.savefig(os.path.join(save_path_epoch, f'test_rPPG.png'))
        else:
            fig.savefig(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.png'))
        plt.close(fig)

    def update_best(self, epoch, hr_pred, hr_gt, val_type='video'):
        cur_mae = np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))
        cur_rmse = np.sqrt(np.mean(np.square(np.array(hr_gt) - np.array(hr_pred))))
        cur_sd = np.std(np.array(hr_gt) - np.array(hr_pred))
        cur_r = pearson_correlation_coefficient(np.array(hr_gt), np.array(hr_pred))

        if cur_mae < self.best_val_mae:
            self.best_val_mae = cur_mae
            self.best_val_rmse = cur_rmse
            self.best_epoch = epoch
            self.best_sd = cur_sd
            self.best_r = cur_r
            # save the model
            torch.save(self.model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_finetune_best.pth'))

        self.logger.info(f'evaluate epoch {epoch}, total val {len(hr_gt)} ----------------------------------')
        self.logger.info(f'{val_type}-level mae of cur model: {np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))}')
        self.logger.info(f'{val_type}-level best mae of model: {self.best_val_mae:.2f}, best rmse: {self.best_val_rmse:.2f}, best epoch: {self.best_epoch}, ' \
                         f'best sd: {self.best_sd:.2f}, best r: {self.best_r:.2f}')
        self.logger.info(f'------------------------------------------------------------------')

    def evaluate_video(self, epoch = 0):
        save_path_epoch = f'/home/zhangyizhu/paper_code/0827_1446'
        hr_gt = []
        hr_pred = []

        with torch.no_grad():

            for i, sample_batched in enumerate(tqdm(self.val_dataloader_video)):
                # get the inputs
                inputs, ecg = sample_batched['video'].to(self.device), sample_batched['ecg'].to(self.device)
                clip_average_HR = sample_batched['clip_avg_hr'].to(self.device)

                clip_len =  inputs.shape[2] // 3

                max_len = inputs.shape[2] // clip_len

                val_model = backbone(frame=clip_len, image_size=self.args.image_size, patches=(4,4,4), dim=96, ff_dim=144,\
                               num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7).to(self.device)
                val_model.load_state_dict(torch.load(f'/home/zhangyizhu/paper_code/0827_1446/ckpt/mvt_finetune_best.pth'))        

                if max_len == 0:
                    continue
                
                # max_len = min(max_len, 3)
                inputs = inputs[:, :, :max_len*clip_len, :, :]
                ecg = ecg[:, :max_len*clip_len]

                psd_total = 0
                psd_gt_total = 0
                for idx in range(max_len):

                    ecg_iter = ecg[:, idx*clip_len : idx*clip_len+clip_len]
                    inputs_iter = inputs[:, :, idx*clip_len : idx*clip_len+clip_len, :, :]
                    
                    psd_gt = TorchLossComputer.complex_absolute(ecg_iter, self.frame_rate, self.bpm_range)
                    psd_gt_total += psd_gt.view(-1).max(0)[1].cpu() + 40

                    ## for mtv finetune model:
                    rPPG_peak, Score1, Score2, Score3 = val_model(inputs_iter, self.args.gra_sharp)
                    rPPG = (rPPG_peak - rPPG_peak.mean(dim=-1, keepdim=True)) / \
                            torch.abs(rPPG_peak).max(dim=-1, keepdim=True).values  # normalize

                    psd = TorchLossComputer.complex_absolute(rPPG[0], self.frame_rate, self.bpm_range)
                    psd_total += psd.view(-1).max(0)[1].cpu() + 40
                    
                hr_pred.append(psd_total / max_len)
                if self.args.dataset == 'V2M':
                    hr_gt.append(psd_gt_total / max_len)
                else:
                    hr_gt.append(clip_average_HR.item())

        for i in range(len(hr_pred)):
            print(hr_pred[i])
        for i in range(len(hr_gt)):
            print(hr_gt[i])
        print("hr_pred")
        print(hr_pred)
        print("hr_gt")
        print(hr_gt)
        
        ## save the results
        # self.draw_rppg_ecg(rPPG, ecg_iter, save_path_epoch)
        # self.update_best(epoch, hr_pred, hr_gt, val_type='video')
    
    def train(self, start_epoch=0, continue_log=''):

        self.prepare_train(start_epoch=start_epoch, continue_log=continue_log)
        self.logger.info(f'prepare train, load ckpt and block gradient, start_epoch: {start_epoch}')
        self.logger.info(f'all parameters have been loaded, start training, load_model: {self.args.add_pretrained} - {self.args.pretrained_model}')

        echo_batches = self.args.echo_batches
        gamma = self.args.gamma
        step_size = self.args.step_size
        eval_step = self.args.eval_step
        lr = self.args.lr * (gamma ** (start_epoch // step_size))
        optimizer = self.optimizer
        scheduler = self.scheduler

        for epoch in range(start_epoch, self.args.epochs):
            if epoch % step_size == step_size - 1:
                lr *= gamma
            loss_rPPG_avg = AvgrageMeter()
            loss_peak_avg = AvgrageMeter()
            loss_kl_avg = AvgrageMeter()
            loss_hr_mae = AvgrageMeter()
            save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
            if not os.path.exists(save_path_epoch):
                os.makedirs(save_path_epoch)
            self.logger.info(f'train epoch: {epoch} lr: {lr}')
            with tqdm(range(len(self.train_dataloader))) as pbar:
                for i, sample_batched in zip(pbar, self.train_dataloader):
                    inputs, ecg = sample_batched['video'].to(self.device), sample_batched['ecg'].to(self.device)
                    clip_average_HR  = sample_batched['clip_avg_hr'].to(self.device)

                    optimizer.zero_grad()

                    # forward + backward + optimize
                    
                    ## for mtv finetune model:
                    rPPG_peak, Score1, Score2, Score3 = self.model(inputs, self.args.gra_sharp)
                    rPPG = (rPPG_peak - rPPG_peak.mean(dim=-1, keepdim=True)) / \
                                torch.abs(rPPG_peak).max(dim=-1, keepdim=True).values  # normalize
                    
                    if epoch > 1:
                        rPPG = cxcorr_align(rPPG, ecg)  # align

                    loss_rPPG = self.criterion_Pearson(rPPG, ecg)    # np_loss
                    clip_average_HR = (clip_average_HR - 40)    # for 'TorchLossComputer.cross_entropy_power_spectrum_loss',\
                                                                # the input HR should be in range [0, 140] instead of [40, 180]

                    fre_loss = 0.0
                    kl_loss = 0.0
                    train_mae = 0.0
                    for bb in range(inputs.shape[0]):
                        kl_loss_temp, fre_loss_temp, train_mae_temp = \
                              TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(rPPG[bb],\
                                                            clip_average_HR[bb], self.frame_rate, 1.0)
                        fre_loss = fre_loss + fre_loss_temp
                        kl_loss = kl_loss + kl_loss_temp
                        train_mae = train_mae + train_mae_temp
                    fre_loss = fre_loss / inputs.shape[0]
                    kl_loss = kl_loss / inputs.shape[0]
                    train_mae = train_mae / inputs.shape[0]

                    if epoch >25:
                        a = 0.05
                        b = 5.0
                    else:
                        # exp descend
                        a = self.a_start*math.pow(self.exp_a, epoch/25.0)
                        # exp ascend
                        b = self.b_start*math.pow(self.exp_b, epoch/25.0)
                    
                    loss = 0.1 * loss_rPPG + b * (fre_loss + kl_loss) # NOTE : WHETHER TO USE NP_LOSS

                    loss.backward()
                    optimizer.step()

                    ## update loss saver
                    n = inputs.size(0)
                    loss_rPPG_avg.update(loss_rPPG.data, n)
                    loss_peak_avg.update(fre_loss.data, n)
                    loss_kl_avg.update(kl_loss.data, n)
                    loss_hr_mae.update(train_mae.data, n)

                    if i % echo_batches == echo_batches - 1:  # info every mini-batches
                        
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, NP_loss = {loss_rPPG_avg.avg:.4f}, ' \
                                f'fre_loss = {loss_peak_avg.avg:.4f}, hr_mae = {loss_hr_mae.avg[0]:.2f}')

                        # save the ecg images
                        self.draw_rppg_ecg(rPPG, ecg, save_path_epoch, train=True, mini_batch=i)

                    pbar.set_description(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.4f}, NP_loss = {loss_rPPG_avg.avg:.3f}, ' \
                                    f'fre_loss = {loss_peak_avg.avg:.3f}, hr_mae = {loss_hr_mae.avg[0]:.2f}')
                        
                    
            scheduler.step()

            # save the model
            torch.save(self.model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_finetune_{epoch}.pth'))

            # delete the last ckpt
            if epoch > 0:
                os.remove(os.path.join(self.save_ckpt_path, f'mvt_finetune_{epoch-1}.pth'))

            # evaluate the model
            if epoch % eval_step == eval_step - 1:
                self.evaluate_video(epoch)              

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## general parameters
    parser.add_argument('--num_rppg', type=int, default=160, help='the number of rPPG')
    parser.add_argument('--test_num_rppg', type=int, default=300, help='the number of rPPG for test')
    parser.add_argument('--image_size', type=int, default=128, help='the number of ecg')
    parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    parser.add_argument('--dataset', type=str, default='VIPL')
 
    ### add for mtv finetune
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_stage', type=str, default='MTV_finetune')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--echo_batches', type=int, default=20, help='the number of mini-batches to print the loss')
    parser.add_argument('--save_path', type=str, default='/home/zhangyizhu/paper_code/MTV_result', help='the path to save the model [ckpt, code, visulization]')

    ## model parameters
    parser.add_argument('--gra_sharp', type=float, default=2.0, help='the parameter of gradient sharpening')
    parser.add_argument('--add_pretrained', type=bool, default=False, help='whether to add pretrained model')
    parser.add_argument('--pretrained_model', type=str, default='/home/zhangyizhu/paper_code/0822_1513/ckpt/mvt_pretrain_best.pth',\
                         help='the path to load the pretrained model')
    parser.add_argument('--pretrained_stem', type=str, default='/home/zhangyizhu/paper_code/0822_1513/ckpt/mvt_pretrain_stem_best.pth',\
                         help='the path to load the pretrained stem')


    args = parser.parse_args()

    set_seed(92)

    codephys_trainer = MTVFinetuneTrainer(args)
    # codephys_trainer.train(start_epoch=0, continue_log='0915_0324') # NOTE: WHETHER TO CONTINUE TRAINING
    # codephys_trainer.prepare_train(start_epoch=5, continue_log='0727_0617')
    codephys_trainer.evaluate_video()