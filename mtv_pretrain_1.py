import torch
import numpy as np
import os
import random
import time
import argparse
import logging
import math
import shutil

from thop import profile
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from archs.Physformer_Pre_Fine import Pretrain_ViT_ST_ST_Compact3_TDC_gra_sharp as backbone
from archs.Physformer_Pre_Fine import Stem as stem_backbone
from datasets.video_mask_datasets import VIPL, UBFC, PURE, BUAA
# from datasets.rppg_datasets import VIPL as VIPL_eva
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


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio=0.75):  # [args.window_size, args.mask_ratio]
        self.frames, self.height, self.width = input_size  # [40,4,4]
        self.num_patches_per_frame =  int(self.height * self.width)  # 1024 = 32 x 32  #16
        self.total_patches = self.frames * self.num_patches_per_frame  # 640
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)  # 12
        self.total_masks = self.frames * self.num_masks_per_frame  # 480

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])  # [num_patches,]
        np.random.shuffle(mask_per_frame)  # [num_patches,]
        mask = np.tile(mask_per_frame, (int(self.frames),1)).flatten()  # [self.frames,num_patches] --> [self.frames*num_patches,]
        return mask 


class MTVPretrainTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.model = backbone(frame=300, image_size=args.image_size, patches=(4,4,4), dim=96, ff_dim=144,\
                               num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7, mask_ratio=args.mask_ratio)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.stem_model = stem_backbone(patch_size=4)
        self.stem_model = torch.nn.DataParallel(self.stem_model).to(self.device)

        ## generate save path
        self.run_date = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        # self.save_path = f'{args.save_path}/{self.run_date}'
        
        ## dataloader NOTE: SELECT YOUR DATASET
        # -------------------------------------------------------
        ## VIPL dataset
        if args.dataset == 'VIPL':
            self.train_dataset = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train',\
                                       w=self.args.image_size, h=self.args.image_size, fold=1, mask_ratio=args.mask_ratio)
            self.val_dataset_clip = VIPL(data_dir='/data2/chushuyang/VIPL', T=960, train='test',\
                                          w=self.args.image_size, h=self.args.image_size, fold=1, mask_ratio=args.mask_ratio)
            # self.val_dataset_clip = VIPL_eva(data_dir='/data2/chushuyang/VIPL', T=-1, train='test', \
            #                                 w=args.image_size, h=args.image_size, fold=2)
        # -------------------------------------------------------
        ## UBFC dataset
        elif args.dataset == 'UBFC':
            self.train_dataset = UBFC(data_dir='/mnt/data/UBFC_PIC', T=args.num_rppg, train='train',\
                                       w=self.args.image_size, h=self.args.image_size, mask_ratio=args.mask_ratio)
            self.val_dataset_clip = UBFC(data_dir='/mnt/data/UBFC_PIC', T=args.num_rppg, train='test',\
                                          w=self.args.image_size, h=self.args.image_size, mask_ratio=args.mask_ratio)
        # -------------------------------------------------------
        ## PURE dataset
        elif args.dataset == 'PURE':
            self.train_dataset = PURE(data_dir='/data/chushuyang/pure', T=args.num_rppg, train='train')
            self.val_dataset_clip = PURE(data_dir='/data/chushuyang/pure', T=args.num_rppg, train='test')
        # -------------------------------------------------------
        ## BUAA dataset
        elif args.dataset == 'BUAA':
            self.train_dataset = BUAA(data_dir='/data2/chushuyang/BUAA', T=args.num_rppg, train='train')
            self.val_dataset_clip = BUAA(data_dir='/data2/chushuyang/BUAA', T=args.num_rppg, train='test')

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=_init_fn)
        self.val_dataloader_clip = DataLoader(self.val_dataset_clip, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=_init_fn)

        ## optimizer
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()) + \
                                                list(self.stem_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        self.criterion_MSE = torch.nn.MSELoss()
        self.criterion_DIFF = torch.nn.L1Loss()
        self.criterion_Pearson = Neg_Pearson()

        ## constant
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).to(self.device)
        self.best_epoch = 0
        self.best_val_mse = 1000    # mean absolute error
        self.frame_rate = 30
        self.a_start = 0.1
        self.b_start = 1.0
        self.exp_a = 0.5
        self.exp_b = 5.0

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
                            format='%(message)s', filemode='a')
        self.logger = logging.getLogger(f'./logs/{self.args.train_stage}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}')
        self.logger.setLevel(logging.INFO)

        ## save proj_file to save_path
        cur_file = os.getcwd()
        cur_file_name = cur_file.split('/')[-1]
        shutil.copytree(cur_file, f'{self.save_path}/{cur_file_name}/2')

        if start_epoch != 0:
            if not os.path.exists(f'{self.save_ckpt_path}/mvt_pretrain_{start_epoch - 1}.pth'):
                raise Exception(f'pretrain model ckpt file {start_epoch - 1} not found')
            self.model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_pretrain_{start_epoch - 1}.pth'))
            self.stem_model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_pretrain_stem_{start_epoch - 1}.pth'))
    
        self.model.train()
        self.stem_model.train()
    
    # def visualize_MAE(self, recon_inputs, inputs_gt, save_path_epoch, train=False, mini_batch=0):
    #     with torch.no_grad():
    #         recon_inputs = recon_inputs.cpu().data.numpy()
    #         inputs_gt = inputs_gt.cpu().data.numpy()
            

        
    #         fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    #         if not train:
    #             fig.savefig(os.path.join(save_path_epoch, f'test_rPPG.png'))
    #         else:
    #             fig.savefig(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.png'))
    #         plt.close(fig)

    # def update_best(self, epoch, loss_mse_avg):
    #     if loss_mse_avg < self.best_val_mse:
    #         self.best_val_mse = loss_mse_avg
    #         self.best_epoch = epoch
    #         # save the model
    #         torch.save(self.model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_pretrain_best.pth'))
    #         torch.save(self.stem_model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_pretrain_stem_best.pth'))

    #     self.logger.info(f'evaluate epoch {epoch} ----------------------------------')
    #     self.logger.info(f'val mse of cur model: {loss_mse_avg}')
    #     self.logger.info(f'val mse of best model: {self.best_val_mse:.4f}, achieved at epoch {self.best_epoch}')
    #     self.logger.info(f'------------------------------------------------------------------')

    def update_best(self, epoch, hr_pred, hr_gt, val_type='video'):
        cur_mae = np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))
        cur_rmse = np.sqrt(np.mean(np.square(np.array(hr_gt) - np.array(hr_pred))))
        cur_sd = np.std(np.array(hr_gt) - np.array(hr_pred))
        cur_r = pearson_correlation_coefficient(np.array(hr_gt), np.array(hr_pred))
        print("mae:"+str(cur_mae))
        print("rmse:"+str(cur_rmse))
        print("sd:"+str(cur_sd))
        print("r:"+str(cur_r))

        # if cur_mae < self.best_val_mae:
        #     self.best_val_mae = cur_mae
        #     self.best_val_rmse = cur_rmse
        #     self.best_epoch = epoch
        #     self.best_sd = cur_sd
        #     self.best_r = cur_r
        #     # save the model
        #     # torch.save(self.model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_finetune_best.pth'))

        # print("evaluate epoch "str(epoch)", total val "str(len(hr_gt)))
        # print(str(val_type)"-level mae of cur model:" str(np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))))
        # self.logger.info(f'{val_type}-level best mae of model: {self.best_val_mae:.2f}, best rmse: {self.best_val_rmse:.2f}, best epoch: {self.best_epoch}, ' \
        #                  f'best sd: {self.best_sd:.2f}, best r: {self.best_r:.2f}')

    def evaluate_clip(self, epoch = 0):
        self.save_ckpt_path = f'{args.save_path}/ckpt'
        self.save_rppg_path = f'{args.save_path}/rppg'
        # self.mask_generator = TubeMaskingGenerator(input_size=(40, 4, 4), mask_ratio=0.75)   #####  # 160 128 128

        # self.logger = logging.getLogger(f'./logs/{self.args.dataset}_{self.run_date}')
        # self.logger.setLevel(logging.INFO)

       
        save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
        loss_mse_list = []
        hr_gt = []
        hr_pred = []

        with torch.no_grad():

            for i, sample_batched in enumerate(tqdm(self.val_dataloader_clip)):
                # get the inputs
                inputs = sample_batched['video'].to(self.device)
                mask = sample_batched['mask'].to(torch.bool).to(self.device)
                # mask = self.mask_generator()
                # mask = torch.from_numpy(mask).to(torch.bool).to(self.device)
                ecg = sample_batched['ecg'].to(self.device)
                clip_average_HR = sample_batched['clip_avg_hr'].to(self.device)

                # print(clip_average_HR.shape)

                # print(inputs.shape)
                # print(mask.shape)

                # clip_len =  inputs.shape[2]  ### pretrain_hr_160
                clip_len = 160
                max_len = inputs.shape[2] // clip_len  
                # print("max_len:"+ str(max_len))    
                
                val_model = backbone(frame=args.num_rppg, image_size=args.image_size, patches=(4,4,4), dim=96, ff_dim=144,\
                               num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7, mask_ratio=args.mask_ratio)
                # val_model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_pretrain_{epoch}.pth'))
                val_model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_pretrain_best.pth'))
                val_model = torch.nn.DataParallel(val_model).to(self.device)

                val_stem_model = stem_backbone(patch_size=4)
                # val_stem_model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_pretrain_stem_{epoch}.pth'))
                val_stem_model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_pretrain_stem_best.pth'))
                val_stem_model = torch.nn.DataParallel(val_stem_model).to(self.device)

                # max_len = min(max_len, 3)
                inputs = inputs[:, :, :max_len*clip_len, :, :]
                # mask = mask[:, :, :max_len*clip_len, :, :]
                ecg = ecg[:, :max_len*clip_len]
                # print(inputs.shape)
                # print(ecg.shape)


                # forward + backward + optimize
                ## for mtv stem model:
                # inputs_patch = self.stem_model(inputs)
                # inputs_gt = inputs_patch.detach()
                # ## for reconstruct model:
                # recon_inputs, rPPG_peak = self.model(inputs_patch, mask)

                psd_total = 0
                psd_gt_total = 0
                for idx in range(max_len):
                    ecg_iter = ecg[:, idx*clip_len : idx*clip_len+clip_len]
                    inputs_iter = inputs[:, :, idx*clip_len : idx*clip_len+clip_len, :, :]
                    # mask = mask[:, :, idx*clip_len : idx*clip_len+clip_len, :, :]
                    psd_gt = TorchLossComputer.complex_absolute(ecg_iter, self.frame_rate, self.bpm_range)
                    psd_gt_total += psd_gt.view(-1).max(0)[1].cpu() + 40

                    ## for mtv finetune model:
                    inputs_patch = val_stem_model(inputs_iter)
                    # print(inputs_patch.shape)
                    recon_inputs, rPPG_peak = val_model(inputs_patch, mask)
                    rPPG = (rPPG_peak - rPPG_peak.mean(dim=-1, keepdim=True)) / \
                            torch.abs(rPPG_peak).max(dim=-1, keepdim=True).values  # normalize
                    # print(rPPG.shape)

                    psd = TorchLossComputer.complex_absolute(rPPG[0], self.frame_rate, self.bpm_range)
                    psd_total += psd.view(-1).max(0)[1].cpu() + 40

                    # loss_mse = self.criterion_MSE(recon_inputs, inputs_gt)
                    # loss_mse_total += loss_mse
                
                # loss_mse_list.append(loss_mse_total.cpu().data.numpy())      
                hr_pred.append(psd_total / max_len)
                # hr_gt.append(psd_gt_total / max_len)
                hr_gt.append(clip_average_HR[0].item())


        # loss_mse_avg = np.mean(loss_mse_list)
        
        ## save the results
        # self.visualize_MAE(recon_inputs, inputs_gt, save_path_epoch, train=False)
        # self.update_best(epoch, loss_mse_avg)
        print(len(hr_pred))
        print(hr_pred)
        print(len(hr_gt))
        print(hr_gt)
        self.update_best(epoch, hr_pred, hr_gt, val_type='video')
        
    
    def train(self, start_epoch=0, continue_log=''):

        self.prepare_train(start_epoch=start_epoch, continue_log=continue_log)
        self.logger.info(f'prepare train, start_epoch: {start_epoch}')

        echo_batches = self.args.echo_batches
        gamma = self.args.gamma
        step_size = self.args.step_size
        eval_step = self.args.eval_step
        lr = self.args.lr * (gamma ** (start_epoch // step_size))
        optimizer = self.optimizer
        scheduler = self.scheduler
        batch_size = self.args.batch_size

        for epoch in range(start_epoch, self.args.epochs):
            if epoch % step_size == step_size - 1:
                lr *= gamma
            loss_mse_avg = AvgrageMeter()
            loss_diff_avg = AvgrageMeter()
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
                    inputs, mask = sample_batched['video'].to(self.device), sample_batched['mask'].to(torch.bool).to(self.device)
                    ecg = sample_batched['ecg'].to(self.device)
                    clip_average_HR  = sample_batched['clip_avg_hr'].to(self.device)
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    
                    ## for mtv stem model:
                    inputs_patch = self.stem_model(inputs)

                    ## for reconstruct model:
                    recon_inputs, rPPG_peak = self.model(inputs_patch, mask)
                    inputs_gt = inputs_patch.detach()

                    print(recon_inputs.shape)

                    loss_mse = self.criterion_MSE(recon_inputs, inputs_gt)

                    try:
                        inputs_gt_diff = torch.diff(inputs_gt.contiguous().view(batch_size, -1, 4, 4, 96) , dim=1)
                        recon_inputs_diff = torch.diff(recon_inputs.contiguous().view(batch_size, -1, 4, 4, 96) , dim=1)
                    except:
                        print(inputs_gt.shape)
                        print(recon_inputs.shape)
                        raise Exception('diff error')

                    loss_diff = self.criterion_DIFF(recon_inputs_diff, inputs_gt_diff)

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
                    
                    # loss = 0.1 * loss_rPPG + b * (fre_loss + kl_loss)

                    loss = 1.0 * loss_mse + 1.0 * loss_diff +  0.1 * (0.1 * loss_rPPG + b * (fre_loss + kl_loss))
                    
                    loss.backward()
                    optimizer.step()

                    ## update loss saver
                    n = inputs.size(0)
                    loss_mse_avg.update(loss_mse.data, n)
                    loss_diff_avg.update(loss_diff.data, n)
                    loss_rPPG_avg.update(loss_rPPG.data, n)
                    loss_peak_avg.update(fre_loss.data, n)
                    loss_kl_avg.update(kl_loss.data, n)
                    loss_hr_mae.update(train_mae.data, n)
    
                    if i % echo_batches == echo_batches - 1:  # info every mini-batches
                        
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.7f}, mse_loss = {loss_mse_avg.avg:.4f}, ' \
                                f'diff_loss = {loss_diff_avg.avg:.4f}')
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, NP_loss = {loss_rPPG_avg.avg:.4f}, ' \
                                f'fre_loss = {loss_peak_avg.avg:.4f}, hr_mae = {loss_hr_mae.avg[0]:.2f}')


                        # save the ecg images
                        # self.draw_rppg_ecg(rPPG, ecg, save_path_epoch, train=True, mini_batch=i)

                    pbar.set_description(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.7f}, mse_loss = {loss_mse_avg.avg:.4f}, ' \
                                f'diff_loss = {loss_diff_avg.avg:.4f}')
                    pbar.set_description(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.4f}, NP_loss = {loss_rPPG_avg.avg:.3f}, ' \
                                    f'fre_loss = {loss_peak_avg.avg:.3f}, hr_mae = {loss_hr_mae.avg[0]:.2f}')
                        
                        
                    
            scheduler.step()

            # save the model
            torch.save(self.model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_pretrain_{epoch}.pth'))
            torch.save(self.stem_model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_pretrain_stem_{epoch}.pth'))

            # delete the model
            if epoch > 0:
                os.remove(os.path.join(self.save_ckpt_path, f'mvt_pretrain_{epoch-1}.pth'))
                os.remove(os.path.join(self.save_ckpt_path, f'mvt_pretrain_stem_{epoch-1}.pth'))

            # evaluate the model
            if epoch % eval_step == eval_step - 1:
                self.evaluate_clip(epoch)              

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## general parameters
    parser.add_argument('--num_rppg', type=int, default=160, help='the number of rPPG')
    parser.add_argument('--image_size', type=int, default=128, help='the number of ecg')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--dataset', type=str, default='VIPL')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='the ratio of mask')

    ### add for mtv pretrain
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_stage', type=str, default='MTV_pretrain')
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--echo_batches', type=int, default=500, help='the number of mini-batches to print the loss')
    parser.add_argument('--save_path', type=str, default='/home/zhangyizhu/MaskFusion/VIPL_400/0813_1451', help='the path to save the model [ckpt, code, visulization]')

    args = parser.parse_args()

    set_seed(92)

    # model = backbone(frame=args.num_rppg, image_size=args.image_size, patches=(4,4,4), dim=96, ff_dim=144,\
    #                            num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7, mask_ratio=0.75)
    # stem_model = stem_backbone(patch_size=4)
    # # summary(model, input_size=(1, 4, 3, 448, 256))
    # input_stem = torch.randn(1, 3, 160, 128, 128)  # [B, 3, 160, 128, 128]
    # flops1, params1 = profile(stem_model, inputs=(input_stem))

    # inputs = torch.randn(1, 1200, 96)  # [B, 3, 160, 128, 128]
    # inputs_arg = torch.randn(640)
    # flops2, params2 = profile(model, inputs=(inputs, inputs_arg))


    # print('flops is %.2fG' % (flops1/1e9)) ## 打印计算量
    # print('params is %.2fM' % (params1/1e6)) ## 打印参数量

    # print('flops is %.2fG' % (flops2/1e9)) ## 打印计算量
    # print('params is %.2fM' % (params2/1e6)) ## 打印参数量

    
    codephys_trainer = MTVPretrainTrainer(args)
    # codephys_trainer.train(start_epoch=0, continue_log='0813_1451') # NOTE: WHETHER TO CONTINUE TRAINING

    # codephys_trainer.prepare_train(start_epoch=5, continue_log='0727_0617')
    codephys_trainer.evaluate_clip(epoch=409)