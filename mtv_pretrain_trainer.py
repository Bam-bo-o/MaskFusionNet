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

class MTVPretrainTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.model = backbone(frame=args.num_rppg, image_size=args.image_size, patches=(4,4,4), dim=96, ff_dim=144,\
                               num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7, mask_ratio=args.mask_ratio)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.stem_model = stem_backbone(patch_size=4)
        self.stem_model = torch.nn.DataParallel(self.stem_model).to(self.device)

        ## generate save path
        self.run_date = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        self.save_path = f'{args.save_path}/{self.run_date}'
        
        ## dataloader NOTE: SELECT YOUR DATASET
        # -------------------------------------------------------
        ## VIPL dataset
        if args.dataset == 'VIPL':
            self.train_dataset = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='train',\
                                       w=self.args.image_size, h=self.args.image_size, fold=2, mask_ratio=args.mask_ratio)
            self.val_dataset_clip = VIPL(data_dir='/data2/chushuyang/VIPL', T=args.num_rppg, train='test',\
                                          w=self.args.image_size, h=self.args.image_size, fold=2, mask_ratio=args.mask_ratio)
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
        shutil.copytree(cur_file, f'{self.save_path}/{cur_file_name}/4')

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

    def update_best(self, epoch, loss_mse_avg):
        if loss_mse_avg < self.best_val_mse:
            self.best_val_mse = loss_mse_avg
            self.best_epoch = epoch
            # save the model
            torch.save(self.model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_pretrain_best.pth'))
            torch.save(self.stem_model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_pretrain_stem_best.pth'))

        self.logger.info(f'evaluate epoch {epoch} ----------------------------------')
        self.logger.info(f'val mse of cur model: {loss_mse_avg}')
        self.logger.info(f'val mse of best model: {self.best_val_mse:.4f}, achieved at epoch {self.best_epoch}')
        self.logger.info(f'------------------------------------------------------------------')

    def evaluate_clip(self, epoch = 0):
        val_model = backbone(frame=args.num_rppg, image_size=args.image_size, patches=(4,4,4), dim=96, ff_dim=144,\
                               num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7, mask_ratio=args.mask_ratio)
        val_model = torch.nn.DataParallel(val_model).to(self.device)
        val_model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_pretrain_{epoch}.pth'))

        val_stem_model = stem_backbone(patch_size=4)
        val_stem_model = torch.nn.DataParallel(val_stem_model).to(self.device)
        val_stem_model.load_state_dict(torch.load(f'{self.save_ckpt_path}/mvt_pretrain_stem_{epoch}.pth'))

        save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
        loss_mse_list = []

        with torch.no_grad():

            for i, sample_batched in enumerate(tqdm(self.val_dataloader_clip)):
                # get the inputs
                inputs, mask = sample_batched['video'].to(self.device), sample_batched['mask'].to(torch.bool).to(self.device)

                # forward + backward + optimize
                
                ## for mtv stem model:
                inputs_patch = self.stem_model(inputs)
                inputs_gt = inputs_patch.detach()

                ## for reconstruct model:
                recon_inputs, rPPG_peak = self.model(inputs_patch, mask)

                loss_mse = self.criterion_MSE(recon_inputs, inputs_gt)

                inputs_gt_diff = torch.diff(inputs_gt.contiguous().view(-1, 40, 4, 4, 96) , dim=1)
                recon_inputs_diff = torch.diff(recon_inputs.contiguous().view(-1, 40, 4, 4, 96) , dim=1)

                loss_diff = self.criterion_DIFF(recon_inputs_diff, inputs_gt_diff)

                loss_mse_list.append(loss_diff.cpu().data.numpy())

        loss_mse_avg = np.mean(loss_mse_list)
        
        ## save the results
        # self.visualize_MAE(recon_inputs, inputs_gt, save_path_epoch, train=False)
        self.update_best(epoch, loss_mse_avg)
    
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
            loss_all_avg = AvgrageMeter()
            loss_hr_mae = AvgrageMeter()
            loss_entropy_avg = AvgrageMeter()
            loss_kl_avg = AvgrageMeter()
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

                    # print(recon_inputs.shape)

                    loss_mse = self.criterion_MSE(recon_inputs, inputs_gt)

                    try:
                        inputs_gt_diff = torch.diff(inputs_gt.contiguous().view(-1, 40, 4, 4, 96) , dim=1)
                        recon_inputs_diff = torch.diff(recon_inputs.contiguous().view(-1, 40, 4, 4, 96) , dim=1)
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

                    loss_tem = loss_rPPG
                    loss_fre = (fre_loss + kl_loss)
                    # loss_fre = fre_loss

                    loss = 1.0 * loss_mse + 1.0 * loss_diff +  1.0 * (0.1 * loss_tem + b * loss_fre)
                    
                    loss.backward()
                    optimizer.step()

                    ## update loss saver
                    n = inputs.size(0)
                    loss_mse_avg.update(loss_mse.data, n)
                    loss_diff_avg.update(loss_diff.data, n)
                    loss_rPPG_avg.update(loss_tem.data, n)
                    loss_peak_avg.update(loss_fre.data, n)
                    loss_entropy_avg.update(fre_loss.data, n)
                    loss_kl_avg.update(kl_loss.data, n)
                    loss_all_avg.update(loss.data, n)
                    loss_hr_mae.update(train_mae.data, n)
    
                    if i % echo_batches == echo_batches - 1:  # info every mini-batches
                        
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.7f}, mse_loss = {loss_mse_avg.avg:.4f}, ' \
                                f'diff_loss = {loss_diff_avg.avg:.4f}')
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, tem_loss = {loss_rPPG_avg.avg:.4f}, ' \
                                f'fre_loss = {loss_peak_avg.avg:.4f}')
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, entropy_loss = {loss_entropy_avg.avg:.4f}, ' \
                                f'kl_loss = {loss_kl_avg.avg:.4f}')
                        self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, all_loss = {loss_all_avg.avg:.4f}, ' \
                                f'hr_mae = {loss_hr_mae.avg[0]:.2f}')


                        # save the ecg images
                        # self.draw_rppg_ecg(rPPG, ecg, save_path_epoch, train=True, mini_batch=i)

                    # pbar.set_description(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.7f}, mse_loss = {loss_mse_avg.avg:.4f}, ' \
                    #             f'diff_loss = {loss_diff_avg.avg:.4f}')
                    # pbar.set_description(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.4f}, NP_loss = {loss_rPPG_avg.avg:.3f}, ' \
                    #                 f'fre_loss = {loss_peak_avg.avg:.3f}')
                    # self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.7f}, mse_loss = {loss_mse_avg.avg:.4f}, ' \
                    #             f'diff_loss = {loss_diff_avg.avg:.4f}')
                    # self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, tem_loss = {loss_rPPG_avg.avg:.4f}, ' \
                    #             f'fre_loss = {loss_peak_avg.avg:.4f}')
                    # self.logger.info(f'epoch : {epoch:0>3}, mini-batch : {i:0>4}, lr = {lr:.5f}, all_loss = {loss_all_avg.avg:.4f}, ' \
                    #             f'hr_mae = {loss_hr_mae.avg[0]:.2f}')

                        
                        
                    
            scheduler.step()

            # save the model
            torch.save(self.model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_pretrain_{epoch}.pth'))
            torch.save(self.stem_model.state_dict(), os.path.join(self.save_ckpt_path, f'mvt_pretrain_stem_{epoch}.pth'))

            # delete the model
            if epoch > 0:
                os.remove(os.path.join(self.save_ckpt_path, f'mvt_pretrain_{epoch-1}.pth'))
                os.remove(os.path.join(self.save_ckpt_path, f'mvt_pretrain_stem_{epoch-1}.pth'))

            # evaluate the model
            # if epoch % eval_step == eval_step - 1:
            #     self.evaluate_clip(epoch)              

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## general parameters
    parser.add_argument('--num_rppg', type=int, default=160, help='the number of rPPG')
    parser.add_argument('--image_size', type=int, default=128, help='the number of ecg')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--dataset', type=str, default='VIPL')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='the ratio of mask')

    ### add for mtv pretrain
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_stage', type=str, default='MTV_pretrain')
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--echo_batches', type=int, default=150, help='the number of mini-batches to print the loss')
    parser.add_argument('--save_path', type=str, default='/home/zhangyizhu/paper_code/MTV_pretrain_result', help='the path to save the model [ckpt, code, visulization]')

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
    codephys_trainer.train(start_epoch=139, continue_log='0425_2251') # NOTE: WHETHER TO CONTINUE TRAINING

    # codephys_trainer.prepare_train(start_epoch=5, continue_log='0727_0617')
    # codephys_trainer.evaluate_video(epoch=5)