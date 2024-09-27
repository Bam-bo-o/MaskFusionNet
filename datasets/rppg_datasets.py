import cv2
import os
import math
import h5py
import scipy
import numpy as np
import scipy.io as sio
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import transforms
# import transforms

def cal_hr(output : torch.Tensor, Fs : float):
    '''
    args:
        output: (1, T)
        Fs: sampling rate
    return:
        hr: heart rate
    '''
    def compute_complex_absolute_given_k(output : torch.Tensor, k : torch.Tensor, N : int):
        two_pi_n_over_N = 2 * math.pi * torch.arange(0, N, dtype=torch.float) / N
        hanning = torch.from_numpy(np.hanning(N)).type(torch.FloatTensor).view(1, -1)

        k = k.type(torch.FloatTensor)
        two_pi_n_over_N = two_pi_n_over_N
        hanning = hanning
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute
    
    output = output.view(1, -1)

    N = output.size()[1]
    bpm_range = torch.arange(40, 180, dtype=torch.float)
    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz
    
    # only calculate feasible PSD range [0.7, 4]Hz
    complex_absolute = compute_complex_absolute_given_k(output, k, N)
    complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute
    whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0) # max返回（values, indices）
    whole_max_idx = whole_max_idx.type(torch.float) # 功率谱密度的峰值对应频率即为心率

    return whole_max_idx + 40	# Analogous Softmax operator

# 返回T帧或者整个视频
# 每一个数据集实现自己的读视频，读帧率的方法
# T=-1表示读整个视频
class BaseDataset(Dataset):
    def __init__(self, data_dir, train='train', T=-1, w=64, h=64):
        '''
        :param data_dir: 数据集的根目录
        :param train: 是否是训练集
        :param T: 读取的帧数，-1表示读取整个视频
        :param transform_rate: getitem返回的视频的帧率和ecg的采样率
        '''
        self.data_dir = data_dir
        self.train = train
        self.T = T
        self.w = w
        self.h = h
        self.data_list = list()
        self.get_data_list()
        # self.aug = 'figsc'
        self.aug = ''
        self.speed_slow = 0.6
        self.speed_fast = 1.4
        self.set_augmentations()

    def get_data_list(self):
        '''
        return: self.data_list list(dict)
            for each sample in self.data_list:
                sample['location']: sample's location
                sample['start_idx']: start index of sample
                sample['video_length']: length of sample
            each os.path.join(sample['location'], 'sample.hdf5') contains:
                'video_data': (C, T, H, W)  ----- for all video
                'ecg_data': (T)             ----- for all video
        '''
        raise NotImplementedError
    
    def set_augmentations(self):
        self.aug_flip = False
        self.aug_illum = False
        self.aug_gauss = False
        self.aug_speed = False
        self.aug_resizedcrop = False
        if self.train == 'train_all' or self.train == 'train':
            self.aug_flip = True if 'f' in self.aug else False
            self.aug_illum = True if 'i' in self.aug else False
            self.aug_gauss = True if 'g' in self.aug else False
            self.aug_speed = True if 's' in self.aug else False
            self.aug_resizedcrop = True if 'c' in self.aug else False
        self.aug_reverse = False ## Don't use this with supervised
    
    def apply_transformations(self, clip, idcs, augment=True):
        speed = 1.0
        if augment:
            ## Time resampling
            if self.aug_speed and np.random.rand() < 0.5:
                clip, idcs, speed = transforms.augment_speed(clip, idcs, self.T, self.speed_slow, self.speed_fast) # clip: (T, H, W, C) -> (C, T, H, W)
            else:
                clip = clip[idcs].transpose(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W)

            ## Randomly horizontal flip
            if self.aug_flip:
                clip = transforms.augment_horizontal_flip(clip)

            ## Randomly reverse time
            if self.aug_reverse:
                clip = transforms.augment_time_reversal(clip)

            ## Illumination noise
            if self.aug_illum:
                clip = transforms.augment_illumination_noise(clip)

            ## Gaussian noise for every pixel
            if self.aug_gauss:
                clip = transforms.augment_gaussian_noise(clip)

            ## Random resized cropping
            if self.aug_resizedcrop:
                clip = transforms.random_resized_crop(clip)

        clip = np.clip(clip, 0, 255)
        clip = clip / 255
        clip = torch.from_numpy(clip).float()

        return clip, idcs, speed

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''return: video, ecg, transform_rate, frame_start, frame_end'''
        sample = self.data_list[index]
        start_idx = sample['start_idx']
        video_length = sample['video_length']
        exist_gt_hr = False
        if self.T == -1:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data']).transpose(1, 2, 3, 0) # T, H, W, C
                ecg = np.array(f['ecg_data']) # T
                if 'gt_hr' in f.keys():
                    gt_hr = np.array(f['gt_hr'])
                    exist_gt_hr = True
        elif start_idx + int(self.T * 1.5) > video_length:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data'][:, start_idx: start_idx + self.T]).transpose(1, 2, 3, 0) # T, H, W, C
                ecg = np.array(f['ecg_data'][start_idx: start_idx + self.T]) # T
        else:
            with h5py.File(os.path.join(sample['location'], 'sample.hdf5'), 'r') as f:
                video_x = np.array(f['video_data'][:, start_idx: start_idx + int(self.T * 1.5)]).transpose(1, 2, 3, 0) # T, H, W, C
                ecg = np.array(f['ecg_data'][start_idx: start_idx + int(self.T * 1.5)]) # T



        idcs = np.arange(0, self.T, dtype=int) if self.T != -1 else np.arange(len(video_x), dtype=int)
        video_x_aug, speed_idcs, speed = self.apply_transformations(video_x, idcs)

        if [self.w, self.h] != video_x.shape[1:3]:
            video_x_aug = torch.nn.functional.interpolate(video_x_aug, size=(self.w, self.h), mode='bilinear', align_corners=False)

        if speed != 1.0:
            min_idx = int(speed_idcs[0])
            max_idx = int(speed_idcs[-1])+1
            orig_x = np.arange(min_idx, max_idx, dtype=int)
            orig_wave = ecg[orig_x]
            wave = np.interp(speed_idcs, orig_x, orig_wave)
            
        else:
            wave = ecg[idcs]

        wave = wave - wave.mean()
        wave = wave / np.abs(wave).max()
        wave = torch.from_numpy(wave).float()

        sample_item = {}
        sample_item['video'] = video_x_aug
        sample_item['ecg'] = wave
        sample_item['clip_avg_hr'] = gt_hr if exist_gt_hr else cal_hr(wave, 30)
        sample_item['location'] = sample['location']

        return sample_item

class PURE(BaseDataset):
    def __init__(self, data_dir='/root/data/pure', train='train', T=-1, w=64, h=64):
        super().__init__(data_dir, train, T, w, h)
        
    def get_data_list(self):
        date_list = os.listdir(self.data_dir)
        date_list.sort()
        train_list = ['06-01', '06-03', '06-04', '06-05', '06-06', '08-01', '08-02', '08-03', '08-04', '08-05', '08-06',\
                    '05-01', '05-02', '05-03', '05-04', '05-05', '05-06', '01-01', '01-02', '01-03', '01-04', '01-05', '01-06',\
                    '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '09-01', '09-02', '09-03', '09-04', '09-05', '09-06',\
                    '07-01', '07-02', '07-03', '07-04', '07-05', '07-06']
        if self.train == 'train':
            date_list = [i for i in date_list if i in train_list]
        elif self.train == 'test':
            date_list = [i for i in date_list if i not in train_list]
        elif 'all' in self.train:
            pass
        else:
            raise NotImplementedError

        for date in date_list:
            sample_dir = os.path.join(self.data_dir, date)
            with h5py.File(os.path.join(sample_dir, 'sample.hdf5'), 'r') as f:
                video_length = f['video_data'].shape[1]   # C, T, H, W
            sample_num = video_length // self.T if self.T != -1 else 1
            for i in range(sample_num):
                sample = {}
                sample['location'] = sample_dir
                sample['start_idx'] = i * self.T
                sample['video_length'] = video_length
                self.data_list.append(sample)

class UBFC(BaseDataset):
    def __init__(self, data_dir='/root/data/UBFC_PIC/CA_PIC', train='train', T=-1, w=64, h=64):
        super().__init__(data_dir, train, T, w, h)

    def get_data_list(self):
        subject_list = os.listdir(self.data_dir)
        subject_list.remove('subject11')    # exist error hr (eq 0) in sample
        subject_list.remove('subject18')    # exist error hr (eq 0) in sample
        subject_list.remove('subject20')    # exist error hr (eq 0) in sample
        subject_list.remove('subject24')    # exist error hr (eq 0) in sample
        subject_list.sort()

        if self.train == 'train':
            subject_list = subject_list[:30]
        elif self.train == 'test':
            subject_list = subject_list[30:]
        elif 'all' in self.train:
            pass
        else:
            raise NotImplementedError
        
        for subject in subject_list:
            file_dir = os.path.join(self.data_dir, subject)
            with h5py.File(os.path.join(file_dir, 'sample.hdf5'), 'r') as f:
                video_length = f['video_data'].shape[1]   # C, T, H, W
            sample_num = video_length // self.T if self.T != -1 else 1
            for i in range(sample_num):
                sample = {}
                sample['location'] = file_dir
                sample['start_idx'] = i * self.T
                sample['video_length'] = video_length
                self.data_list.append(sample)

class VIPL(BaseDataset):
    def __init__(self, data_dir = '/root/data/VIPL', train='train', T=-1, w=64, h=64, fold=5):
        self.fold = fold
        super().__init__(data_dir, train, T, w, h)

    def get_data_list(self):
        self.fold_split_dir = os.path.join(self.data_dir, 'VIPL_fold')
        self.fold_list = []
        for i in range(1, 6):
            mat_path = os.path.join(self.fold_split_dir, f'fold{i}.mat')
            mat = sio.loadmat(mat_path)
            self.fold_list.append(mat[f'fold{i}'].reshape(-1))

        if self.train == 'train':
            # all flod except self.fold
            fold = np.concatenate(self.fold_list[:self.fold - 1] + self.fold_list[self.fold:])
        elif self.train == 'test':
            fold = self.fold_list[self.fold - 1]
        elif 'all' in self.train:
            fold = np.concatenate(self.fold_list)
        else:
            raise NotImplementedError

        # print(fold)
        p_lists = [f'p{i}' for i in fold]
        p_lists.sort()
        for p_name in p_lists:
            p_root = os.path.join(self.data_dir, p_name)
            v_lists = os.listdir(p_root)
            v_lists.sort()
            for v_name in v_lists:
                v_root = os.path.join(p_root, v_name)
                source_lists = os.listdir(v_root)
                if 'source4' in source_lists:
                    source_lists.remove('source4')
                source_lists.sort()
                for source_name in source_lists:
                    if os.path.join(v_root, source_name) in [f'{self.data_dir}/p32/v7/source3', f'{self.data_dir}/p45/v1/source2', \
                                                            f'{self.data_dir}/p19/v2/source2']: # 32-7-3, 45-1-2 lack of wave, 19-2-2 lack of frame 
                        continue
                    with h5py.File(os.path.join(v_root, source_name, 'sample.hdf5'), 'r') as f:
                        video_length = f['video_data'].shape[1]   # C, T, H, W
                    sample_num = video_length // self.T if self.T != -1 else 1
                    for i in range(sample_num):
                        sample = {}
                        sample['location'] = os.path.join(v_root, source_name)
                        sample['start_idx'] = i * self.T
                        sample['video_length'] = video_length
                        self.data_list.append(sample)
                    # print(os.path.join(v_root, source_name))

class BUAA(BaseDataset):
    def __init__(self, data_dir = '/data2/chushuyang/BUAA', train='train', T=-1, w=64, h=64):
        super().__init__(data_dir, train, T, w, h)

    def get_data_list(self):
        sub_list = os.listdir(self.data_dir)
        sub_list.sort()
        if self.train == 'train':
            sub_list = sub_list[:10]
        elif self.train == 'test':
            sub_list = sub_list[10:]
        elif 'all' in self.train:
            pass
        else:
            raise NotImplementedError
        
        for sub in sorted(sub_list):
            sub_path = os.path.join(self.data_dir, sub)
            for lux in sorted(os.listdir(sub_path)):
                lux_rate = lux.split(' ')[1]
                if float(lux_rate) < 10:
                    continue
                file_dir = os.path.join(sub_path, lux)
                with h5py.File(os.path.join(file_dir, 'sample.hdf5'), 'r') as f:
                    video_length = f['video_data'].shape[1]   # C, T, H, W
                sample_num = video_length // self.T if self.T != -1 else 1
                for i in range(sample_num):
                    sample = {}
                    sample['location'] = file_dir
                    sample['start_idx'] = i * self.T
                    sample['video_length'] = video_length
                    self.data_list.append(sample)

class MMSE(BaseDataset):
    def __init__(self, data_dir = '/data3/wuruize/MMSE-HR_process', train='test', T=-1, w=64, h=64):
        super().__init__(data_dir, train, T, w, h)

    def get_data_list(self):
        with open(f'{self.data_dir}/white_name.txt', 'r') as f:
            white_name_list = f.readlines()
        white_name_list = [i.strip() for i in white_name_list]
        subject_list = os.listdir(self.data_dir)
        subject_list.sort()
        for subject in subject_list:
            subject_path = os.path.join(self.data_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            source_list = os.listdir(subject_path)
            source_list.sort()
            for source in source_list:
                if os.path.join(subject_path, source) not in white_name_list:
                    continue
                with h5py.File(os.path.join(subject_path, source, 'sample.hdf5'), 'r') as f:
                    video_length = f['video_data'].shape[1]   # C, T, H, W
        
                sample = {}
                sample['location'] = os.path.join(subject_path, source)
                sample['start_idx'] = self.T
                sample['video_length'] = video_length
                self.data_list.append(sample)
                
class COHFACE(BaseDataset):
    def __init__(self, data_dir='/root/data/pure', train='train', T=-1, w=64, h=64):
        super().__init__(data_dir, train, T, w, h)

    def get_data_list(self):
        test_txt = '/data/chushuyang/COHFACE/cohface/protocols/all/test.txt'
        all_txt = '/data/chushuyang/COHFACE/cohface/protocols/all/all.txt'
        train_txt = '/data/chushuyang/COHFACE/cohface/protocols/all/train.txt'
        
        if self.train == 'train':
            with open(train_txt, 'r') as f:
                all_list = f.readlines()
        elif self.train == 'test':
            with open(test_txt, 'r') as f:
                all_list = f.readlines()
        elif 'all' in self.train:
            with open(all_txt, 'r') as f:
                all_list = f.readlines()
        else:
            raise NotImplementedError

        for data_sample in all_list:
            [px, v_src] = data_sample.strip().split('/')[:2]
            px_path = os.path.join(self.data_dir, px)
            if (px, v_src) in [('11', '2'), ('3', '2'), ('3', '3'), ('6', '2')]:
                continue
            sample_dir = os.path.join(px_path, v_src)
            with h5py.File(os.path.join(sample_dir, 'sample.hdf5'), 'r') as f:
                video_length = f['video_data'].shape[1]   # C, T, H, W
            sample_num = video_length // self.T if self.T != -1 else 1
            for i in range(sample_num):
                sample = {}
                sample['location'] = sample_dir
                sample['start_idx'] = i * self.T
                sample['video_length'] = video_length
                self.data_list.append(sample)


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    dataset = MMSE(train=False, T=160)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data in enumerate(dataloader):
        print(f'{data["video"].shape} {data["ecg"].shape}')
        print(data['clip_avg_hr'])
        break