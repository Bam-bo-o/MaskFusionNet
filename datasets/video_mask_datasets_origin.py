import cv2
import os
import json
import math
import h5py
import scipy
import numpy as np
import scipy.io as sio
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from torchvision import transforms

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

class Normaliztion(object):
    '''
        normalize into [-1, 1]
        image = (image - 127.5)/127.5
    '''
    def __call__(self, sample):
        video_x = sample['video']
        new_video_x = (video_x - 127.5) / 127.5
        sample['video'] = new_video_x
        return sample


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio=0.25):  # [args.window_size, args.mask_ratio]
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

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio=0.75):  # [args.window_size, args.mask_ratio]
        self.frames, self.height, self.width = input_size  # [40,4,4]
        self.num_patches =  int(self.height * self.width * self.frames)  # 1024 = 32 x 32  #16
        # self.total_patches = self.frames * self.num_patches_per_frame  # 640
        self.num_masks = int(mask_ratio * self.num_patches)  # 12
        # self.total_masks = self.frames * self.num_masks_per_frame  # 480

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_masks
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_masks),
            np.ones(self.num_masks),
        ])  # [num_patches,]
        np.random.shuffle(mask)  # [num_patches,]
        # mask = np.tile(mask_per_frame, (int(self.frames),1)).flatten()  # [self.frames,num_patches] --> [self.frames*num_patches,]
        return mask


# 返回T帧或者整个视频
# 每一个数据集实现自己的读视频，读帧率的方法
# T=-1表示读整个视频
class BaseDataset(Dataset):
    def __init__(self, data_dir, train=True, T=-1, transform_rate=30, w=128, h=128, limit_batches=0):
        """
        :param data_dir: 数据集的根目录
        :param train: 是否是训练集
        :param T: 读取的帧数，-1表示读取整个视频
        :param transform_rate: getitem返回的视频的帧率和ecg的采样率
        """
        self.data_dir = data_dir
        self.train = train
        self.T = T
        self.transform_rate = transform_rate
        self.transforms = transforms.Compose([Normaliztion()])  #transforms_list
        self.w = w
        self.h = h
        self.limit_batches = limit_batches
        self.rate_threshold = 3 # 帧率差距大于这个值才会重采样
        self.data_list = list()
        self.get_data_list() # 返回一个list，每个元素是一个dict，frame_path(一个列表), ecg, frame_rate(原始的，需要更改频率), ecg_rate（原始的需要更改频率）
        self.mask_generator = TubeMaskingGenerator(input_size=(self.T // 4, self.h // 32, self.w // 32), mask_ratio=0.75)

    def get_data_list(self):
        raise NotImplementedError

    def __len__(self):
        if self.limit_batches > 0:
            return int(len(self.data_list) * self.limit_batches)
        return len(self.data_list)

    def __getitem__(self, index):
        """return: video, ecg, transform_rate, frame_start, frame_end"""
        # 读视频， 读帧率
        video = torch.from_numpy(self.read_video(self.data_list[index]["frame_path"])).permute(3, 0, 1, 2).float() # c, t, h, w
        # 重采样
        if abs(self.transform_rate - self.data_list[index]["frame_rate"]) > self.rate_threshold:
            if self.T == -1:
                video = torch.nn.functional.interpolate(video.unsqueeze(0),\
                     scale_factor=(self.transform_rate / self.data_list[index]["frame_rate"], 1, 1), mode="trilinear").squeeze(0)
            else:
                video = torch.nn.functional.interpolate(video.unsqueeze(0),\
                     size=(self.T, self.h, self.w), mode="trilinear", align_corners=True).squeeze(0)
        if self.T != video.shape[0] and self.T != -1:
            video = torch.nn.functional.interpolate(video.unsqueeze(0),\
                 size=(self.T, self.h, self.w), mode="trilinear", align_corners=True).squeeze(0)
       
        sample = {"video": video, "mask": self.mask_generator()}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def read_video(self, frame_path):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        video_x = np.zeros((len(frame_path), self.w, self.h, 3))
        for i, frame in enumerate(frame_path):
            imageBGR = cv2.imread(frame)
            try:
                imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
            except:
                print(f'error in {frame}')
            video_x[i, :, :, :] = cv2.resize(imageRGB, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        return video_x

    # fps = frame_rate = frame_num / duration, duration = frame_num / frame_rate, frame_num = duration * frame_rate
    # 采样前后的duration相同，所以重采样后的视频长度 = 重采样前的视频长度 * 重采样后的帧率 / 重采样前的帧率
    # 给定采样后的帧数T, 重采样前的视频长度 = 重采样后的视频长度 * 重采样前的帧率 / 重采样后的帧率
    def calculate_chunk_split(self, old_length, old_rate, new_rate, T):
        if T == -1:
            return [[0, old_length]]
        elif abs(self.transform_rate - old_rate) <= self.rate_threshold: # 如果在误差范围之内，则不调频
            chunk_split = list(range(0, old_length - T + 1, T))
            chunk_list = []
            for chunk in chunk_split:
                if chunk + T <= old_length:
                    chunk_list.append([chunk, chunk + T])
            return chunk_list
        else:
            new_length = round(old_length * new_rate / old_rate) # 重采样后的视频长度，向下取整
            if new_length < T:
                raise ValueError("video length is too short")
            else:
                map_to_old = int(T * old_rate / new_rate) + 1 # 重采样后的帧数，向上取整，目的是为了避免某些边界情况，这里的划分满足了，但是调频后的视频不满足。与时间维度插值的mathcail一致
                chunk_split = list(range(0, old_length - map_to_old + 1, map_to_old)) #
                chunk_list = []
                for chunk in chunk_split:
                    if chunk + map_to_old <= old_length:
                        chunk_list.append([chunk, chunk + map_to_old])
            return chunk_list


class PURE(BaseDataset):
    def __init__(self, data_dir, train=True, T=-1, transform_rate=30, w=128, h=128, limit_batches=0):
        super().__init__(data_dir, train, T, transform_rate, w, h, limit_batches)

    def get_data_list(self):
        date_list = os.listdir(self.data_dir)
        date_list.sort()
        train_list = ['06-01', '06-03', '06-04', '06-05', '06-06', '08-01', '08-02', '08-03', '08-04', '08-05', '08-06',\
                    '05-01', '05-02', '05-03', '05-04', '05-05', '05-06', '01-01', '01-02', '01-03', '01-04', '01-05', '01-06',\
                    '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '09-01', '09-02', '09-03', '09-04', '09-05', '09-06',\
                    '07-01', '07-02', '07-03', '07-04', '07-05', '07-06']
        if self.train == 'train':
            date_list = [i for i in date_list if i in train_list]
        else:
            date_list = [i for i in date_list if i not in train_list]
        
        for date in date_list:
            # read video
            pic_type = 'align_crop_pic'
            video_dir = os.path.join(self.data_dir, date, pic_type)
            # read label
            json_file = os.path.join(self.data_dir, date, date + ".json")
            with open(json_file, 'r') as f:
                data = json.load(f)
            ecg_time_stamp = np.array([i['Timestamp'] for i in data['/FullPackage']])
            ecg = np.array([i['Value']['waveform'] for i in data['/FullPackage']])
            video_time_stamp = np.array([i['Timestamp'] for i in data['/Image']])
            # 基于json文件中的时间戳，保证视频图片文件名的顺序正确
            frame_path = []
            for i in range(len(video_time_stamp)):
                frame_path.append(os.path.join(video_dir, f"Image{video_time_stamp[i]}.png"))

            assert len(ecg_time_stamp) == len(ecg)

            ecg_time_diffs = np.diff(ecg_time_stamp / 1e9)
            ecg_rate = 1 / ecg_time_diffs.mean()
            frame_time_diffs = np.diff(video_time_stamp / 1e9)
            frame_rate = 1 / frame_time_diffs.mean()

            ecg_chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, self.transform_rate, self.T)
            frame_chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)

            try:
                assert len(ecg_chunk_split) == len(frame_chunk_split), "ecg and frame chunk split is not equal"
            except:
                continue

            for idx in range(len(ecg_chunk_split)):
                start_idx, end_idx = ecg_chunk_split[idx]
                ecg_select = ecg[start_idx:end_idx]
                start_idx, end_idx = frame_chunk_split[idx]
                # print(idx, start_idx, end_idx)
                frame_select = frame_path[start_idx:end_idx]
                self.data_list.append({'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate, 'ecg_rate': ecg_rate})


class UBFC(BaseDataset):
    def __init__(self, data_dir='/root/data/UBFC_RAW', train=True, T=-1, transform_rate=30, limit_batches=0, w=128, h=128):
        super().__init__(data_dir, train, T, transform_rate, w, h, limit_batches)

    def get_data_list(self):
        subject_list = os.listdir(self.data_dir)
        subject_list.remove('subject11')
        subject_list.remove('subject18')
        subject_list.remove('subject20')
        subject_list.remove('subject24') # 这个视频会导致在PhysNet中心率出现负值（减去40之后）
        subject_list.sort()
        if self.train == 'train':
            subject_list = subject_list[:30]
        else:
            subject_list = subject_list[30:]
        for subject in subject_list:
            video_dir = os.path.join('/root/data/UBFC_PIC/CA_PIC', subject)
            file_dir = os.path.join(self.data_dir, subject)
            frame_list = os.listdir(video_dir)
            frame_list_int = [int(i.split(".")[0]) for i in frame_list]
            frame_list_int.sort() # 转为整数保证视频图片文件名的顺序正确
            frame_path = []
            for frame_name in frame_list_int:
                frame_path.append(os.path.join(video_dir, f"{frame_name}.png"))
            # read label
            with open(os.path.join(file_dir, 'ground_truth.txt'), 'r') as f:
                data = f.readlines()
            data_timestamp = np.array([float(strr.replace('e','E')) for strr in list(data[2].split())]) # 每一帧的时间戳，单位s, 起始的时间戳为0
            data_Hr = np.array([float(strr.replace('e','E')) for strr in list(data[1].split())])
            data_ecg = np.array([float(strr.replace('e','E')) for strr in list(data[0].split())])
            assert len(data_timestamp) == len(data_Hr) == len(data_ecg)
            assert len(data_timestamp) == len(frame_path)
            time_diffs = np.diff(data_timestamp)
            frame_rate = 1 / time_diffs.mean()
            chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)
            for start_idx, end_idx in chunk_split:
                frame_select = frame_path[start_idx:end_idx]
                ecg_select = data_ecg[start_idx:end_idx]
                self.data_list.append({'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate, 'ecg_rate': frame_rate})

class VIPL(BaseDataset):
    def __init__(self, data_dir = '/root/data/VIPL', train=True, T=-1, transform_rate=30, w=128, h=128, limit_batches=0, fold=5):
        self.fold = fold
        super().__init__(data_dir, train, T, transform_rate, w, h, limit_batches)

    def get_data_list(self):
        self.fold_split_dir = os.path.join(self.data_dir, "VIPL_fold")
        self.fold_list = []
        for i in range(1, 6):
            mat_path = os.path.join(self.fold_split_dir, f"fold{i}.mat")
            mat = sio.loadmat(mat_path)
            self.fold_list.append(mat[f"fold{i}"].reshape(-1))

        if self.train == 'train':
            # all flod except self.fold
            fold = np.concatenate(self.fold_list[:self.fold - 1] + self.fold_list[self.fold:])
        else:
            fold = self.fold_list[self.fold - 1]

        # print(fold)
        p_lists = [f"p{i}" for i in fold]
        p_lists.sort()
        bad_img = []
        for p_name in p_lists:
            p_root = os.path.join(self.data_dir, p_name)
            v_lists = os.listdir(p_root)
            v_lists.sort()
            for v_name in v_lists:
                v_root = os.path.join(p_root, v_name)
                source_lists = os.listdir(v_root)
                if "source4" in source_lists:
                    source_lists.remove("source4")
                source_lists.sort()
                for source_name in source_lists:
                    # read video
                    pic_type = 'align_crop_pic'    # NOTE: pic, align_crop_pic, aligned_pic
                    frame_dir = os.path.join(v_root, source_name, pic_type)
                    if frame_dir in [f'{self.data_dir}/p32/v7/source3/{pic_type}', f'{self.data_dir}/p45/v1/source2/{pic_type}', \
                                    f'{self.data_dir}/p19/v2/source2/{pic_type}']: # 32-7-3, 45-1-2 lack of wave, 19-2-2 lack of frame 
                        continue
                    frame_list = os.listdir(frame_dir)
                    try:
                        frame_list_int = [int(i.split(".")[0]) for i in frame_list]
                        frame_list_int.sort() # 转为整数保证视频图片文件名的顺序正确
                    except:
                        print(frame_dir)
                    frame_path = []
                    for frame_name in frame_list_int:
                        frame_path.append(os.path.join(frame_dir, f"{frame_name:0>5}.png")) # NOTE: pic : frame_name, align_crop_pic : frame_name:0>5
                    # read label
                    gt_HR_csv = os.path.join(v_root, source_name, "gt_HR.csv")
                    gt_SpO2_csv = os.path.join(v_root, source_name, "gt_SpO2.csv")
                    wave_csv = os.path.join(v_root, source_name, "wave.csv")
                    with open(wave_csv, 'r') as f:
                        data = f.readlines()
                        data = data[1:]
                        ecg = np.array([int(i) for i in data])
                    # read time.txt calculate frame_rate, ecg_rate
                    rate_txt = os.path.join(v_root, source_name, "rate.txt")
                    with open(rate_txt, 'r') as f:
                        data = f.read().splitlines()
                        # print(data)
                        frame_rate = float(data[0].split(":")[1])
                        total_frame = float(data[1].split(":")[1])
                        total_time = float(data[2].split(":")[1])
                    ecg_rate = len(ecg) / total_time
                    try: # 有可能出现按照目标帧率下，帧数不够的情况
                        ecg_chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, self.transform_rate, self.T)
                        frame_chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)
                    except:
                        continue
                    split_length = min(len(ecg_chunk_split), len(frame_chunk_split))
                    ecg_chunk_split = ecg_chunk_split[:split_length]
                    frame_chunk_split = frame_chunk_split[:split_length]

                    assert len(ecg_chunk_split) == len(frame_chunk_split), "ecg and frame chunk split is not equal"

                    for idx in range(len(ecg_chunk_split)):
                        start_idx, end_idx = ecg_chunk_split[idx]
                        ecg_select = ecg[start_idx:end_idx]
                        start_idx, end_idx = frame_chunk_split[idx]
                        # print(idx, start_idx, end_idx)
                        frame_select = frame_path[start_idx:end_idx]
                        self.data_list.append({'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate})


#只用了 lux >= 10 的
class BUAA(BaseDataset):
    def __init__(self, data_dir='/data2/chushuyang/BUAA', train=True, T=-1, transform_rate=30, limit_batches=0):
        super().__init__(data_dir, train, T, transform_rate, limit_batches=limit_batches)

    def get_data_list(self):
        subject_list = os.listdir(self.data_dir)
        subject_list.sort()

        if self.train == 'train':
            subject_list = [subject_list[i] for i in range(10)]

        else:
            subject_list = [subject_list[i] for i in range(10,13)]

        for subject in subject_list:
            sub_dir = os.path.join(self.data_dir, subject)

            lux_dir = os.listdir(os.path.join(self.data_dir, sub_dir))
            lux_dir = [i for i in lux_dir if float(i[3:]) >= 10] # lux >= 10
            for lux in lux_dir:
                video_dir  = os.path.join(self.data_dir, sub_dir, lux)
                video_dir_list = os.listdir(video_dir)
                video_name  = [i for i in video_dir_list if "avi" in i][0]

                pic_type = 'align_crop_pic'    # NOTE: pic, align_crop_pic, aligned_pic
                frame_dir = os.path.join(video_dir, pic_type)
                frame_list = os.listdir(frame_dir)
                try:
                    frame_list_int = [int(i.split(".")[0]) for i in frame_list]
                    frame_list_int.sort() # 转为整数保证视频图片文件名的顺序正确
                except:
                    print(frame_dir)
                frame_path = []
                for frame_name in frame_list_int:
                    frame_path.append(os.path.join(frame_dir, f"{frame_name:0>5}.png"))

                hr_name  = video_name.split(".avi")[0] + ".csv"
                ppg_name  = video_name.split(".avi")[0] + "_wave.csv"


                df = pd.read_csv(os.path.join(video_dir,ppg_name))
                data_ecg = np.array(df).reshape(-1)

                ppg = data_ecg  # segNum*160*2 这是从帧数的角度考虑扩增2倍
                x = np.linspace(1,len(ppg) ,len(ppg) )  # 160/30*60=160*2
                #print(len(ppg),len(x),len(os.listdir(tempPath + '/pic')),segNum) #3817 3840 1920 
                funcInterpolate = scipy.interpolate.interp1d(x, ppg, kind="slinear")

                xNew = np.linspace(1, len(ppg) , 1800 )
                data_ecg = funcInterpolate(xNew)

                frame_rate = 30.0
                chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)
                for start_idx, end_idx in chunk_split:
                    frame_select = frame_path[start_idx:end_idx]
                    ecg_select = data_ecg[start_idx:end_idx]
                    self.data_list.append({'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate, 'ecg_rate': frame_rate})


class COHFACE(BaseDataset):
    def __init__(self, data_dir='/root/data/COHFACE/cohface', train=True, T=-1, transform_rate=30, limit_batches=0):
        super().__init__(data_dir, train, T, transform_rate, limit_batches)

    def get_data_list(self):
        protocol_train = '/root/data/COHFACE/protocols/all_train.txt'
        protocol_test = '/root/data/COHFACE/protocols/all_test.txt'
        p_lists = []
        if self.train == 'train':
            with open(protocol_train, 'r') as f:
                data = f.readlines()
        else:
            with open(protocol_test, 'r') as f:
                data = f.readlines()
        for line in data:
            p_lists.append(os.path.join(self.data_dir, line.split()[0]))

        for data_path in p_lists:
            # read video
            pic_type = 'align_crop_pic'    # NOTE: pic, align_crop_pic, aligned_pic
            backup_pic_type = 'pic'
            frame_dir = os.path.join(data_path, pic_type)
            frame_list = os.listdir(frame_dir)
            if len(frame_list) == 0:
                frame_dir = os.path.join(data_path, backup_pic_type)
                frame_list = os.listdir(frame_dir)
            try:
                frame_list_int = [int(i.split(".")[0]) for i in frame_list]
                frame_list_int.sort() # 转为整数保证视频图片文件名的顺序正确
            except:
                print(frame_dir)
            frame_path = []
            for frame_name in frame_list_int:
                frame_path.append(os.path.join(frame_dir, f"{frame_name:0>5}.png")) # NOTE: pic : frame_name, align_crop_pic : frame_name:0>5
            # read label
            ecg_path = os.path.join(data_path, 'data.hdf5')
            with h5py.File(ecg_path, 'r') as f:
                ecg = np.array(f['pulse'])
            # read time.txt calculate frame_rate, ecg_rate
            rate_txt = os.path.join(data_path, 'rate.txt')
            with open(rate_txt, 'r') as f:
                data = f.read().splitlines()
                frame_rate = float(data[0].split(":")[1])
                total_time = float(data[2].split(":")[1])
            ecg_rate = len(ecg) / total_time
            try: # 有可能出现按照目标帧率下，帧数不够的情况
                ecg_chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, self.transform_rate, self.T)
                frame_chunk_split = self.calculate_chunk_split(len(frame_path), frame_rate, self.transform_rate, self.T)
            except:
                continue
            split_length = min(len(ecg_chunk_split), len(frame_chunk_split))
            ecg_chunk_split = ecg_chunk_split[:split_length]
            frame_chunk_split = frame_chunk_split[:split_length]
            assert len(ecg_chunk_split) == len(frame_chunk_split), "ecg and frame chunk split is not equal"
            for idx in range(len(ecg_chunk_split)):
                start_idx, end_idx = ecg_chunk_split[idx]
                ecg_select = ecg[start_idx:end_idx]
                start_idx, end_idx = frame_chunk_split[idx]
                # print(idx, start_idx, end_idx)
                frame_select = frame_path[start_idx:end_idx]
                self.data_list.append({'frame_path': frame_select, 'ecg': ecg_select, 'frame_rate': frame_rate, 'ecg_rate': ecg_rate})


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    dataset = VIPL_yzt(train=False, T=-1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data in enumerate(dataloader):
        # print(i, data['frame_path'][0])
        print(data['video'].shape)
        print(data['ecg'].shape)
        print(data['frameRate'])
        print(data['ecgRate'])
        print(data['clipAverageHR'])
        print(data['clipAverageHR'].shape)
        break