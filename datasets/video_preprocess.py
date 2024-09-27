## 加载图像数据
import cv2
import os
import argparse
import sys
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utils import crop_faces

def video_preprocess(dataset = "", root_dir = ""):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        fnames = os.listdir(dir)
        for fname in fnames:
            if fname.split('.')[0][-1] == ')':
                os.remove(os.path.join(dir, fname))
                print(f'---------  remove {os.path.join(dir, fname)} for endwith ([0-9])  ---------')
        for fname in sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0])):
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
        return images
    
    def make_dataset_from_video(video):
        images = []
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fname = f'{i:05d}'
            if ret:
                images.append((fname, frame))
        cap.release()
        return images
    
    if dataset == "VFHQ":
        ## 预处理
        save_dir = f'{root_dir}_preprocessed'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, dir in enumerate(os.listdir(root_dir)):
            print(f'processing {dir} [{idx} / {len(os.listdir(root_dir))}] ...')
            dir_path = os.path.join(root_dir, dir)
            save_path = os.path.join(save_dir, dir)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            files = make_dataset(dir_path)
            
            image_size = 256
            scale = 1.0
            center_sigma = 1.0
            xy_sigma = 3.0
            use_fa = False

            crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

            if crops is None:
                print(f'too less face detected in file {dir_path} -----------------')
                continue

            for i in range(len(crops)):
                img = crops[i]
                img.save(os.path.join(save_path, files[i][0] + '.png'))
            print(f'generated {len(crops)} png images in file {save_path}')
    
    elif dataset == "VIPL":
        for p in sorted(os.listdir(root_dir)):
            if not p.startswith('p'):
                continue
            p_path = os.path.join(root_dir, p)
            for v in sorted(os.listdir(p_path)):
                v_path = os.path.join(p_path, v)
                for source in sorted(os.listdir(v_path)):
                    if source == 'source4':
                        print(f'jump over {v_path}/source4 -----------------')
                        continue
                    save_path = os.path.join(v_path, source, 'align_crop_pic')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    elif os.path.exists(os.path.join(v_path, source, 'pic')):
                        if len(os.listdir(save_path)) == len(os.listdir(os.path.join(v_path, source, 'pic'))):
                            print(f'already processed {save_path} -----------------')
                            continue

                    source_path = os.path.join(v_path, source)
                    video_path = os.path.join(source_path, 'video.avi')
                    print(f'processing {video_path}  ...')
                    
                    files = make_dataset_from_video(video_path)

                    image_size = 128
                    # scale = 1.0  # 仅 align
                    scale = 0.8  # align + crop
                    center_sigma = 1.0
                    xy_sigma = 3.0
                    use_fa = False

                    crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

                    if crops is None:
                        print(f'too less face detected in video {video_path} -----------------')
                        continue

                    for i in range(len(crops)):
                        img = crops[i]
                        img.save(os.path.join(save_path, files[i][0] + '.png'))
                    print(f'generated {len(crops)} png images in file {save_path}')

    elif dataset == "UBFC":
        for idx, subject in enumerate(sorted(os.listdir(root_dir))):
            print(f'processing {subject} [{idx} / {len(os.listdir(root_dir))}] ...')
            subject_path = os.path.join(root_dir, subject)

            video_path = os.path.join(subject_path, '001vid.avi')
            save_path = os.path.join(subject_path, 'align_crop_pic')

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            files = make_dataset_from_video(video_path)

            image_size = 128
            scale = 0.8
            center_sigma = 1.0
            xy_sigma = 3.0
            use_fa = False

            crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

            if crops is None:
                print(f'too less face detected in file {video_path} -----------------')
                continue

            for i in range(len(crops)):
                img = crops[i]
                img.save(os.path.join(save_path, files[i][0] + '.png'))
            print(f'generated {len(crops)} png images in file {save_path}')

    elif dataset == "PURE":
        date_list = sorted(os.listdir(root_dir))
        for date in date_list:
            # read video
            video_dir = os.path.join(root_dir, date, date)
            video_save_dir = os.path.join(root_dir, date, 'align_crop_pic')
            if not os.path.exists(video_save_dir):
                os.makedirs(video_save_dir)
            else:
                if len(os.listdir(video_save_dir)) == len(os.listdir(video_dir)):
                    print(f'already processed {video_save_dir} -----------------')
                    continue
            files = []
            for fname in sorted(os.listdir(video_dir)):
                if is_image_file(fname):
                    path = os.path.join(video_dir, fname)
                    fname = fname.split('.')[0]
                    files.append((fname, path))
                else:
                    raise ValueError(f'frame {fname} is not png')
            
            image_size = 128
            scale = 0.8
            center_sigma = 1.0
            xy_sigma = 3.0
            use_fa = False

            crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

            if crops is None:
                print(f'too less face detected in file {video_dir} -----------------')
                continue

            for i in range(len(crops)):
                img = crops[i]
                img.save(os.path.join(video_save_dir, files[i][0] + '.png'))
            print(f'generated {len(crops)} png images in file {video_save_dir}')

    elif dataset == "COHFACE":
        for px in sorted(os.listdir(root_dir)):
            px_path = os.path.join(root_dir, px)
            if not os.path.isdir(px_path) or px == 'protocols':
                continue
            for v_src in sorted(os.listdir(px_path)):
                v_path = os.path.join(px_path, v_src, 'data.avi')
                save_path = os.path.join(px_path, v_src, 'align_crop_pic')
                print(f'processing {v_path}  ...')
                cap = cv2.VideoCapture(v_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    if len(os.listdir(save_path)) == frame_count:
                        print(f'already processed {save_path} -----------------')
                        continue

                files = make_dataset_from_video(v_path)

                image_size = 128
                scale = 0.8
                center_sigma = 1.0
                xy_sigma = 3.0
                use_fa = False

                crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

                if crops is None:
                    print(f'too less face detected in video {v_path} -----------------')
                    continue

                for i in range(len(crops)):
                    img = crops[i]
                    img.save(os.path.join(save_path, files[i][0] + '.png'))
                print(f'generated {len(crops)} png images in file {save_path}')

    elif dataset == "BUAA":
        BUAA_sub_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        for sub in sorted(os.listdir(root_dir)):
            sub_idx = int(sub.split(' ')[1])
            sub_path = os.path.join(root_dir, sub)
            for lux in sorted(os.listdir(sub_path)):
                lux_rate = lux.split(' ')[1]
                if float(lux_rate) < 10:
                    continue
                lux_path = os.path.join(sub_path, lux)
                save_path = os.path.join(lux_path, 'align_crop_pic')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    if len(os.listdir(save_path)) > 1:
                        print(f'already processed {save_path} -----------------')
                        continue

                video_path = os.path.join(lux_path, f'lux{lux_rate}_{BUAA_sub_name[sub_idx-1]}.avi')
                print(f'processing {video_path}  ...')
                
                files = make_dataset_from_video(video_path)

                image_size = 128
                # scale = 1.0  # 仅 align
                scale = 0.8  # align + crop
                center_sigma = 1.0
                xy_sigma = 3.0
                use_fa = False

                crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)

                if crops is None:
                    print(f'too less face detected in video {video_path} -----------------')
                    continue

                for i in range(len(crops)):
                    img = crops[i]
                    img.save(os.path.join(save_path, files[i][0] + '.png'))
                print(f'generated {len(crops)} png images in file {save_path}')

    elif dataset == "MMSE":
        new_root_dir = f'{root_dir}_preprocessed'
        sub_dir_list = ["first 10 subjects 2D", "T10_T11_30Subjects"]
        sub_label_dir_list = ["first 10 subjects Phydata released/Phydata", "T10_T11_30PhyBPHRData"]

        for i in [0, 1]:
            sub_dir = sub_dir_list[i]
            sub_label_dir = sub_label_dir_list[i]

            subject_list = sorted(os.listdir(os.path.join(root_dir, sub_dir)))
            for subject in subject_list:
                subject_dir = os.path.join(root_dir, sub_dir, subject)
                subject_label_dir = os.path.join(root_dir, sub_label_dir, subject)

                for T in sorted(os.listdir(subject_dir)):
                    video_dir = os.path.join(subject_dir, T)

                    new_T_dir = os.path.join(new_root_dir, subject, T)
                    video_save_dir = os.path.join(new_root_dir, subject, T, "pic")
                    if not os.path.exists(video_save_dir):
                        os.makedirs(video_save_dir)
                    if len(os.listdir(video_save_dir)) == len(os.listdir(video_dir)):
                        print(f'already processed {video_save_dir} -----------------')
                        continue

                    files = []
                    for fname in sorted(os.listdir(video_dir)):
                        if is_image_file(fname):
                            path = os.path.join(video_dir, fname)
                            fname = fname.split('.')[0]
                            files.append((fname, path))
                        else:
                            raise ValueError(f'frame {fname} is not png')

                    image_size = 128
                    scale = 0.8
                    center_sigma = 1.0
                    xy_sigma = 3.0
                    use_fa = False

                    print(f'processing {video_dir} ...')
                    crops, orig_images, quads = crop_faces(image_size, files, scale, center_sigma, xy_sigma, use_fa)
                    if crops is None:
                        print(f'too less face detected in file {video_dir} -----------------')
                        continue

                    for i in range(len(crops)):
                        img = crops[i]
                        img.save(os.path.join(video_save_dir, files[i][0] + '.png'))
                    print(f'generated {len(crops)} png images in file {video_save_dir}')

                    # move label
                    BP_txt = os.path.join(subject_label_dir, T, "BP_mmHg.txt")
                    Pulse_txt = os.path.join(subject_label_dir, T, "Pulse Rate_BPM.txt")

                    new_BP_txt = os.path.join(new_T_dir, "BP_mmHg.txt")
                    new_Pulse_txt = os.path.join(new_T_dir, "Pulse Rate_BPM.txt")

                    os.system(f'cp {BP_txt} {new_BP_txt}')
                    os.system(f'cp {Pulse_txt} {new_Pulse_txt}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COHFACE', help='VFHQ or VIPL or UBFC or PURE or COHFACE')
    parser.add_argument('--dataset_dir', type=str, default='/home/haoyum/download/COHFACE', help='dataset dir')
    args = parser.parse_args()
    video_preprocess(args.dataset, args.dataset_dir)
