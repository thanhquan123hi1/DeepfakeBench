import torch
import random
import numpy as np
import os
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

class pairDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # 1. Phân loại danh sách ảnh Real (0) và Fake (1)
        self.fake_imglist = [(img, label, 1) for img, label in zip(self.image_list, self.label_list) if label != 0]
        self.real_imglist = [(img, label, 0) for img, label in zip(self.image_list, self.label_list) if label == 0]

        # 2. Xây dựng Real Pool theo Frame (Khắc phục lỗi dấu \)
        self.real_pool = {}
        for img_path, spe_label, label in self.real_imglist:
            # CHUẨN HÓA PATH: Ép mọi dấu \ thành /
            clean_path = img_path.replace('\\', '/')
            parts = clean_path.split('/')
            
            frame_name = parts[-1]   # Ví dụ: '000.png'
            video_id = parts[-2]     # Ví dụ: '188'
            
            if video_id not in self.real_pool:
                self.real_pool[video_id] = {}
            self.real_pool[video_id][frame_name] = (img_path, spe_label, label)

    def __getitem__(self, index, norm=True):
        fake_image_path, fake_spe_label, fake_label = self.fake_imglist[index]
        
        # CHUẨN HÓA PATH FAKE
        clean_fake_path = fake_image_path.replace('\\', '/')
        parts = clean_fake_path.split('/')
        frame_name = parts[-1]
        fake_video_id = parts[-2]
        
        # Trích xuất ID nguồn (Ví dụ '635_642' -> '635')
        source_real_id = fake_video_id.split('_')[0] if '_' in fake_video_id else fake_video_id
        
        # Logic tìm ảnh Real
        if source_real_id in self.real_pool:
            if frame_name in self.real_pool[source_real_id]:
                real_match = self.real_pool[source_real_id][frame_name]
            else:
                real_match = random.choice(list(self.real_pool[source_real_id].values()))
            real_image_path, real_spe_label, real_label = real_match
        else:
            # Nếu code chạy vào đây nghĩa là ID không khớp, ta sẽ in cảnh báo (tùy chọn)
            # print(f"⚠️ Không tìm thấy source {source_real_id} cho fake video {fake_video_id}")
            real_index = random.randint(0, len(self.real_imglist) - 1)
            real_image_path, real_spe_label, real_label = self.real_imglist[real_index]

        # Load ảnh
        fake_image = np.array(self.load_rgb(fake_image_path))
        real_image = np.array(self.load_rgb(real_image_path))

        # Data Augmentation
        fake_trans, _, _ = self.data_aug(fake_image, None, None)
        real_trans, _, _ = self.data_aug(real_image, None, None)

        if not norm:
            return {"fake": (fake_trans, fake_label), "real": (real_trans, real_label)}

        # Normalize to Tensor
        fake_trans = self.normalize(self.to_tensor(fake_trans))
        real_trans = self.normalize(self.to_tensor(real_trans))

        return {
            "fake": (fake_trans, fake_label, fake_spe_label, None, None), 
            "real": (real_trans, real_label, real_spe_label, None, None)
        }

    def __len__(self):
        return len(self.fake_imglist)

    @staticmethod
    def collate_fn(batch):
        f_imgs, f_labels, f_spe_labels, _, _ = zip(*[data["fake"] for data in batch])
        r_imgs, r_labels, r_spe_labels, _, _ = zip(*[data["real"] for data in batch])

        images = torch.cat([torch.stack(r_imgs), torch.stack(f_imgs)], dim=0)
        labels = torch.cat([torch.LongTensor(r_labels), torch.LongTensor(f_labels)], dim=0)
        spe_labels = torch.cat([torch.LongTensor(r_spe_labels), torch.LongTensor(f_spe_labels)], dim=0)

        return {'image': images, 'label': labels, 'label_spe': spe_labels}
