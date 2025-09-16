# import os
# import glob
# import pandas as pd
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
# import yaml

# class TposeBFPDataset(Dataset):
#     def __init__(self, root_dir, config_path,mode='train'):
#         with open(config_path,'r')as f:
#             cfg=yaml.safe_load(f)
#         prefixes=cfg['dataset']['prefixes']
#         image_pattern=cfg['dataset']['image_pattern']
#         allowed_suffixes=set(cfg['dataset']['allowed_suffixes'])
#         self.samples = []
#         imagenet_mean = [0.485, 0.456, 0.406]
#         imagenet_std = [0.229, 0.224, 0.225]

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(imagenet_mean, imagenet_std),
#         ])


#         # root_dir 하위의 모든 서브폴더를 순회
#         for ds in os.listdir(root_dir):
#             if not any(ds.startswith(p) for p in prefixes):
#                 continue

#             if "to" not in ds:
#                 continue
#             tail = ds.split("to", 1)[-1]
#             last_num = tail[-3:] if len(tail) >= 3 else tail
#             is_test_ds = (last_num == "505")

#             if (mode == "test" and not is_test_ds) or (mode == "train" and is_test_ds):
#                 continue

#             ds_path = os.path.join(root_dir, ds)
#             print('ds_path',ds_path)
#             if not os.path.isdir(ds_path):
#                 continue

#             # 각 dataset_dir 내의 모든 피험자 디렉터리 탐색
#             for person_id in os.listdir(ds_path):
#                 person_path = os.path.join(ds_path, person_id)
#                 if not os.path.isdir(person_path):
#                     continue

#                 csv_path = os.path.join(person_path, "csv", f"{person_id}.csv")
#                 img_dir  = os.path.join(person_path, "Image")

#                 if not os.path.exists(csv_path) or not os.path.isdir(img_dir):
#                     continue


#                 # CSV에서 체지방율 읽기
#                 try:
#                     df  = pd.read_csv(csv_path, encoding="utf-8-sig", header=1)
#                     bfp = float(df["체지방율"].values[0])
#                 except Exception as e:
#                     print(f"⚠️ CSV 오류 ({ds}/{person_id}): {e}")
#                     continue

#                 # "03_*.jpg" 이미지 중에서 허용된 suf _*_*.jpg") # "03_*_*.jpg" ->  전체 옷 "03_01_*.jpg"-> naked
#                 pattern = os.path.join(img_dir, imga)
#                 for img_path in glob.glob(pattern):
#                     fname = os.path.splitext(os.path.basename(img_path))[0]
#                     suffix = fname[-2:]
#                     if suffix in allowed_suffixes:
#                         self.samples.append((img_path, bfp))
#                     # else: segmentation 이미지나 다른 파일은 무시

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, bfp = self.samples[idx]
#         img = Image.open(img_path).convert("RGB")
#         img = self.transform(img)
#         return img, torch.tensor(bfp, dtype=torch.float32)
import os
import glob
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import yaml

class TposeBFPDataset(Dataset):
    def __init__(self, root_dir, config_path, mode='train'):
        with open(config_path, "r",encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        prefixes = cfg["dataset"]["prefixes"]
        img_pattern = cfg["dataset"]["image_pattern"]
        allowed_suffixes = set(cfg["dataset"]["allowed_suffixes"])

        self.samples = []
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

        for ds in os.listdir(root_dir):
            if not any(ds.startswith(p) for p in prefixes):
                continue
            if "to" not in ds:
                continue
            tail = ds.split("to", 1)[-1]
            last_num = tail[-3:] if len(tail) >= 3 else tail
            is_test_ds = (last_num == "505")

            if (mode == "test" and not is_test_ds) or (mode == "train" and is_test_ds):
                continue

            ds_path = os.path.join(root_dir, ds)
            if not os.path.isdir(ds_path):
                continue

            for person_id in os.listdir(ds_path):
                person_path = os.path.join(ds_path, person_id)
                if not os.path.isdir(person_path):
                    continue

                csv_path = os.path.join(person_path, "csv", f"{person_id}.csv")
                img_dir  = os.path.join(person_path, "Image")

                if not os.path.exists(csv_path) or not os.path.isdir(img_dir):
                    continue

                try:
                    df  = pd.read_csv(csv_path, encoding="utf-8-sig", header=1)
                    bfp = float(df["체지방율"].values[0])
                except Exception as e:
                    print(f"⚠️ CSV 오류 ({ds}/{person_id}): {e}")
                    continue

                pattern = os.path.join(img_dir, img_pattern)
                for img_path in glob.glob(pattern):
                    fname = os.path.splitext(os.path.basename(img_path))[0]
                    suffix = fname[-2:]
                    if suffix in allowed_suffixes:
                        self.samples.append((img_path, bfp))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bfp = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(bfp, dtype=torch.float32)
