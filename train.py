# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision.models import resnet34
# from dataset import TposeBFPDataset
# from tqdm import tqdm
# import argparse

# # ──────────────────────────────────────────────────────────────────────────
# # ───────────────────BaseLine───────────────────────────────────────────────
# class BFPRegressor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = resnet34(pretrained=True)
#         self.model.fc = nn.Linear(self.model.fc.in_features, 1)

#     def forward(self, x):
#         return self.model(x).squeeze(1)
# # ──────────────────────────────────────────────────────────────────────────


# @torch.no_grad()
# def evaluate_mae(model, dataloader, device):
#     model.eval()
#     criterion = nn.L1Loss(reduction="sum")  # 샘플 합으로 모은 뒤 평균
#     total = 0
#     total_abs_err = 0.0
#     for imgs, labels in tqdm(dataloader, desc="Valid", unit="batch"):
#         imgs = imgs.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
#         preds = model(imgs)
#         total_abs_err += criterion(preds, labels).item()
#         total += imgs.size(0)
#     return total_abs_err / max(total, 1)

# if __name__ == "__main__":
#     # 데이터셋 경로
#     dataset_path = r"D:/bfp/Training/orig_data"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     save_dir   = "female_model"
#     os.makedirs(save_dir, exist_ok=True)
#     best_ckpt_path = os.path.join(save_dir, "best_mae.pth")

#     # ─── 하이퍼파라미터 ───────────────────────────────────────────
#     epochs = 500
#     batch_size = 256
#     lr = 1e-4
#     num_workers = 4

#     # 데이터셋 & 로더
#     print("/n[INFO] Loading train dataset...")
#     dataset = TposeBFPDataset(root_dir=dataset_path, mode='train')
#     loader  = DataLoader(
#         dataset,
#         batch_size=256,           # 배치 사이즈는 GPU 메모리 여유 보고 조절
#         shuffle=True,
#         num_workers=4,           # CPU 코어 수에 맞춰 조절
#         pin_memory=True,         # GPU 직접 전송 속도 개선
#     )

#     print("/n[INFO] Loading test dataset...")
#     test_dataset = TposeBFPDataset(root_dir=dataset_path, mode='test')
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=256,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )

#     # 모델, 손실, 옵티마이저
#     model = BFPRegressor().to(device)
#     criterion = nn.L1Loss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

#     # ─── Best 추적 ────────────────────────────────────────────────
#     best_mae = float("inf")
#     best_epoch = -1
#     best_state = None

#     for epoch in range(1, epochs + 1):
#         model.train()
#         running_abs_err = 0.0
#         total_samples = 0

#         print(f"/n🔁 [Epoch {epoch}] --------------------")
#         for imgs, labels in tqdm(loader, desc=f"Epoch {epoch}", unit="batch"):
#             imgs, labels = imgs.to(device), labels.to(device)
#             preds = model(imgs)
#             loss = criterion(preds, labels)


#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_abs_err += loss.item() * imgs.size(0)
#             total_samples += imgs.size(0)

#         train_mae = running_abs_err / max(total_samples, 1)
#         print(f"✅ [Epoch {epoch}] Train MAE: {train_mae:.4f}")

#         # ─── test ───────────────────────────────────────────
#         val_mae = evaluate_mae(model, test_loader, device)
#         print(f"🧪 [Epoch {epoch}] Valid MAE:  {val_mae:.4f}")

#         # ─── Best 갱신 시에만 저장 ───────────────────────────────
#         if val_mae < best_mae:
#             best_mae = val_mae
#             best_epoch = epoch
#             best_state = {k: v.cpu() for k, v in model.state_dict().items()}  # CPU에 저장
#             torch.save(best_state, best_ckpt_path)
#             print(f"💾 New BEST! MAE={best_mae:.4f} @epoch {best_epoch} -> {best_ckpt_path}")

#     # ─── 종료 후 Best 정보 재로딩(모델을 best 상태로 유지) ───────
#     if best_state is not None:
#         model.load_state_dict(best_state)
#     print("/n=========== BEST RESULT ===========")
#     print(f"Best MAE: {best_mae:.4f} (epoch {best_epoch})")
#     print(f"Saved:    {best_ckpt_path}")
#     print("===================================")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from dataset import TposeBFPDataset
from tqdm import tqdm
import argparse, yaml

class BFPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)

@torch.no_grad()
def evaluate_mae(model, dataloader, device):
    model.eval()
    criterion = nn.L1Loss(reduction="sum")
    total, total_abs_err = 0, 0.0
    for imgs, labels in tqdm(dataloader, desc="Valid", unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        total_abs_err += criterion(preds, labels).item()
        total += imgs.size(0)
    return total_abs_err / max(total, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r",encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg["dataset"]["path"]
    save_dir     = cfg["train"]["save_dir"]
    epochs       = cfg["train"]["epochs"]
    batch_size   = cfg["train"]["batch_size"]
    lr           = cfg["train"]["lr"]
    num_workers  = cfg["train"]["num_workers"]

    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_mae.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[INFO] Loading train dataset...")
    dataset = TposeBFPDataset(root_dir=dataset_path, config_path=args.config, mode='train')
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers, pin_memory=True)

    print("\n[INFO] Loading test dataset...")
    test_dataset = TposeBFPDataset(root_dir=dataset_path, config_path=args.config, mode='test')
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    model = BFPRegressor().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_mae, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(1, epochs + 1):
        model.train()
        running_abs_err, total_samples = 0.0, 0
        print(f"\n🔁 [Epoch {epoch}] --------------------")
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch}", unit="batch"):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_abs_err += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

        train_mae = running_abs_err / max(total_samples, 1)
        print(f"✅ [Epoch {epoch}] Train MAE: {train_mae:.4f}")

        val_mae = evaluate_mae(model, test_loader, device)
        print(f"🧪 [Epoch {epoch}] Valid MAE:  {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, best_ckpt_path)
            print(f"💾 New BEST! MAE={best_mae:.4f} @epoch {best_epoch} -> {best_ckpt_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

        # best 결과 텍스트로 저장
        result_path = os.path.join(save_dir, "best_result.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"Best MAE: {best_mae:.4f}\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Checkpoint: {best_ckpt_path}\n")

    print("\n=========== BEST RESULT ===========")
    print(f"Best MAE: {best_mae:.4f} (epoch {best_epoch})")
    print(f"Saved:    {best_ckpt_path}")
    print("===================================")
