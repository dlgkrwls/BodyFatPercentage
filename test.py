import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TposeBFPDataset
from train import BFPRegressor  # BFPRegressor를 정의한 파일에서 import

def test_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Testing", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(imgs)

            loss = criterion(preds, labels)
            # 디버그 로그가 너무 많다면 아래 두 줄은 주석 처리하세요.
            # print('preds:', preds)
            # print('labels:', labels)

            total_loss += loss.item() * imgs.size(0)

    mae = total_loss / len(dataloader.dataset)  # L1Loss이므로 MAE
    return mae

if __name__ == "__main__":
    # ─── 설정 ───────────────────────────────────────────────────────
    test_dataset_path = r"E:/BFP/BodyMeasurement/data/Validation/orig_data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── 데이터 로더 ─────────────────────────────────────────────────
    test_dataset = TposeBFPDataset(root_dir=test_dataset_path, mode='test')
    test_loader  = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ─── Best 추적 변수 ─────────────────────────────────────────────
    best_mae = float("inf")
    best_epoch = None
    best_ckpt = None

    # 에폭별 평가
    for i in range(200, 501):
        checkpoint_path = f"male_model/bfp_epoch{i}.pth"  # 평가할 체크포인트
        print('checkpoint_path',checkpoint_path)
        # ─── 모델 & 손실 함수 로드 ───────────────────────────────────────
        model = BFPRegressor().to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        criterion = nn.L1Loss()

        # ─── 테스트 실행 ─────────────────────────────────────────────────
        test_mae = test_model(model, test_loader, criterion, device)
        print(f"============= epoch {i} =================")
        print(f"Test MAE: {test_mae:.4f}")
        print("========================================")

        # ─── Best 갱신 ─────────────────────────────────────────────────
        if test_mae < best_mae:
            best_mae = test_mae
            best_epoch = i
            best_ckpt = checkpoint_path

    # ─── 전체 루프 종료 후 한 번만 Best 출력 ─────────────────────────
    print("\n=========== BEST RESULT ===========")
    print(f"Best MAE: {best_mae:.4f} (epoch {best_epoch}, ckpt: {best_ckpt})")
    print("===================================")
