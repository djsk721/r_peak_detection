from models.model import UNet, RPeakModel
from models.trainer import QRSDetectionTrainer, RPeakDetectionTrainer, train_model
from models.loss import FocalLoss
from data.dataset import ECGDataset, create_qrs_segmentation_dataset, custom_collate_fn
from data.dataset import create_regression_dataset
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import argparse
from sklearn.model_selection import train_test_split
import os 
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='ECG QRS Complex 및 R-peak 검출 모델 학습')
    
    # 데이터 관련 인자
    parser.add_argument('--mit_path', type=str, default='../mit-bih-arrhythmia-database',
                      help='MIT-BIH 데이터베이스 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='배치 크기')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='검증 데이터 비율')

    # 학습 관련 인자
    parser.add_argument('--num_epochs', type=int, default=1,
                      help='학습 에폭 수')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                      help='Early stopping patience')
    parser.add_argument('--qrs_save_path', type=str, default='models/saved_models/qrs_model.pth',
                      help='QRS 모델 저장 경로')
    parser.add_argument('--rpeak_save_path', type=str, default='models/saved_models/rpeak_model.pth',
                      help='R-peak 모델 저장 경로')
    parser.add_argument('--device', type=str, default='cuda',
                      help='사용할 장치 (cuda, cpu)')
    parser.add_argument('--qrs_model_path', type=str, default='models/saved_models/qrs_model.pth',
                      help='QRS 모델 저장 경로')
    
    return parser.parse_args()


# 사용 예시
if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    # 데이터셋 생성
    files = os.listdir(args.mit_path)

    # 확장자를 제외한 파일 이름만 추출
    records_to_use = []
    for file in files:
        # 확장자가 있는 경우 제거
        file_name = os.path.splitext(file)[0]
        # 중복 제거 (같은 이름의 다른 확장자 파일이 있을 수 있음)
        if file_name not in records_to_use:
            records_to_use.append(file_name)
    records_to_use.remove('102-0')
    
    X, y, Z = create_qrs_segmentation_dataset(records_to_use, args.mit_path)

    X_processed = np.array([x[0] for x in tqdm(X, desc="ECG 전처리 중")])    
    
    # 훈련 데이터와 나머지 데이터 분할 (80% vs 20%)
    X_train, X_temp, y_train, y_temp, Z_train, Z_temp = train_test_split(X_processed, y, Z, test_size=0.2, random_state=42)
    # 검증 데이터와 테스트 데이터 분할 (각각 10%)
    X_val, X_test, y_val, y_test, Z_val, Z_test = train_test_split(X_temp, y_temp, Z_temp, test_size=0.5, random_state=42)
    
    # U-Net 모델을 위한 데이터셋 및 데이터 로더 생성
    qrs_train_dataset = ECGDataset(X_train, y_train, Z_train)
    qrs_val_dataset = ECGDataset(X_val, y_val, Z_val)
    qrs_test_dataset = ECGDataset(X_test, y_test, Z_test)
    
    qrs_train_loader = DataLoader(qrs_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True, drop_last=True)
    qrs_val_loader = DataLoader(qrs_val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True)
    qrs_test_loader = DataLoader(qrs_test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True)
    # QRS Complex 검출 모델 학습
    qrs_model = UNet(in_channels=1, num_classes=1).to(device)
    qrs_optimizer = torch.optim.Adam(qrs_model.parameters())
    qrs_criterion = FocalLoss()
    qrs_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        qrs_optimizer, mode='min', factor=0.8, patience=2
    )
    # qrs_trainer = QRSDetectionTrainer(
    #     model=qrs_model,
    #     optimizer=qrs_optimizer,
    #     criterion=qrs_criterion,
    #     scheduler=qrs_scheduler,
    #     device=device
    # )
    
    # train_model(
    #     trainer=qrs_trainer,
    #     train_loader=qrs_train_loader,
    #     val_loader=qrs_val_loader,
    #     test_loader=qrs_test_loader,
    #     num_epochs=args.num_epochs,
    #     save_path=args.qrs_save_path
    # )
    
    X_reg_train, y_reg_train = create_regression_dataset(qrs_train_loader, device)
    X_reg_val, y_reg_val = create_regression_dataset(qrs_val_loader, device)
    X_reg_test, y_reg_test = create_regression_dataset(qrs_test_loader, device)
    
    # 회귀 모델 데이터 로더 생성
    rpeak_train_dataset = TensorDataset(X_reg_train, y_reg_train)
    rpeak_val_dataset = TensorDataset(X_reg_val, y_reg_val)
    rpeak_test_dataset = TensorDataset(X_reg_test, y_reg_test)


    r_peak_train_loader = DataLoader(rpeak_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    r_peak_val_loader = DataLoader(rpeak_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    r_peak_test_loader = DataLoader(rpeak_test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    
    # R-peak 검출 모델 학습
    rpeak_model = RPeakModel()
    rpeak_optimizer = torch.optim.Adam(rpeak_model.parameters())
    rpeak_criterion = nn.MSELoss()
    rpeak_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(rpeak_optimizer, mode='min', factor=0.8, patience=2)
    rpeak_trainer = RPeakDetectionTrainer(
        model=rpeak_model,
        optimizer=rpeak_optimizer,
        criterion=rpeak_criterion,
        scheduler=rpeak_scheduler,
        device=device
    )
    
    train_model(
        trainer=rpeak_trainer,
        train_loader=r_peak_train_loader,
        val_loader=r_peak_val_loader,
        test_loader=r_peak_test_loader,
        num_epochs=args.num_epochs,
        save_path=args.rpeak_save_path
    )