from torch.utils.data import Dataset
import torch
import numpy as np
import wfdb
import scipy.signal as signal
from ecg_processing.preprocess import preprocess_ecg, extract_qrs_complexes

# U-Net을 위한 데이터셋 (ECG 신호 → QRS 마스크)
class ECGDataset(Dataset):
    def __init__(self, ecg_signals, qrs_masks, rpeaks, sample_length=500):
        self.ecg_signals = ecg_signals
        self.qrs_masks = qrs_masks
        self.rpeaks = rpeaks
        self.sample_length = sample_length
    def __len__(self):
        return len(self.ecg_signals)
    
    def __getitem__(self, idx):
        ecg_signal = self.ecg_signals[idx].astype(np.float32)
        qrs_mask = self.qrs_masks[idx].astype(np.float32)
        rpeaks = self.rpeaks[idx]
        ecg_signal = preprocess_ecg(ecg_signal, 500)
        
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)
        qrs_mask = torch.tensor(qrs_mask, dtype=torch.float32)
        
        # 차원 조정 (1, 5000 형태로)
        ecg_signal = ecg_signal.view(1, 5000)  # 1, 5000 형태로 변경
        qrs_mask = qrs_mask.view(1, 5000)  # 1, 5000 형태로 변경
        
        return ecg_signal, qrs_mask, rpeaks
    
def custom_collate_fn(batch):
    """
    통합 데이터셋 collate 정의
    """
    ecgs = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    # rpeaks는 배치로 묶지 않고 리스트 그대로 유지
    rpeaks = [item[2] for item in batch]
    
    # 텐서로 변환하여 배치 생성
    ecgs = torch.stack(ecgs)
    masks = torch.stack(masks)
    
    return ecgs, masks, rpeaks[0]


# 데이터 로드 함수
def load_mitbih_record(mitdb_path, record_num):
    """
    MIT-BIH 데이터베이스에서 특정 레코드를 로드
    
    Args:
        record_num (int): 레코드 번호 (예: 100, 101 등)
    
    Returns:
        signals, fields: 신호 데이터와 메타데이터
    """
    record_name = f"{mitdb_path}/{record_num}"
    signals, fields = wfdb.rdsamp(record_name)
    
    # 500Hz로 리샘플링
    fs_original = fields['fs']
    fs_target = 500
    
    if fs_original != fs_target:
        # 리샘플링 비율 계산
        resampling_factor = fs_target / fs_original
        
        new_length = int(len(signals) * resampling_factor)
        
        resampled_signals = np.zeros((new_length, signals.shape[1]))
        for i in range(signals.shape[1]):
            resampled_signals[:, i] = signal.resample(signals[:, i], new_length)
        
        fields['fs'] = fs_target
        
        return resampled_signals, fields
    
    return signals, fields

# 어노테이션 로드 함수
def load_annotations(mitdb_path, record_num):
    """
    MIT-BIH 데이터베이스에서 특정 레코드의 어노테이션을 로드
    
    Args:
        record_num (int): 레코드 번호
    
    Returns:
        ann: 어노테이션 객체
    """
    record_name = f"{mitdb_path}/{record_num}"
    ann = wfdb.rdann(record_name, 'atr')
    
    # 어노테이션 위치를 500Hz에 맞게 조정
    fs_original = 360  # MIT-BIH의 원래 샘플링 주파수
    fs_target = 500
    
    if fs_original != fs_target:
        # 어노테이션 위치 조정
        resampling_factor = fs_target / fs_original
        ann.sample = np.array([int(s * resampling_factor) for s in ann.sample])
    
    return ann 

# QRS 복합체 검출을 위한 데이터셋 생성 함수
def create_qrs_segmentation_dataset(records, mit_path, window_size=5000, qrs_width=100):
    """
    QRS 복합체 검출을 위한 세그멘테이션 데이터셋 생성
    
    Args:
        records: 사용할 ECG 레코드 목록
        window_size: 윈도우 크기 (기본값 5000)
        qrs_width: QRS 복합체의 평균 너비 (픽셀 단위)
    
    Returns:
        X: ECG 신호 윈도우 배열 (shape: [B, 1, 5000]) - B는 배치 크기
        y: 세그멘테이션 마스크 (shape: [B, 5000]) - 1: QRS 복합체, 0: 배경
    """
    X = []
    y = []
    Z = []
    
    for record in records:
        ecg_signal, fields = load_mitbih_record(mit_path, record)
        ann = load_annotations(mit_path, record)
        
        # R-peak 위치 가져오기
        rpeak_locations = []
        for i, symbol in enumerate(ann.symbol):
            if symbol in ['N', 'L', 'R', 'V', 'A']:  # 정상 및 비정상 QRS 복합체 포함
                rpeak_locations.append(ann.sample[i])
        
        # 윈도우 생성
        for i in range(0, len(ecg_signal) - window_size, window_size // 2):  # 50% 오버랩
            window = ecg_signal[i:i+window_size, 0]  # 첫 번째 채널만 사용 Lead I
            
            # 세그멘테이션 마스크 생성
            mask = np.zeros(window_size)
            
            # 윈도우 내 R-peak 위치 저장
            window_rpeaks = []
            
            # 윈도우 내의 모든 R-peak에 대해 QRS 복합체 영역 마킹
            for rpeak in rpeak_locations:
                if i <= rpeak < i + window_size:
                    # 윈도우 내 상대적 R-peak 위치 저장
                    window_rpeaks.append(rpeak - i)
                    
                    # QRS 복합체는 R-peak를 중심으로 qrs_width/2 픽셀 양쪽으로 확장
                    qrs_start = max(0, rpeak - i - qrs_width // 2)
                    qrs_end = min(window_size, rpeak - i + qrs_width // 2)
                    mask[qrs_start:qrs_end] = 1
            
            # R-peak가 없는 윈도우는 건너뛰기
            if np.sum(mask) == 0:
                continue
                
            # 채널 차원 추가 (B, 1, 5000) 형태로 만들기
            X.append(window.reshape(1, -1))
            y.append(mask)
            Z.append(window_rpeaks)
    # X는 (B, 1, 5000) 형태, y는 (B, 5000) 형태로 변환
    X = np.array(X)
    y = np.array(y)
    
    return X, y, Z

def create_regression_dataset(data_loader, device):
    X_reg = []
    y_reg = []
    
    with torch.no_grad():
        for inputs, labels, r_peaks_idx in data_loader:
            inputs = inputs.cpu().numpy()
            labels = labels.cpu().numpy()
            # 배치의 각 샘플에 대해 처리
            for i in range(inputs.shape[0]):
                ecg = inputs[i]
                pred_mask = labels[i][0]  # 첫 번째 채널 선택
                # QRS 복합체 추출 및 리샘플링
                qrs_complexes, qrs_regions, _ = extract_qrs_complexes(ecg, pred_mask, fixed_length=32, r_peaks=r_peaks_idx)
                
                if len(qrs_complexes) == 0:
                    continue
                
                # 각 QRS 복합체에 대해 R 피크 위치 찾기
                for j, (qrs, region) in enumerate(zip(qrs_complexes, qrs_regions)):
                    start, end = region
                    qrs_original = ecg[0, start:end]  # 원본 QRS 복합체
                    
                    if len(qrs_original) < 5:  # 너무 짧은 QRS는 무시
                        continue
                    
                    r_peak_idx = np.argmax(qrs_original)
                    # R 피크 위치를 0~1 사이로 정규화
                    normalized_position = r_peak_idx / (len(qrs_original) - 1) if len(qrs_original) > 1 else 0.5
                    
                    # 데이터셋에 추가
                    X_reg.append(qrs.reshape(1, -1))  # 채널 차원 추가 (1, 32)
                    y_reg.append([normalized_position])
    
    # 텐서로 변환
    X_reg = torch.FloatTensor(np.array(X_reg))
    y_reg = torch.FloatTensor(np.array(y_reg))
    
    return X_reg, y_reg
