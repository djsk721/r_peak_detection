import numpy as np 
import scipy
import torch


# ============================================================================
# 신호 정규화 함수
# ============================================================================

def normalize_to_range(signal, min_val, max_val):
    """
    신호를 지정된 범위로 정규화합니다.
    """
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    
    # 신호의 범위가 0인 경우(모든 값이 동일한 경우) 처리
    if signal_max == signal_min:
        return np.zeros_like(signal)
    
    # 신호를 0-1 범위로 정규화한 후 원하는 범위로 스케일링
    normalized = (signal - signal_min) / (signal_max - signal_min)
    scaled = normalized * (max_val - min_val) + min_val
    
    return scaled


# ============================================================================
# ECG 신호 전처리 함수
# ============================================================================

def preprocess_ecg(ecg_signal, sampling_rate, lowcut=0.5, highcut=40, min_val=-1, max_val=1, power_line_freq=60):
    if len(ecg_signal) == 0:
        return np.array([])
    
    if np.all(np.isnan(ecg_signal)):
        return np.zeros_like(ecg_signal)
    
    pad_length = int(sampling_rate)
    padded_signal = np.pad(ecg_signal, (pad_length, pad_length), 'symmetric')
    
    baseline = scipy.signal.savgol_filter(padded_signal, int(sampling_rate/2)*2+1, 3)
    baseline_removed = padded_signal - baseline
    
    # 버터워스 필터 적용
    nyquist = 0.5 * sampling_rate
    order = 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered_signal = scipy.signal.filtfilt(b, a, baseline_removed)
    
    # 노치 필터는 선택적으로 적용
    if power_line_freq > 0:
        w0 = power_line_freq / nyquist
        Q = 30.0
        b_notch, a_notch = scipy.signal.iirnotch(w0, Q, sampling_rate)
        filtered_signal = scipy.signal.filtfilt(b_notch, a_notch, filtered_signal)
    
    # 패딩 제거
    filtered_signal = filtered_signal[pad_length:-pad_length]
    
    # 정규화
    min_signal = np.min(filtered_signal)
    max_signal = np.max(filtered_signal)
    if max_signal > min_signal:
        processed_signal = (filtered_signal - min_signal) / (max_signal - min_signal)
        processed_signal = processed_signal * (max_val - min_val) + min_val
    else:
        processed_signal = np.zeros_like(filtered_signal)
    
    return processed_signal

def extract_qrs_complexes(ecg_signal, qrs_mask, fixed_length=32, r_peaks=None):
    """
    ECG 신호에서 QRS 컴플렉스를 추출하고 고정 길이로 리샘플링
    주어진 R-peak 위치도 함께 리샘플링
    
    Args:
        ecg_signal: ECG 신호
        qrs_mask: QRS 컴플렉스 마스크
        fixed_length: 리샘플링할 고정 길이
        r_peaks: 원본 신호에서의 R-peak 위치 리스트 (선택적)
    """
    # 텐서를 numpy 배열로 변환
    if isinstance(ecg_signal, torch.Tensor):
        ecg_signal = ecg_signal.cpu().numpy()
    if isinstance(qrs_mask, torch.Tensor):
        qrs_mask = qrs_mask.cpu().numpy()
    
    # 차원 확인 및 조정
    if len(ecg_signal.shape) > 1:
        ecg_signal = ecg_signal.squeeze()
    if len(qrs_mask.shape) > 1:
        qrs_mask = qrs_mask.squeeze()
    
    # QRS 컴플렉스 구간 찾기
    qrs_regions = []
    in_qrs = False
    start_idx = 0
    
    for i, val in enumerate(qrs_mask):
        if val > 0.5 and not in_qrs:  # QRS 시작
            in_qrs = True
            start_idx = i
        elif val <= 0.5 and in_qrs:  # QRS 종료
            in_qrs = False
            qrs_regions.append((start_idx, i))
    # 마지막 QRS가 신호 끝까지 계속되는 경우
    if in_qrs:
        qrs_regions.append((start_idx, len(qrs_mask)))
    # QRS 컴플렉스 추출 및 리샘플링
    resampled_qrs = []
    resampled_r_peaks = []
    
    for i, (start, end) in enumerate(qrs_regions):
        if end - start < 5:  # 너무 짧은 QRS는 무시
            continue
        qrs_complex = ecg_signal[start:end]
        
        # 리샘플링 (고정 길이로)
        resampled = scipy.signal.resample(qrs_complex, fixed_length)
        resampled_qrs.append(resampled)
        
        # R-peak 위치 리샘플링
        if r_peaks is not None and i < len(r_peaks):
            # 주어진 R-peak이 현재 QRS 영역 내에 있는지 확인
            r_peak = r_peaks[i]
            if start <= r_peak < end:
                # 원본 영역 내에서의 상대적 위치 계산
                relative_pos = (r_peak - start) / (end - start)
                # 리샘플링된 영역에서의 위치 계산
                resampled_r_peak = int(relative_pos * fixed_length)
                resampled_r_peaks.append(resampled_r_peak)
        else:
            # R-peak이 주어지지 않은 경우 신호의 최대값 위치로 가정
            original_r_peak = np.argmax(qrs_complex)
            resampled_r_peak = int(original_r_peak * (fixed_length / len(qrs_complex)))
            resampled_r_peaks.append(resampled_r_peak)
    
    return np.array(resampled_qrs), qrs_regions, np.array(resampled_r_peaks)