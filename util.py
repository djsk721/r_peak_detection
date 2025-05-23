import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import neurokit2 as nk
import matplotlib.pyplot as plt
import biosppy
import scipy
import scipy.signal
import scipy.ndimage
from pantom import Pan_Tompkins_Plus_Plus
from pptx import Presentation
from pptx.util import Inches
import glob
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef



def fix_seed(SEED=42):
    os.environ['SEED'] = str(SEED)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)


def plot_ecg_with_multiple_algorithms(results, title, lead='I', plot_path=None, signal_quality=None):
    """
    여러 알고리즘의 R-peak 검출 결과를 하나의 그래프로 비교하는 함수
    
    Parameters:
    -----------
    results : dict
        extract_ecg_and_rpeaks 함수에서 반환된 결과 딕셔너리
    title : str
        그래프 제목
    lead : str
        ECG 리드 이름
    plot_path : str, optional
        그래프를 저장할 경로
    signal_quality : float, optional
        신호 품질 점수
    """
    # 필요한 데이터 추출
    ensemble_rpeaks = results.get('ensemble_rpeaks', [])
    algorithm_results = results.get('algorithm_results', {})
    emsemble_plot_signal = results.get('plot_signal', [])
    
    # 서브플롯 설정
    fig, axes = plt.subplots(len(algorithm_results) + 1, 1, figsize=(20, 20))
    fig.suptitle(f"{title} \n(Signal Quality: {signal_quality:.2f})", fontsize=16)
    
    # 각 알고리즘별 결과 플롯
    for i, (name, result) in enumerate(algorithm_results.items()):
        ax = axes[i]
        plot_signal = result.get('plot_signal', [])
        ax.plot(plot_signal, label="Signal")

        # R-peaks 표시
        r_peak = result.get('original_rpeaks', [])
        corrected_rpeaks = result.get('corrected_rpeaks', [])
        
        if len(r_peak) > 0:
            ax.scatter(r_peak, plot_signal[r_peak], 
                      color='blue', marker='o', s=50, label='R-peaks')        
        if len(corrected_rpeaks) > 0:
            ax.scatter(corrected_rpeaks, plot_signal[corrected_rpeaks], 
                      color='red', marker='o', s=50, label='Corr. R-peaks')
                
        ax.set_title(f'{name}')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    # 앙상블 결과 플롯
    ax_ensemble = axes[-1]
    ax_ensemble.plot(emsemble_plot_signal, label="Signal")
    
    if len(ensemble_rpeaks) > 0:
        ax_ensemble.scatter(ensemble_rpeaks, emsemble_plot_signal[ensemble_rpeaks], 
                           color='purple', marker='*', s=50, label='Ensemble')
            
    ax_ensemble.set_title('Ensemble Method (R-peaks Detection)')
    ax_ensemble.set_ylabel('Amplitude')
    ax_ensemble.set_xlabel('Samples')
    ax_ensemble.grid(True, alpha=0.3)
    ax_ensemble.legend(loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 제목 공간 확보
    
    if plot_path:
        plt.savefig(plot_path)
    else:
        plt.show()
    plt.close()
    

def plot_ecg_with_peaks(signal, signals_info, title, show_r_peaks=True, show_other_peaks=False, show_wave_markers=False, path=None):
    """
    ECG 신호와 피크를 시각화하는 함수
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG 신호 데이터
    signals_info : dict
        ECG 신호의 피크 정보를 담은 딕셔너리
    patient_index : int
        환자 인덱스
    show_r_peaks : bool
        R-peaks 표시 여부
    show_other_peaks : bool
        R-peaks 외 다른 피크 표시 여부
    show_wave_markers : bool
        파형 마커(onsets, offsets) 표시 여부
    path : str, optional
        그래프를 저장할 경로
    """
    plt.figure(figsize=(20, 10))
    
    # 정제된 ECG 신호 플롯
    plt.plot(signal, label="Raw ECG")
    # 피크 타입 정의
    peak_types = {
        'ECG_R_Peaks': {'color': 'red', 'marker': 'o', 'label': 'R-peaks'},
    }
    
    # 다른 피크 타입 추가 (show_other_peaks가 True일 때만)
    if show_other_peaks:
        peak_types.update({
            'ECG_T_Peaks': {'color': 'purple', 'marker': 'o', 'label': 'T-peaks'},
            'ECG_P_Peaks': {'color': 'orange', 'marker': 'o', 'label': 'P-peaks'},
            'ECG_Q_Peaks': {'color': 'green', 'marker': 'o', 'label': 'Q-peaks'},
            'ECG_S_Peaks': {'color': 'blue', 'marker': 'o', 'label': 'S-peaks'}
        })
    
    # R-peaks만 표시하거나 show_r_peaks가 True일 때
    if show_r_peaks:
        # 모든 피크 타입 플롯 - 정제된 신호에 맞춰 표시
        for peak_name, peak_info in peak_types.items():
            if peak_name in signals_info.keys():
                peak_indices = signals_info[peak_name]
                if len(peak_indices) > 0:
                    plt.scatter(
                        peak_indices, 
                        signal[peak_indices],
                        color=peak_info['color'], 
                        marker=peak_info['marker'], 
                        label=peak_info['label']
                    )

    # 시작점과 종료점 마커 정의 (show_wave_markers가 True일 때만)
    if show_wave_markers:
        print("signals_info : ", signals_info)
        print(signals_info['ECG_P_Onsets'].shape)
        
        wave_markers = {
            # Onsets
            'ECG_P_Onsets': {'color': 'blue', 'linestyle': '--', 'label': 'P-onsets'},
            'ECG_T_Onsets': {'color': 'cyan', 'linestyle': '--', 'label': 'T-onsets'},
            'ECG_QRS_Onsets': {'color': 'magenta', 'linestyle': '--', 'label': 'QRS-onsets'},
            # Offsets
            'ECG_P_Offsets': {'color': 'blue', 'linestyle': ':', 'label': 'P-offsets'},
            'ECG_T_Offsets': {'color': 'cyan', 'linestyle': ':', 'label': 'T-offsets'},
            'ECG_QRS_Offsets': {'color': 'magenta', 'linestyle': ':', 'label': 'QRS-offsets'}
        }

        for marker_name, marker_info in wave_markers.items():
            if marker_name in signals_info.keys():
                indices = np.where(signals_info[marker_name] == 1)[0]
                if len(indices) > 0:
                    plt.axvline(x=indices[0], color=marker_info['color'], 
                                linestyle=marker_info['linestyle'], alpha=0.7, 
                                label=marker_info['label'])
                    
                    # 나머지는 범례 없이 표시
                    for idx in indices[1:]:
                        plt.axvline(x=idx, color=marker_info['color'], 
                                    linestyle=marker_info['linestyle'], alpha=0.7)
    plt.title(f'ECG Signal Analysis with Detected Peaks')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # 범례 위치 조정
    plt.tight_layout()  # 레이아웃 자동 조정
    if path:
        plt.savefig(path)
    else:
        plt.show()
        
    plt.close()
    


def visualize_hrv_features(ecg_signal, peaks, hrv_features, ecg_features=None, title=None, path=None):
    """
    ECG 신호와 HRV 특성을 종합적으로 시각화하는 함수
    
    Args:
        ecg_signal: ECG 신호 데이터
        peaks: 검출된 피크 정보를 담은 딕셔너리
        hrv_features: HRV 특성 정보를 담은 딕셔너리
        ecg_features: ECG 특성 정보를 담은 딕셔너리 (선택적)
        sampling_rate: 샘플링 레이트 (Hz)
        title: 그래프 제목 (선택적)
        path: 그래프를 저장할 경로 (선택적)
    """
    # 서브플롯 설정 (ECG 특성 유무에 따라 행 수 조정)
    n_rows = 4 if ecg_features else 3
    fig = plt.figure(figsize=(20, n_rows * 5))
    
    # 1. ECG 신호와 피크 표시
    ax1 = plt.subplot(n_rows, 1, 1)
    ax1.plot(ecg_signal, label="ECG Signal")
    
    # 피크 타입 정의
    peak_types = {
        'ECG_R_Peaks': {'color': 'red', 'marker': 'o', 'label': 'R-peaks'},
        'ECG_P_Peaks': {'color': 'orange', 'marker': 'o', 'label': 'P-peaks'},
        'ECG_Q_Peaks': {'color': 'green', 'marker': 'o', 'label': 'Q-peaks'},
        'ECG_S_Peaks': {'color': 'blue', 'marker': 'o', 'label': 'S-peaks'},
        'ECG_T_Peaks': {'color': 'purple', 'marker': 'o', 'label': 'T-peaks'}
    }
    
    # 모든 피크 타입 플롯
    for peak_name, peak_info in peak_types.items():
        if peak_name in peaks.keys() and len(peaks[peak_name]) > 0:
            ax1.scatter(
                peaks[peak_name], 
                ecg_signal[peaks[peak_name]],
                color=peak_info['color'], 
                marker=peak_info['marker'], 
                label=peak_info['label']
            )
    
    ax1.set_title("ECG Signal with Detected Peaks")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # 2. RR 간격 시계열 플롯
    ax2 = plt.subplot(n_rows, 1, 2)
    rr_intervals = hrv_features['rr_intervals']
    ax2.plot(rr_intervals, 'o-', color='blue')
    ax2.axhline(y=np.mean(rr_intervals), color='r', linestyle='--', label=f'Mean: {np.mean(rr_intervals):.2f} ms')
    ax2.fill_between(range(len(rr_intervals)), 
                    np.mean(rr_intervals) - np.std(rr_intervals), 
                    np.mean(rr_intervals) + np.std(rr_intervals), 
                    color='red', alpha=0.2, label=f'SD: {np.std(rr_intervals):.2f} ms')
    ax2.set_title("RR Intervals Time Series")
    ax2.set_xlabel("Beat Number")
    ax2.set_ylabel("RR Interval (ms)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # 3. Poincaré 플롯 (비선형 HRV 분석)
    ax3 = plt.subplot(n_rows, 1, 3)
    if len(rr_intervals) > 1:
        rr_n = np.array(rr_intervals[:-1])
        rr_n1 = np.array(rr_intervals[1:])
        
        # 산점도 그리기
        ax3.scatter(rr_n, rr_n1, color='blue', alpha=0.7)
        
        # SD1, SD2 타원 그리기
        sd1 = hrv_features['nonlinear']['sd1']
        sd2 = hrv_features['nonlinear']['sd2']
        
        if sd1 is not None and sd2 is not None:
            # 타원 중심점 (평균 RR, 평균 RR)
            center = (np.mean(rr_n), np.mean(rr_n1))
            
            # 타원 그리기
            from matplotlib.patches import Ellipse
            ellipse = Ellipse(xy=center, width=2*sd2, height=2*sd1, 
                             angle=45, edgecolor='red', fc='none', lw=2)
            ax3.add_patch(ellipse)
            
            # SD1, SD2 선 그리기
            ax3.plot([center[0], center[0]], [center[1], center[1] + sd1], 'r-', label=f'SD1: {sd1:.2f} ms')
            ax3.plot([center[0], center[0] + sd2], [center[1], center[1]], 'g-', label=f'SD2: {sd2:.2f} ms')
            
            # 정체선 그리기 (y=x)
            min_rr = min(min(rr_n), min(rr_n1))
            max_rr = max(max(rr_n), max(rr_n1))
            ax3.plot([min_rr, max_rr], [min_rr, max_rr], 'k--', alpha=0.5)
            
            # 비율 표시
            ratio = hrv_features['nonlinear']['sd1_sd2_ratio']
            ax3.text(0.05, 0.05, f'SD1/SD2: {ratio:.3f}', transform=ax3.transAxes, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    ax3.set_title("Poincaré Plot")
    ax3.set_xlabel("RR(n) (ms)")
    ax3.set_ylabel("RR(n+1) (ms)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # 4. ECG 특성 요약 (선택적)
    if ecg_features:
        ax4 = plt.subplot(n_rows, 1, 4)
        
        # 특성 이름과 평균값 추출
        feature_names = []
        feature_values = []
        
        for feature_name, feature_data in ecg_features.items():
            if 'stats' in feature_data and 'mean' in feature_data['stats']:
                feature_names.append(feature_name.replace('_', ' ').title())
                feature_values.append(feature_data['stats']['mean'])
        
        # 바 차트로 표시
        bars = ax4.bar(feature_names, feature_values, color=['blue', 'green', 'orange', 'purple', 'red'])
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f} ms', ha='center', va='bottom')
        
        ax4.set_title("ECG Features Summary (Mean Values)")
        ax4.set_ylabel("Duration (ms)")
        ax4.grid(True, alpha=0.3, axis='y')
        
    # HRV 시간 영역 특성 표시
    time_domain = hrv_features['time_domain']
    info_text = (
        f"Mean HR: {60000 / np.mean(hrv_features['rr_intervals']):.2f} bpm\n"
        f"Time Domain HRV Metrics:\n"
        f"Mean RR: {time_domain['mean_rr']:.2f} ms\n"
        f"SDNN: {time_domain['sdnn']:.2f} ms\n"
        f"RMSSD: {time_domain['rmssd']:.2f} ms\n"
        f"pNN20: {time_domain['pnn20']:.2f}%\n"
        f"pNN50: {time_domain['pnn50']:.2f}%\n"
        f"HR Max-Min: {time_domain['hr_max_min']:.2f} bpm"
    )
    
    # 정보 텍스트 추가
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 전체 제목 설정
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92 if title else 0.95)  # 제목 공간 확보
    
    # 저장 또는 표시
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
    
def save_images_to_pptx(image_folder, save_path, output_pptx):
    """
    이미지 폴더의 모든 이미지를 PowerPoint 문서로 저장합니다.
    각 슬라이드에 하나의 이미지만 표시됩니다.
    
    Args:
        image_folder (str): 이미지가 저장된 폴더 경로
        output_pptx (str): 저장할 PowerPoint 파일 경로
    """
    # 새 프레젠테이션 생성
    prs = Presentation()
    
    # 이미지 파일 목록 가져오기 (png 파일만)
    image_files = glob.glob(os.path.join(image_folder, "*.png"))
    
    # 파일명에서 인덱스 추출하여 정렬
    def get_index(file_path):
        # 파일명에서 마지막 언더스코어 이후의 숫자를 인덱스로 사용
        filename = os.path.basename(file_path)
        index_str = filename.split('_')[-2].split('.')[0]
        return int(index_str)
    
    # 인덱스 기준으로 이미지 파일 정렬
    image_files.sort(key=get_index)
    
    # 각 이미지를 슬라이드에 추가
    for image_path in image_files:
        # 빈 슬라이드 추가
        slide_layout = prs.slide_layouts[6]  # 빈 레이아웃
        slide = prs.slides.add_slide(slide_layout)
        
        # 슬라이드 크기 가져오기
        slide_width = prs.slide_width
        slide_height = prs.slide_height
        
        # 여백 설정
        margin = Inches(0.5)
        
        # 이미지를 슬라이드 크기에 맞게 조정
        left = margin
        top = margin
        width = slide_width - (2 * margin)
        height = slide_height - (2 * margin) - Inches(0.5)  # 텍스트 상자 공간 확보
        
        
        # 이미지 추가 (슬라이드 크기에 맞게 자동 조정)
        slide.shapes.add_picture(image_path, left, top, width=width, height=height)
    
    # PowerPoint 파일 저장
    prs.save(os.path.join(save_path, output_pptx))
    print(f"PowerPoint 문서가 {output_pptx}에 저장되었습니다.")
    
def assess_signal_quality(signal, sampling_rate):
    """
    ECG 신호의 품질을 평가하는 함수
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG 신호 데이터
    sampling_rate : float
        샘플링 레이트
        
    Returns:
    --------
    quality_score : float
        신호 품질 점수 (0~1 사이 값, 1이 최상)
    """
    # 신호가 없거나 모두 0인 경우
    if len(signal) == 0 or np.all(signal == 0):
        return 0.0
    
    # 1. 신호 대 잡음비(SNR) 추정
    # 신호의 표준편차와 중앙값 절대편차(MAD)를 사용하여 SNR 추정
    signal_std = np.std(signal)
    signal_mad = np.median(np.abs(signal - np.median(signal)))
    estimated_snr = signal_std / (signal_mad + 1e-10)  # 0 나누기 방지
    snr_score = min(1.0, estimated_snr / 5.0)  # SNR 5 이상이면 최대 점수
    
    # 2. 기저선 변동 평가
    # 저주파 성분 추출을 위한 필터링
    nyquist = 0.5 * sampling_rate
    low_freq = 0.5 / nyquist
    b, a = scipy.signal.butter(2, low_freq, 'lowpass')
    baseline = scipy.signal.filtfilt(b, a, signal)
    baseline_variation = np.std(baseline) / (np.std(signal) + 1e-10)
    baseline_score = 1.0 - min(1.0, baseline_variation)
    
    # 3. 이상치 비율 평가
    # 평균에서 3 표준편차 이상 벗어난 샘플 비율
    mean = np.mean(signal)
    std = np.std(signal)
    outliers = np.abs(signal - mean) > 3 * std
    outlier_ratio = np.sum(outliers) / len(signal)
    outlier_score = 1.0 - min(1.0, outlier_ratio * 10)  # 10% 이상이면 최저 점수
    
    # 4. 신호 연속성 평가
    # 연속된 샘플 간의 급격한 변화 감지
    diff = np.abs(np.diff(signal))
    sudden_changes = diff > 5 * np.std(diff)
    continuity_score = 1.0 - min(1.0, np.sum(sudden_changes) / len(diff) * 10)
    
    # 종합 품질 점수 계산 (가중 평균)
    quality_score = (snr_score * 0.4 + 
                     baseline_score * 0.3 + 
                     outlier_score * 0.2 + 
                     continuity_score * 0.1)
    
    return quality_score




# TODO : 수정 필요(구조적인 문제)
def r_peak_detection(ecg_signal, qrs_mask, rpeak_model, device, fs=500):
    """
    R-peak 검출 함수
    
    Args:
        ecg_signal: ECG 신호
        qrs_mask: QRS 마스크
        rpeak_model: R-peak 위치 예측 회귀 모델
        device: 연산 장치 (CPU 또는 GPU)
        fs: 샘플링 주파수 (Hz)
    
    Returns:
        r_peak_locations: 검출된 R-peak 위치 목록
    """
    # 1. QRS 마스크 후처리
    refined_mask = refine_qrs_mask(qrs_mask)
    # 2. QRS 컴플렉스 추출
    qrs_complexes, qrs_regions, _ = extract_qrs_complexes(ecg_signal, refined_mask)
    if len(qrs_complexes) == 0:
        return []
    # 3. 각 QRS 컴플렉스에서 R 피크 위치 예측
    r_peak_locations = []
    with torch.no_grad():
        for i, qrs in enumerate(qrs_complexes):
            # 입력 텐서 형태 확인 및 조정
            qrs_tensor = torch.FloatTensor(qrs).unsqueeze(0).unsqueeze(0).to(device)
            
            # 정규화된 위치 예측 (0~1 사이 값)
            normalized_position = rpeak_model(qrs_tensor).item()
            normalized_position = max(0, min(1, normalized_position))
            # 정규화된 위치를 실제 샘플 위치로 변환
            start, end = qrs_regions[i]
            r_peak_idx = start + int(normalized_position * (end - start))
            r_peak_locations.append(r_peak_idx)
    # 4. R-peak 위치 정밀화
    refined_peaks = refine_r_peak_locations(ecg_signal, r_peak_locations)
    
    # 5. 생리학적 제약 기반 검증
    validated_peaks = validate_r_peaks(ecg_signal, refined_peaks, fs)
    
    return validated_peaks

# TODO : 수정 필요(구조적인 문제)
def predict_r_peaks(ecg_signal, qrs_mask, rpeak_model, device, fs=500):
    """
    R-peak 검출 함수
    
    Args:
        ecg_signal: ECG 신호 데이터
        qrs_mask: QRS 복합체 마스크 (U-Net 예측 결과)
        rpeak_model: R 피크 위치 예측 회귀 모델
        device: 연산 장치 (CPU 또는 GPU)
        fs: 샘플링 주파수 (Hz)
        
    Returns:
        dict: R-peak 검출 결과와 신호 품질 정보를 포함한 딕셔너리
    """
    rpeak_model.to(device)
    rpeak_model.eval()
    
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
    
    # R-peak 검출
    r_peak_locations = r_peak_detection(ecg_signal, qrs_mask, rpeak_model, device, fs)
    
    # 결과 딕셔너리 생성
    result = {
        'R-peak Regression': {
            'original_rpeaks': np.array(r_peak_locations),
            'corrected_rpeaks': np.array(r_peak_locations),
            'plot_signal': ecg_signal
        }
    }
    
    return result

# tol 일반적으로 활용하는 오차 범위 50 ms
def tolerance_matching(true_peaks, predicted_peaks, fs=500, tol=50):
    tolerance_samples = int((tol / 1000) * fs)
    matched_predicted = set()
    TP, FP, FN = 0, 0, 0
    for t in true_peaks:
        if len(predicted_peaks) == 0:
            FN += 1
            continue

        distances = np.abs(np.array(predicted_peaks) - t)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist <= tolerance_samples:
            TP += 1
            matched_predicted.add(min_dist_idx)
        else:
            FN += 1

    FP = len(predicted_peaks) - len(matched_predicted)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1
    }
    
    
# UNet 모델을 사용하여 QRS 컴플렉스 예측 및 시각화
def visualize_qrs_prediction(model, ecg_signal, true_mask=None, threshold=0.5, save_path=None):
    """
    UNet 모델을 사용하여 QRS 컴플렉스를 예측하고 시각화하는 함수
    
    Args:
        model: 학습된 UNet 모델
        ecg_signal: 입력 ECG 신호 (정규화된 형태)
        true_mask: 실제 QRS 마스크 (선택적)
        threshold: 예측 임계값 (기본값: 0.5)
    """
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 입력 데이터 준비
    with torch.no_grad():
        input_tensor = torch.FloatTensor(ecg_signal).unsqueeze(0).to(device)
        
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)
        
        # 예측 결과를 CPU로 이동하고 numpy 배열로 변환
        predicted_mask = (probs > threshold).float().cpu().squeeze().numpy()
    
    # 시각화
    plt.figure(figsize=(15, 6))
    
    # ECG 신호 그리기
    plt.plot(ecg_signal.squeeze().cpu().numpy(), label='ECG Signal')
    
    # 예측된 QRS 마스크 그리기
    plt.plot(predicted_mask, 'g', alpha=0.6, label='Predicted QRS Mask')
    # # 예측 확률값 그리기
    # plt.plot(probs.squeeze().cpu().numpy(), 'b', alpha=0.6, label='Prediction Probability')
    
    # 실제 마스크가 제공된 경우 함께 표시
    if true_mask is not None:
        plt.plot(true_mask.squeeze().cpu().numpy(), 'r', alpha=0.4, label='True QRS Mask')
    
    plt.title('ECG Signal and Predicted QRS Complex')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    