import numpy as np
import scipy
import neurokit2 as nk
import biosppy

from preprocess import preprocess_ecg
from util import assess_signal_quality
from pantom import Pan_Tompkins_Plus_Plus

# ============================================================================
# peak detection 유틸리티
# ============================================================================

def find_peak_in_range(signal, r_idx, sample_range, min_mode=False, tol=0.2):
    """
    지정된 범위 내에서 최대값 또는 최소값을 찾는 함수
    
    Args:
        signal: ECG 신호
        r_idx: 기준 R-peak 위치
        sample_range: 탐색 범위 (시작, 끝) 튜플
        min_mode: True면 최소값, False면 최대값 탐색
        
    Returns:
        peak_idx: 찾은 피크의 인덱스, 유효한 범위가 없으면 None
    """
    start = max(0, round(r_idx + sample_range[0] * tol + sample_range[0]))
    end = min(len(signal) - 1, round(r_idx + sample_range[1] * tol + sample_range[1]))
    
    if start >= end:
        return None
    
    if min_mode:
        peak_offset = np.argmin(signal[start:end])
    else:
        peak_offset = np.argmax(signal[start:end])
    
    return start + peak_offset

def correct_rpeaks(signal=None, rpeaks=None, sampling_rate=1000.0, tol=0.05):
    """허용 오차 내에서 최대값으로 R-peak 위치를 보정합니다.

    Parameters
    ----------
    signal : array
        ECG 신호
    rpeaks : array
        R-peak 위치 인덱스
    sampling_rate : int, float, optional
        샘플링 주파수 (Hz)
    tol : int, float, optional
        보정 허용 오차 (초)

    Returns
    -------
    rpeaks : array
        보정된 R-peak 위치 인덱스
    """
    # 입력 확인
    if signal is None:
        raise TypeError("입력 신호를 지정해주세요.")

    if rpeaks is None:
        raise TypeError("입력 R-peak를 지정해주세요.")

    tol = int(tol * sampling_rate)
    length = len(signal)

    newR = []
    for r in rpeaks:
        a = r - tol
        if a < 0:
            a = 0
        b = r + tol
        if b > length:
            b = length
        newR.append(a + np.argmax(signal[a:b]))

    newR = sorted(list(set(newR)))
    newR = np.array(newR, dtype="int")

    return newR

def calculate_statistics(data, feature_name):
    """
    데이터의 통계 정보를 계산하고 출력하는 함수
    
    Args:
        data: 통계를 계산할 데이터 리스트
        feature_name: 특성 이름 (출력용)
        
    Returns:
        stats: 통계 정보를 담은 딕셔너리
    """
    # 통계 정보 계산
    avg_value = np.mean(data)
    std_value = np.std(data)
    min_value = np.min(data)
    max_value = np.max(data)
    median_value = np.median(data)
    q1_value = np.percentile(data, 25)
    q3_value = np.percentile(data, 75)
    iqr_value = q3_value - q1_value

    # 통계 정보를 딕셔너리로 반환
    stats = {
        "mean": avg_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "median": median_value,
        "q1": q1_value,
        "q3": q3_value,
        "iqr": iqr_value
    }
    
    return stats


# ============================================================================
# Pan-Tompkins 알고리즘
# ============================================================================

def pantom_rpeak_detection(ecg, low_cutoff=5, high_cutoff=15, fs=500):
    """
    ECG 신호에서 R-peak를 검출하는 함수
    
    Args:
        ecg (numpy.ndarray): ECG 신호 데이터
        fs (int): 샘플링 주파수 (Hz)
        
    Returns:
        numpy.ndarray: 검출된 R-peak의 인덱스
    """
    # 파라미터 설정
    max_QRS_duration = 0.150  # sec
    window_size = int(max_QRS_duration * fs)
    
    # 밴드패스 필터 적용
    lowpass = scipy.signal.butter(1, high_cutoff / (fs / 2.0), "low")
    highpass = scipy.signal.butter(1, low_cutoff / (fs / 2.0), "high")
    ecg_low = scipy.signal.filtfilt(*lowpass, x=ecg)
    ecg_band = scipy.signal.filtfilt(*highpass, x=ecg_low)
    
    # 미분 및 제곱
    diff = np.diff(ecg_band)
    squared = np.square(diff)
    
    # 이동 평균 필터
    mwa = np.pad(squared, (window_size - 1, 0), "constant", constant_values=(0, 0))
    mwa = np.convolve(mwa, np.ones(window_size), "valid")
    for i in range(1, window_size):
        mwa[i - 1] = mwa[i - 1] / i
    mwa[window_size - 1 :] = mwa[window_size - 1 :] / window_size
    mwa[: int(max_QRS_duration * fs * 2)] = 0
    
    # 가우시안 필터로 신호 스무딩 및 미분
    energy = scipy.ndimage.gaussian_filter1d(mwa, fs / 8.0)
    energy_diff = np.diff(energy)
    
    # 미분값이 0을 교차하는 지점 찾기 (피크 위치)
    r_peaks = (energy_diff[:-1] > 0) & (energy_diff[1:] < 0)
    r_peaks = np.flatnonzero(r_peaks)
    r_peaks -= int(window_size / 2)
    return {'rpeaks': r_peaks}


# ============================================================================
# 앙상블 방법
# ============================================================================

def ensemble_rpeaks_detection(all_rpeaks, preprocessed_signal, sampling_rate, min_votes=3, use_weighted=True, debug=False, algorithm_weights=None, algorithm_names=None):
    """
    여러 알고리즘에서 검출된 R-peaks를 앙상블하여 최종 R-peaks를 결정하는 함수
    
    Parameters:
    -----------
    all_rpeaks : list
        각 알고리즘에서 검출된 R-peaks 리스트
    preprocessed_signal : ndarray
        전처리된 ECG 신호
    sampling_rate : int
        샘플링 레이트 (Hz)
    min_votes : int, optional
        최소 투표 수 (기본값: 3)
    use_weighted : bool, optional
        가중치 기반 앙상블 사용 여부 (기본값: True)
    debug : bool, optional
        디버깅 정보 출력 여부 (기본값: False)
    algorithm_weights : dict, optional
        각 알고리즘별 사전 정의된 가중치 (기본값: None)
    algorithm_names : list, optional
        all_rpeaks 리스트와 매칭되는 알고리즘 이름 리스트 (기본값: None)
        
    Returns:
    --------
    ndarray
        앙상블 방법으로 보정된 최종 R-peaks
    """
    # 모든 R-peaks를 하나의 배열로 합치고 정렬
    all_peaks_combined = np.sort(np.concatenate(all_rpeaks))
    
    # 유사한 위치의 피크를 그룹화하기 위한 허용 오차 설정 (샘플 단위)
    tolerance = int(0.05 * sampling_rate)  # 50ms 허용 오차

    # 그룹화된 피크 위치 저장 (최적화된 방식)
    grouped_peaks = []
    current_group = [all_peaks_combined[0]]
    
    for peak in all_peaks_combined[1:]:
        if peak - current_group[-1] <= tolerance:
            current_group.append(peak)
        else:
            grouped_peaks.append(int(np.median(current_group)))
            current_group = [peak]
            
    # 마지막 그룹 처리
    if current_group:
        grouped_peaks.append(int(np.median(current_group)))
    
    # 가중치 계산 (use_weighted가 False면 건너뜀)
    weights = np.ones(len(all_rpeaks))
    
    if use_weighted:
        # 1. RR 간격 기반 품질 점수 계산
        for i, algorithm_peaks in enumerate(all_rpeaks):
            if len(algorithm_peaks) > 1:
                # RR 간격 계산 (ms 단위)
                rr_intervals = np.diff(algorithm_peaks) / sampling_rate * 1000
                
                # 생리학적 유효성 검사를 벡터화
                valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
                valid_ratio = np.mean(valid_mask)
                
                # 통계 계산
                rr_mean = np.mean(rr_intervals)
                rr_std = np.std(rr_intervals)
                rr_median = np.median(rr_intervals)
                
                # 변동 계수 (낮을수록 좋음)
                cv = rr_std / rr_mean if rr_mean > 0 else 1.0
                
                # 극단적 이상치 비율 (낮을수록 좋음)
                outlier_mask = np.abs(rr_intervals - rr_median) > (3 * rr_std)
                outlier_ratio = np.mean(outlier_mask)
                
                # 심박수 범위 검사 (60-180 BPM 범위 내 비율)
                hr_valid_mask = (rr_intervals >= 333) & (rr_intervals <= 1000)  # 60-180 BPM
                hr_valid_ratio = np.mean(hr_valid_mask)
                
                # 종합 품질 점수 계산 (0~1 사이 값) - 심박수 범위 검사 추가
                quality_score = (valid_ratio * 0.4) + ((1 - cv) * 0.2) + ((1 - outlier_ratio) * 0.2) + (hr_valid_ratio * 0.2)
                weights[i] = max(0.1, min(1.0, quality_score))
            else:
                weights[i] = 0.1
        
        # 2. 사전 정의된 알고리즘 가중치 적용 (있는 경우)
        if algorithm_weights is not None and algorithm_names is not None:
            for i, alg_name in enumerate(algorithm_names):
                if i < len(weights) and alg_name in algorithm_weights:
                    # 품질 점수와 사전 정의된 가중치를 곱하여 최종 가중치 계산
                    weights[i] *= algorithm_weights[alg_name]
        
        # 가중치 정규화 (합이 알고리즘 수와 같도록)
        weights = weights / np.sum(weights) * len(weights)
        
        # 디버깅 정보 출력
        if debug and algorithm_names is not None:
            print("알고리즘별 가중치:")
            for i, alg_name in enumerate(algorithm_names):
                if i < len(weights):
                    print(f"  {alg_name}: {weights[i]:.2f}")
    
    # 투표 계산 (딕셔너리 대신 defaultdict 사용)
    from collections import defaultdict
    votes = defaultdict(float)
    
    # 각 알고리즘의 R-peaks에 대해 투표 계산
    for i, algorithm_peaks in enumerate(all_rpeaks):
        weight = weights[i]
        
        # 각 피크에 대해 가장 가까운 그룹화된 피크 찾기
        for peak in algorithm_peaks:
            # NumPy 배열 연산으로 최적화
            distances = np.abs(np.array(grouped_peaks) - peak)
            closest_idx = np.argmin(distances)
            closest_peak = grouped_peaks[closest_idx]
            
            # 허용 오차 내에 있는지 확인
            if distances[closest_idx] <= tolerance:
                votes[closest_peak] += weight
                
    # 최소 투표 수 이상을 받은 피크만 선택
    ensemble_rpeaks = np.array([peak for peak, vote_count in votes.items() if vote_count >= min_votes])
    ensemble_rpeaks = np.sort(ensemble_rpeaks)
    
    # 디버깅 정보 출력
    if debug:
        from pprint import pprint
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        print(f"최소 투표 수: {min_votes:.2f}")
        print("투표 수가 높은 순서대로 정렬:")
        pprint(sorted_votes)
    
    return ensemble_rpeaks

def extract_ecg_and_rpeaks(raw_signal, sampling_rate, lowcut=0.5, highcut=40, min_votes_ratio=0.3, debug=False, r_min=250, r_max=4750):
    """
    ECG 신호에서 R-peak를 추출하는 함수
    앙상블 방법을 사용하여 더 정확한 R-peak 검출
    
    Parameters:
    -----------
    raw_signal : numpy.ndarray
        ECG 신호 데이터
    sampling_rate : float
        샘플링 레이트
    lowcut : float
        대역 통과 필터의 하한 주파수 (Hz)
    highcut : float
        대역 통과 필터의 상한 주파수 (Hz)
    min_votes_ratio : float
        최소 투표 비율 (0~1 사이, 기본값: 0.3)
        전체 알고리즘 중 이 비율 이상의 알고리즘이 동의해야 R-peak로 인정
    debug : bool
        debug 모드 여부
    r_min : int, optional
        R-peak 검출 시작 인덱스 (기본값: 250)
    r_max : int, optional
        R-peak 검출 종료 인덱스 (기본값: 4750)
        
    Returns:
    --------
    dict: {
        'raw_signal': 원본 ECG 신호,
        'preprocessed_signal': 통일된 전처리 ECG 신호,
        'ensemble_rpeaks': 앙상블 방법으로 결정된 R-peak 위치,
        'plot_signal': 시각화를 위한 신호,
        'algorithm_results': 각 알고리즘별 R-peak 결과 딕셔너리,
        'signal_quality': 신호 품질 점수
    }
    """
    m_detector = Pan_Tompkins_Plus_Plus()
    
    # 신호 품질 평가
    signal_quality = assess_signal_quality(raw_signal, sampling_rate)
    
    # 전처리 신호 생성 - 모든 알고리즘에 동일하게 사용
    preprocessed_signal = preprocess_ecg(raw_signal, sampling_rate, lowcut, highcut)
    
    info_pantom = pantom_rpeak_detection(raw_signal, sampling_rate)
    info_pantom_pp = m_detector.rpeak_detection(raw_signal, sampling_rate)
    
    ecg_cleaned = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate)
    _, info_nk = nk.ecg_peaks(ecg_cleaned=ecg_cleaned, sampling_rate=sampling_rate)
    
    algorithms = {
        'NeuroKit2': {'rpeaks': info_nk['ECG_R_Peaks']},
        'Pan-Tomkins': {'rpeaks': info_pantom['rpeaks']},
        'Pan-Tomkins++': {'rpeaks': info_pantom_pp['rpeaks']},
        'Hamilton': biosppy.signals.ecg.hamilton_segmenter,
        'Christov': biosppy.signals.ecg.christov_segmenter,
        'Engzee': biosppy.signals.ecg.engzee_segmenter
    }
    
    num_algorithms = len(algorithms)
    min_votes = max(2, round(num_algorithms * min_votes_ratio))
    
    # 각 알고리즘 적용 및 보정
    all_rpeaks = []
    algorithm_results = {}
    
    for name, algorithm in algorithms.items():
        if isinstance(algorithm, dict):
            rpeaks = algorithm['rpeaks']
        else:
            result = algorithm(preprocessed_signal, sampling_rate=sampling_rate)
            rpeaks = result['rpeaks']
            
        # 모든 알고리즘에 동일한 신호로 R-peak 위치 보정
        corrected_rpeaks = correct_rpeaks(raw_signal, rpeaks, sampling_rate=sampling_rate, tol=0.05)
        all_rpeaks.append(corrected_rpeaks)
        
        # 각 알고리즘별 결과 저장
        algorithm_results[name] = {
            'original_rpeaks': rpeaks,
            'corrected_rpeaks': corrected_rpeaks,
            'plot_signal': raw_signal  # 모든 알고리즘에 동일한 신호 사용
        }
    
    # 알고리즘 가중치 설정
    algorithm_weights = {
        'NeuroKit2': 1.0,
        'Pan-Tomkins': 0.8,
        'Pan-Tomkins++': 1.3,
        'Hamilton': 0.8,
        'Christov': 1.2,
        'Engzee': 0.7,
    }
    
    # 신호 품질에 따른 가중치 조정
    if signal_quality < 0.7:  # 낮은 품질
        if debug:
            print(f"신호 품질이 매우 낮음 ({signal_quality:.2f})")
        # 노이즈에 강한 알고리즘 우선
        for alg in algorithm_weights:
            if alg == 'Engzee':
                algorithm_weights[alg] *= 2.0
            else:
                algorithm_weights[alg] *= 0.9
    elif signal_quality < 0.75:  # 보통 품질
        if debug:
            print(f"신호 품질이 낮음 ({signal_quality:.2f})")
        algorithm_weights['Engzee'] *= 1.5
        algorithm_weights['Christov'] *= 0.9
    elif signal_quality > 0.85:  # 높은 품질
        if debug:
            print(f"신호 품질이 우수함 ({signal_quality:.2f})")
        # 정밀한 알고리즘 우선
        algorithm_weights['NeuroKit2'] *= 1.2
        algorithm_weights['Hamilton'] *= 1.1
        algorithm_weights['Pan-Tomkins++'] *= 1.1
        
    # 알고리즘 이름 리스트 (all_rpeaks와 동일한 순서)
    algorithm_names = list(algorithm_results.keys())
    
    # 앙상블 방법으로 최종 R-peaks 결정 (가중치 전달)
    ensemble_rpeaks = ensemble_rpeaks_detection(
        all_rpeaks, 
        raw_signal,
        sampling_rate, 
        min_votes=min_votes, 
        use_weighted=True,
        algorithm_weights=algorithm_weights,
        algorithm_names=algorithm_names,
        debug=debug 
    )
    ensemble_rpeaks = ensemble_rpeaks[(ensemble_rpeaks >= r_min) & (ensemble_rpeaks <= r_max)]
    # 결과를 딕셔너리로 반환 - 키 이름 통일
    results = {
        'raw_signal': raw_signal,
        'preprocessed_signal': preprocessed_signal,
        'ensemble_rpeaks': ensemble_rpeaks,
        'plot_signal': raw_signal,
        'algorithm_results': algorithm_results,
        'signal_quality': signal_quality
    }
    return results

def find_peaks_relative_to_rr(signal, r_peaks, sampling_rate=500, debug=False, tol=0.2):
    """
    절대 시간 범위(ms)와 상대 범위(%)를 조합하여 R-peak 기준으로 다른 피크들을 찾는 함수
    
    Args:
        signal: ECG 신호
        r_peaks: 검출된 R-peak 위치 배열
        sampling_rate: 샘플링 레이트 (Hz)
        debug: 디버그 모드 활성화 여부
        tol: 피크 검출 시 허용 오차 (0.0~1.0 사이의 값, 기본값: 0.0)
            - 값이 클수록 더 넓은 범위에서 피크를 검색함
        
    Returns:
        peaks: 모든 피크 위치를 담은 딕셔너리
    """
    peaks = {'ECG_R_Peaks': r_peaks, 'ECG_Q_Peaks': [], 'ECG_S_Peaks': [], 'ECG_P_Peaks': [], 'ECG_T_Peaks': []}
    
    # 임상적 절대 시간 범위 (ms)
    clinical_p_range = (-240, -80)
    clinical_q_range = (-40, -30)
    clinical_s_range = (20, 50)
    clinical_t_range = (150, 300)
    
    # RR 간격을 기준으로 탐색 범위 동적 조정
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000
        mean_rr = np.mean(rr_intervals)
    else:
        # R-peak가 하나만 있는 경우 기본값 설정
        mean_rr = 1000  # 1초 (60 bpm에 해당)
    
    # 심박수 계산 (bpm)
    hr = 60000 / mean_rr
    if debug:
        print(f"평균 RR 간격: {mean_rr:.2f}ms, 심박수: {hr:.2f}bpm")
    
    # 정상 심박수 범위 정의
    normal_hr_min = 60
    normal_hr_max = 100
    
    # 샘플 변환 계수 미리 계산
    ms_to_sample = sampling_rate / 1000
    
    # 각 R-peak에 대해 개별적으로 탐색 범위 조정
    for i, r_idx in enumerate(r_peaks):
        # 현재 심박의 RR 간격 계산
        if i > 0:
            # 이전 RR 간격 (현재 R-peak와 이전 R-peak 사이)
            pre_local_rr = (r_idx - r_peaks[i-1]) / ms_to_sample  # ms
        else:
            # 첫 번째 R-peak는 다음 간격 사용 (없으면 평균 사용)
            pre_local_rr = mean_rr
            
        # 다음 RR 간격 (현재 R-peak와 다음 R-peak 사이)
        if i < len(r_peaks) - 1:
            post_local_rr = (r_peaks[i+1] - r_idx) / ms_to_sample  # ms
            next_r_idx = r_peaks[i+1]
            estimated_q_offset = round(clinical_q_range[0] * ms_to_sample)  # Q-peak의 일반적인 오프셋
            next_q_idx = next_r_idx + estimated_q_offset  # 다음 R-peak 기준 Q-peak 예상 위치
        else:
            # 마지막 R-peak는 이전 간격 사용
            post_local_rr = pre_local_rr
            next_r_idx = float('inf')  # 기본값으로 무한대 설정
            next_q_idx = float('inf')  # 기본값으로 무한대 설정
        
        # 심박수 계산 (이전 및 다음 RR 간격 기반)
        pre_local_hr = 60000 / pre_local_rr
        post_local_hr = 60000 / post_local_rr
        
        # 심박수에 따른 동적 조정 계수 계산
        pre_local_hr_factor = 1.0
        post_local_hr_factor = 1.0
        
        if pre_local_hr <= normal_hr_min:
            # 느린 심박수: 정상 최소값(60bpm)을 기준으로 조정
            pre_local_hr_factor = normal_hr_min / pre_local_hr
        elif pre_local_hr >= normal_hr_max:
            # 빠른 심박수: 정상 최대값(100bpm)을 기준으로 조정
            pre_local_hr_factor = normal_hr_max / pre_local_hr
            
        if post_local_hr <= normal_hr_min:
            post_local_hr_factor = normal_hr_min / post_local_hr
        elif post_local_hr >= normal_hr_max:
            post_local_hr_factor = normal_hr_max / post_local_hr
        
        # 절대 범위 조정 (심박수에 따라 약간 조정)
        # P, Q 파형은 이전 심박수 기준으로 조정
        # S, T 파형은 다음 심박수 기준으로 조정
        
        # 샘플 단위로 변환 (한 번에 계산)
        p_samples = (round(clinical_p_range[0] * pre_local_hr_factor * ms_to_sample), 
                     round(clinical_p_range[1] * pre_local_hr_factor * ms_to_sample))
        q_samples = (round(clinical_q_range[0] * pre_local_hr_factor * ms_to_sample), 
                     round(clinical_q_range[1] * pre_local_hr_factor * ms_to_sample))
        s_samples = (round(clinical_s_range[0] * post_local_hr_factor * ms_to_sample), 
                     round(clinical_s_range[1] * post_local_hr_factor * ms_to_sample))
        t_samples = (round(clinical_t_range[0] * post_local_hr_factor * ms_to_sample), 
                     round(clinical_t_range[1] * post_local_hr_factor * ms_to_sample))
        
        # 피크 검출
        q_idx = find_peak_in_range(signal, r_idx, q_samples, min_mode=True, tol=tol)
        if q_idx is not None and q_idx < next_r_idx: 
            peaks['ECG_Q_Peaks'].append(q_idx)
        
        s_idx = find_peak_in_range(signal, r_idx, s_samples, min_mode=True, tol=tol)
        if s_idx is not None and s_idx < next_r_idx: 
            peaks['ECG_S_Peaks'].append(s_idx)
        
        t_idx = find_peak_in_range(signal, r_idx, t_samples, min_mode=False, tol=tol)
        if t_idx is not None and t_idx < next_q_idx: 
            peaks['ECG_T_Peaks'].append(t_idx)
        
        p_idx = find_peak_in_range(signal, r_idx, p_samples, min_mode=False, tol=tol)
        if p_idx is not None and p_idx < next_r_idx: 
            peaks['ECG_P_Peaks'].append(p_idx)
    
        # 디버그 정보 출력 (필요한 경우만)
        if debug and (pre_local_hr > 100 or post_local_hr > 100 or post_local_hr < 60 or pre_local_hr < 60):
            print(f"R-peak 위치: {r_idx}")
            print(f"이전 심박수: {pre_local_hr:.1f} bpm, 이전 RR 간격: {pre_local_rr:.1f} ms")
            print(f"다음 심박수: {post_local_hr:.1f} bpm, 다음 RR 간격: {post_local_rr:.1f} ms")
            print(f"이전 심박수 범위: {pre_local_hr_factor:.2f}, 다음 심박수 범위: {post_local_hr_factor:.2f}")
            print("범위 정보 (sample):")
            print(f"P-wave 범위: {r_idx + p_samples[0]} ~ {r_idx + p_samples[1]} (샘플: {p_samples[0]} ~ {p_samples[1]})")
            print(f"Q-wave 범위: {r_idx + q_samples[0]} ~ {r_idx + q_samples[1]} (샘플: {q_samples[0]} ~ {q_samples[1]})")
            print(f"S-wave 범위: {r_idx + s_samples[0]} ~ {r_idx + s_samples[1]} (샘플: {s_samples[0]} ~ {s_samples[1]})")
            print(f"T-wave 범위: {r_idx + t_samples[0]} ~ {r_idx + t_samples[1]} (샘플: {t_samples[0]} ~ {t_samples[1]})")
            print(f"P-peak 인덱스: {p_idx}")
            print(f"Q-peak 인덱스: {q_idx}")
            print(f"R-peak 인덱스: {r_idx}")
            print(f"S-peak 인덱스: {s_idx}")
            print(f"T-peak 인덱스: {t_idx}")
            print("-" * 50)
            
    # 배열로 변환 (한 번에 처리)
    for key in peaks:
        if key != 'ECG_R_Peaks' and peaks[key]:  # 빈 리스트가 아닌 경우만 변환
            peaks[key] = np.array(peaks[key])
    
    return peaks


# # ============================================================================
# # 정답 값과 모델 예측 결과 비교 TODO: 수정 필요 (구조적인 문제)
# # ============================================================================
# def r_peak_prediction(ecg_signal, unet_model, rpeak_model, device, best_threshold, sampling_rate=500):
#     """
#     ECG 신호에서 R-peak를 예측하는 함수
    
#     Parameters:
#     -----------
#     ecg_signal : numpy.ndarray
#         ECG 신호 데이터 (numpy 배열)
#     unet_model : torch.nn.Module
#         학습된 U-Net 모델
#     rpeak_model : torch.nn.Module
#         학습된 R-peak 회귀 모델
#     device : torch.device
#         연산을 수행할 디바이스
#     best_threshold : float
#         U-Net 모델의 출력을 이진화하기 위한 임계값
#     sampling_rate : int, optional
#         ECG 신호의 샘플링 주파수 (Hz), 기본값: 500
        
#     Returns:
#     --------
#     numpy.ndarray
#         최종 검출된 R-peak 위치 배열
#     """
#     # numpy 배열을 텐서로 변환
#     if isinstance(ecg_signal, np.ndarray):
#         test_ecg = torch.from_numpy(ecg_signal).float().to(device)
#     else:
#         test_ecg = ecg_signal.to(device)
    
#     # 차원 확인 및 조정
#     if len(test_ecg.shape) == 1:
#         test_ecg = test_ecg.unsqueeze(0).unsqueeze(0)  # [1, 1, signal_length]
#     elif len(test_ecg.shape) == 2:
#         test_ecg = test_ecg.unsqueeze(1)  # [batch, 1, signal_length]
    
#     # U-Net 모델로 QRS 마스크 예측
#     with torch.no_grad():
#         outputs = unet_model(test_ecg)
#         qrs_mask = (torch.sigmoid(outputs) > best_threshold).float().cpu().squeeze().numpy()
    
#     # ECG 신호를 numpy로 변환
#     current_ecg_np = test_ecg.cpu().numpy().squeeze()
    
#     # 앙상블 방법으로 R-peak 검출
#     result = extract_ecg_and_rpeaks(current_ecg_np, sampling_rate, 0.5, 40, min_votes_ratio=0.5)
    
#     # 회귀 모델로 R-peak 예측
#     model_rpeaks = predict_r_peaks(current_ecg_np, qrs_mask, rpeak_model, device, sampling_rate)
    
#     # 앙상블 방법으로 검출된 R-peak
#     ensemble_rpeaks = np.array(result['ensemble_rpeaks'])
    
#     # 회귀 모델로 예측된 R-peak
#     regression_rpeaks = np.array(model_rpeaks['R-peak Regression']['corrected_rpeaks'])
    
#     # 신호 품질에 따른 최종 R-peak 결정
#     signal_quality = result['signal_quality']
    
#     # 샘플링 레이트 기반 시간 간격 계산 (ms 단위)
#     time_150ms = int(0.15 * sampling_rate)
#     time_100ms = int(0.1 * sampling_rate)
#     time_50ms = int(0.05 * sampling_rate)
    
#     # 신호 품질에 따른 R-peak 결정 전략
#     if signal_quality > 0.85:  # 높은 신호 품질
#         # 앙상블 결과 우선 사용, 회귀 모델로 보완
#         final_rpeaks = ensemble_rpeaks.copy()
        
#         # 앙상블에서 놓친 R-peak를 회귀 모델에서 찾아 추가
#         if len(ensemble_rpeaks) > 0:
#             for reg_peak in regression_rpeaks:
#                 # 가장 가까운 앙상블 피크와의 거리 계산
#                 min_distance = np.min(np.abs(ensemble_rpeaks - reg_peak))
#                 if min_distance > time_150ms:  # 150ms 이상 차이나면 추가
#                     final_rpeaks = np.append(final_rpeaks, reg_peak)
#         else:
#             final_rpeaks = regression_rpeaks.copy()
                
#     elif signal_quality > 0.6:  # 중간 신호 품질
#         # 회귀 모델 결과 우선 사용
#         final_rpeaks = regression_rpeaks.copy()
        
#         # 회귀 모델에 없는 앙상블 피크 추가 검토
#         if len(regression_rpeaks) > 0:
#             for ens_peak in ensemble_rpeaks:
#                 min_distance = np.min(np.abs(regression_rpeaks - ens_peak))
#                 if min_distance > time_100ms:  # 100ms 이상 차이나면 추가
#                     final_rpeaks = np.append(final_rpeaks, ens_peak)
#         else:
#             final_rpeaks = ensemble_rpeaks.copy()
                
#     else:  # 낮은 신호 품질
#         # 두 모델 모두에서 검출된 피크만 신뢰
#         final_rpeaks = np.array([], dtype=int)
        
#         # 두 결과가 유사한 피크만 선택
#         for peak in regression_rpeaks:
#             if len(ensemble_rpeaks) > 0:
#                 min_distance = np.min(np.abs(ensemble_rpeaks - peak))
#                 if min_distance < time_50ms:  # 50ms 이내에 앙상블 피크가 있으면 추가
#                     final_rpeaks = np.append(final_rpeaks, peak)
        
#         # 결과가 너무 적으면 회귀 모델 결과 사용
#         if len(final_rpeaks) < 3 and len(regression_rpeaks) >= 3:
#             final_rpeaks = regression_rpeaks.copy()
    
#     # 중복 제거 및 정렬
#     if len(final_rpeaks) > 0:
#         final_rpeaks = np.unique(final_rpeaks.astype(int))
#         final_rpeaks = np.sort(final_rpeaks)
    
#     results = {
#         'regression_rpeaks': regression_rpeaks,
#         'ensemble_rpeaks': final_rpeaks,
#         'plot_signal': ecg_signal,
#         'signal_quality': signal_quality
#     }
#     return results
