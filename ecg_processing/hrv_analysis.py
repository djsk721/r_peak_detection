import numpy as np 
from preprocessing.util import samples_to_ms
from preprocessing.util import calculate_statistics

# ECG 특성 추출 통합 함수 정의
def extract_ecg_features(peaks, sampling_rate=500):
    """
    ECG 신호에서 다양한 특성(PR 간격, QRS 지속 시간, ST 세그먼트, QT 간격)을 추출하는 통합 함수
    
    Args:
        peaks: ECG 신호에서 검출된 피크 정보를 담은 딕셔너리
        sampling_rate: 샘플링 레이트 (Hz), 기본값 500Hz
        
    Returns:
        features: 추출된 모든 특성과 통계 정보를 담은 딕셔너리
    """
    features = {}
    
    # 필요한 피크 데이터 추출
    p_peaks = peaks.get('ECG_P_Peaks', [])
    q_peaks = peaks.get('ECG_Q_Peaks', [])
    r_peaks = peaks.get('ECG_R_Peaks', [])
    s_peaks = peaks.get('ECG_S_Peaks', [])
    t_peaks = peaks.get('ECG_T_Peaks', [])
    
    rr_intervals = np.diff(r_peaks) * (1000 / sampling_rate)
    mean_rr = rr_intervals  # 평균 RR 간격 (ms)
    
    # 1. PR 간격 계산
    pr_intervals_ms = []
    min_length = min(len(p_peaks), len(r_peaks))
    for i in range(min_length):
        pr_interval = r_peaks[i] - p_peaks[i]
        pr_interval = samples_to_ms(pr_interval, sampling_rate)
        pr_intervals_ms.append(pr_interval)
    
    # 2. QRS 지속 시간 계산
    qrs_durations_ms = []
    min_length = min(len(q_peaks), len(s_peaks))
    for i in range(min_length):
        qrs_duration = s_peaks[i] - q_peaks[i]
        qrs_duration = samples_to_ms(qrs_duration, sampling_rate)
        qrs_durations_ms.append(qrs_duration)
    
    # 3. ST 세그먼트 계산
    st_segments_ms = []
    min_length = min(len(s_peaks), len(t_peaks))
    for i in range(min_length):
        st_segment = t_peaks[i] - s_peaks[i]
        st_segment = samples_to_ms(st_segment, sampling_rate)
        st_segments_ms.append(st_segment)
    
    # 4. QT 간격 계산
    qt_intervals_ms = []
    min_length = min(len(q_peaks), len(t_peaks))
    for i in range(min_length):
        qt_interval = t_peaks[i] - q_peaks[i]
        qt_interval = samples_to_ms(qt_interval, sampling_rate)
        qt_intervals_ms.append(qt_interval)
        
    # 5. QTC 간격 계산
    qtc_intervals_ms = []
    min_length = len(qt_intervals_ms)
    mean_rr_value = np.mean(mean_rr)
    for i in range(min_length):
        # 마지막 간격에는 mean_rr의 평균값 사용
        qtc_interval = qt_intervals_ms[i] / np.sqrt(mean_rr_value)
        qtc_intervals_ms.append(qtc_interval)
    
    # 각 특성에 대한 통계 계산 및 저장
    features['PR_Interval'] = {
        'values': pr_intervals_ms,
        'stats': calculate_statistics(pr_intervals_ms, "PR 간격")
    }
    
    features['QRS_Duration'] = {
        'values': qrs_durations_ms,
        'stats': calculate_statistics(qrs_durations_ms, "QRS 지속 시간")
    }
    
    features['ST_Segments'] = {
        'values': st_segments_ms,
        'stats': calculate_statistics(st_segments_ms, "ST 세그먼트")
    }
    
    features['QT_Interval'] = {
        'values': qt_intervals_ms,
        'stats': calculate_statistics(qt_intervals_ms, "QT 간격")
    }
    
    features['QTc'] = {
        'values': qtc_intervals_ms,
        'stats': calculate_statistics(qtc_intervals_ms, "QTc 간격")
    }
    return features

def calculate_tinn(rr_intervals, bin_width=7.8125):
    """
    TINN 계산 함수
    
    Args:
        rr_intervals: RR 간격 배열 (밀리초 단위)
        bin_width: 히스토그램 bin의 너비 (ms)
        
    Returns:
        tinn: TINN 값 (ms)
        triangle_points: (M, N, O) 삼각형의 세 점
    """
    if np.min(rr_intervals) == np.max(rr_intervals):
        print("경고: 모든 RR 간격이 동일합니다. TINN을 0으로 설정합니다.")
        return 0, (np.min(rr_intervals), np.min(rr_intervals), np.min(rr_intervals))
    # 히스토그램 생성
    bins = np.arange(min(rr_intervals), max(rr_intervals) + bin_width, bin_width)
    hist, bin_edges = np.histogram(rr_intervals, bins=bins)
    # 최대 빈도 지점 찾기 (N)
    max_idx = np.argmax(hist)
    n_x = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2
    n_y = hist[max_idx]
    
    # 삼각형 맞추기 (최적의 M과 O 찾기)
    best_fit = float('inf')
    m, o = 0, 0
    
    for i in range(max_idx):
        m_candidate = bin_edges[i]
        for j in range(max_idx + 1, len(bin_edges) - 1):
            o_candidate = bin_edges[j]
            
            # 삼각형 모델 생성
            triangle = np.zeros_like(hist)
            for k in range(len(hist)):
                bin_center = (bin_edges[k] + bin_edges[k + 1]) / 2
                if bin_center <= n_x:
                    # 왼쪽 부분
                    if bin_center >= m_candidate:
                        triangle[k] = n_y * (bin_center - m_candidate) / (n_x - m_candidate)
                else:
                    # 오른쪽 부분
                    if bin_center <= o_candidate:
                        triangle[k] = n_y * (o_candidate - bin_center) / (o_candidate - n_x)
            
            # 오차 계산
            error = np.sum((hist - triangle) ** 2)
            if error < best_fit:
                best_fit = error
                m, o = m_candidate, o_candidate
    
    tinn = o - m
    triangle_points = (m, n_x, o)
    
    return tinn, triangle_points



def extract_ecg_features(peaks, sampling_rate=500):
    """
    ECG 신호에서 다양한 특성(PR 간격, QRS 지속 시간, ST 세그먼트, QT 간격)을 추출하는 통합 함수
    
    Args:
        peaks: ECG 신호에서 검출된 피크 정보를 담은 딕셔너리
        sampling_rate: 샘플링 레이트 (Hz), 기본값 500Hz
        
    Returns:
        features: 추출된 모든 특성과 통계 정보를 담은 딕셔너리
    """
    features = {}
    
    # 필요한 피크 데이터 추출
    p_peaks = peaks.get('ECG_P_Peaks', [])
    q_peaks = peaks.get('ECG_Q_Peaks', [])
    r_peaks = peaks.get('ECG_R_Peaks', [])
    s_peaks = peaks.get('ECG_S_Peaks', [])
    t_peaks = peaks.get('ECG_T_Peaks', [])
    
    rr_intervals = np.diff(r_peaks) * (1000 / sampling_rate)
    mean_rr = rr_intervals  # 평균 RR 간격 (ms)
    
    # 1. PR 간격 계산
    pr_intervals_ms = []
    min_length = min(len(p_peaks), len(r_peaks))
    for i in range(min_length):
        pr_interval = r_peaks[i] - p_peaks[i]
        pr_interval = samples_to_ms(pr_interval, sampling_rate)
        pr_intervals_ms.append(pr_interval)
    
    # 2. QRS 지속 시간 계산
    qrs_durations_ms = []
    min_length = min(len(q_peaks), len(s_peaks))
    for i in range(min_length):
        qrs_duration = s_peaks[i] - q_peaks[i]
        qrs_duration = samples_to_ms(qrs_duration, sampling_rate)
        qrs_durations_ms.append(qrs_duration)
    
    # 3. ST 세그먼트 계산
    st_segments_ms = []
    min_length = min(len(s_peaks), len(t_peaks))
    for i in range(min_length):
        st_segment = t_peaks[i] - s_peaks[i]
        st_segment = samples_to_ms(st_segment, sampling_rate)
        st_segments_ms.append(st_segment)
    
    # 4. QT 간격 계산
    qt_intervals_ms = []
    min_length = min(len(q_peaks), len(t_peaks))
    for i in range(min_length):
        qt_interval = t_peaks[i] - q_peaks[i]
        qt_interval = samples_to_ms(qt_interval, sampling_rate)
        qt_intervals_ms.append(qt_interval)
        
    # 5. QTC 간격 계산
    qtc_intervals_ms = []
    min_length = len(qt_intervals_ms)
    mean_rr_value = np.mean(mean_rr)
    for i in range(min_length):
        # 마지막 간격에는 mean_rr의 평균값 사용
        qtc_interval = qt_intervals_ms[i] / np.sqrt(mean_rr_value)
        qtc_intervals_ms.append(qtc_interval)
    
    # 각 특성에 대한 통계 계산 및 저장
    features['PR_Interval'] = {
        'values': pr_intervals_ms,
        'stats': calculate_statistics(pr_intervals_ms, "PR 간격")
    }
    
    features['QRS_Duration'] = {
        'values': qrs_durations_ms,
        'stats': calculate_statistics(qrs_durations_ms, "QRS 지속 시간")
    }
    
    features['ST_Segments'] = {
        'values': st_segments_ms,
        'stats': calculate_statistics(st_segments_ms, "ST 세그먼트")
    }
    
    features['QT_Interval'] = {
        'values': qt_intervals_ms,
        'stats': calculate_statistics(qt_intervals_ms, "QT 간격")
    }
    
    features['QTc'] = {
        'values': qtc_intervals_ms,
        'stats': calculate_statistics(qtc_intervals_ms, "QTc 간격")
    }
    return features


def calculate_hrv_features(r_peaks, sampling_rate=500):
    """
    R-peak 위치를 기반으로 HRV 특성을 계산하는 함수
    
    Args:
        r_peaks: R-peak의 위치 배열 (샘플 단위)
        sampling_rate: 샘플링 레이트 (Hz)
        
    Returns:
        hrv_features: HRV 특성을 담은 딕셔너리
    """
    # RR 간격 계산 (밀리초 단위)
    rr_intervals = np.diff(r_peaks) * (1000 / sampling_rate)
    # 시간 영역 HRV 특성
    time_domain = {}
    
    # 기본 통계
    time_domain['mean_rr'] = np.mean(rr_intervals)  # 평균 RR 간격 (ms)
    time_domain['sdnn'] = np.std(rr_intervals)      # RR 간격의 표준편차 (ms)
    
    # Heart rate 계산 (60000 / RR 간격(ms))
    heart_rates = 60000 / rr_intervals
    time_domain['hr_max_min'] = np.max(heart_rates) - np.min(heart_rates)  # 최대 심박수 - 최소 심박수
    
    # SDRR: RR 간격의 표준편차 (SDNN과 동일하지만 다른 명칭으로도 사용됨)
    time_domain['sdrr'] = np.std(rr_intervals)
    

    # NN50: 연속된 RR 간격의 차이가 50ms를 초과하는 쌍의 수
    nn50 = sum(abs(np.diff(rr_intervals)) > 50)
    time_domain['nn50'] = nn50
    
    # pNN50: NN50을 총 RR 간격 수로 나눈 비율 (%)
    if len(rr_intervals) > 1:
        time_domain['pnn50'] = (nn50 / (len(rr_intervals) - 1)) * 100
    else:
        time_domain['pnn50'] = 0
        
    # NN20: 연속된 RR 간격의 차이가 20ms를 초과하는 쌍의 수
    nn20 = sum(abs(np.diff(rr_intervals)) > 20)
    time_domain['nn20'] = nn20
    
    # pNN20: NN20을 총 RR 간격 수로 나눈 비율 (%)
    if len(rr_intervals) > 1:
        time_domain['pnn20'] = (nn20 / (len(rr_intervals) - 1)) * 100
    else:
        time_domain['pnn20'] = 0
    
    # RMSSD: 연속된 RR 간격 차이의 제곱 평균의 제곱근
    if len(rr_intervals) > 1:
        time_domain['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    else:
        time_domain['rmssd'] = 0
    
    # TINN: NN 간격(정상 RR 간격) 히스토그램의 기하학적 특성을 측정
    # https://ekja.org/upload/media/kja-22324-supplementary-material.pdf
    tinn = calculate_tinn(rr_intervals)
    time_domain['tinn'] = tinn
    
    # 비선형 HRV 특성 (Poincaré plot)
    nonlinear = {}
    
    if len(rr_intervals) > 1:
        # SD1, SD2 계산 (Poincaré plot)
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        
        sd1 = np.std(np.subtract(rr_n, rr_n1) / np.sqrt(2))
        sd2 = np.std(np.add(rr_n, rr_n1) / np.sqrt(2))
        
        nonlinear['sd1'] = sd1
        nonlinear['sd2'] = sd2
        nonlinear['sd1_sd2_ratio'] = sd2 / sd1 if sd1 > 0 else 0
    else:
        nonlinear['sd1'] = None
        nonlinear['sd2'] = None
        nonlinear['sd1_sd2_ratio'] = None
    
    # 모든 HRV 특성을 하나의 딕셔너리로 통합
    hrv_features = {
        'rr_intervals': rr_intervals.tolist(),
        'time_domain': time_domain,
        'nonlinear': nonlinear
    }
    
    return hrv_features