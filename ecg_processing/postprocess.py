import numpy as np

# TODO: 수정 필요 (구조적인 문제)
def refine_qrs_mask(qrs_mask, min_qrs_width=35, max_qrs_width=60, min_gap=60):
    """
    QRS 마스크를 후처리하여 너무 좁은 QRS 영역 제거, 너무 넓은 QRS 영역 분할, 
    그리고 너무 가까운 QRS 영역 병합
    
    Args:
        qrs_mask: 원본 QRS 마스크
        min_qrs_width: 최소 QRS 너비 (샘플 수)
        max_qrs_width: 최대 QRS 너비 (샘플 수)
        min_gap: QRS 영역 간 최소 간격 (샘플 수)
    
    Returns:
        refined_mask: 정제된 QRS 마스크
    """
    # 마스크 복사
    refined_mask = qrs_mask.copy()
    
    # QRS 영역 찾기
    qrs_regions = []
    in_qrs = False
    start_idx = 0
    
    for i, val in enumerate(refined_mask):
        if val > 0.5 and not in_qrs:  # QRS 시작
            in_qrs = True
            start_idx = i
        elif val <= 0.5 and in_qrs:  # QRS 종료
            in_qrs = False
            qrs_regions.append((start_idx, i))
    
    # 마지막 QRS가 신호 끝까지 계속되는 경우
    if in_qrs:
        qrs_regions.append((start_idx, len(refined_mask)))
    
    # 너무 좁은 QRS 영역 제거
    valid_regions = []
    for start, end in qrs_regions:
        if end - start >= min_qrs_width:
            valid_regions.append((start, end))
        else:
            refined_mask[start:end] = 0  # 마스크에서 제거
    
    # 너무 넓은 QRS 영역 분할
    split_regions = []
    for start, end in valid_regions:
        width = end - start
        if width > max_qrs_width:
            # 신호 기반으로 분할점 찾기 (여기서는 간단히 중간점으로 분할)
            mid = start + width // 2
            split_regions.append((start, mid))
            split_regions.append((mid, end))
        else:
            split_regions.append((start, end))
    
    # 너무 가까운 QRS 영역 병합
    merged_regions = []
    if split_regions:
        current_start, current_end = split_regions[0]
        
        for i in range(1, len(split_regions)):
            next_start, next_end = split_regions[i]
            
            if next_start - current_end < min_gap:
                # 병합
                current_end = next_end
            else:
                merged_regions.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        # 마지막 영역 추가
        merged_regions.append((current_start, current_end))
    
    # 정제된 마스크 생성
    refined_mask = np.zeros_like(qrs_mask)
    for start, end in merged_regions:
        refined_mask[start:end] = 1
    
    return refined_mask


# ============================================================================
# postprocessing
# ============================================================================
def validate_r_peaks(ecg_signal, r_peak_locations, fs=500, min_rr_ms=200, max_rr_ms=2000):
    """
    생리학적 제약 조건을 기반으로 R-peak 위치 검증 및 보정
    
    Args:
        ecg_signal: ECG 신호
        r_peak_locations: 검출된 R-peak 위치 목록
        fs: 샘플링 주파수 (Hz)
        min_rr_ms: 최소 R-R 간격 (밀리초)
        max_rr_ms: 최대 R-R 간격 (밀리초)
    
    Returns:
        validated_peaks: 검증된 R-peak 위치 목록
    """
    if len(r_peak_locations) <= 1:
        return r_peak_locations
    
    # 샘플 단위로 변환
    min_rr_samples = int(min_rr_ms * fs / 1000)
    max_rr_samples = int(max_rr_ms * fs / 1000)
    
    # R-peak 위치 정렬
    r_peak_locations = sorted(r_peak_locations)
    
    # 검증된 R-peak 목록 초기화
    validated_peaks = [r_peak_locations[0]]  # 첫 번째 피크는 유지
    
    for i in range(1, len(r_peak_locations)):
        current_peak = r_peak_locations[i]
        last_valid_peak = validated_peaks[-1]
        rr_interval = current_peak - last_valid_peak
        
        # 최소 R-R 간격 검사
        if rr_interval < min_rr_samples:
            # 두 피크 중 더 높은 진폭을 가진 피크 선택
            if current_peak < len(ecg_signal) and last_valid_peak < len(ecg_signal):
                if abs(ecg_signal[current_peak]) > abs(ecg_signal[last_valid_peak]):
                    # 현재 피크가 더 강함 - 이전 피크 대체
                    validated_peaks[-1] = current_peak
            continue
        
        # 최대 R-R 간격 검사 (선택적)
        if rr_interval > max_rr_samples:
            # 놓친 R-peak가 있을 수 있음 - 중간에 피크 찾기 시도
            mid_point = (last_valid_peak + current_peak) // 2
            window_size = min(rr_interval // 4, 100)  # 검색 윈도우 크기
            
            # 중간 지점 주변에서 피크 찾기
            search_start = max(0, mid_point - window_size)
            search_end = min(len(ecg_signal), mid_point + window_size)
            
            if search_end > search_start:
                # 검색 윈도우 내에서 가장 큰 피크 찾기
                window_signal = np.abs(ecg_signal[search_start:search_end])
                max_idx = np.argmax(window_signal)
                potential_peak = search_start + max_idx
                
                # 충분히 큰 피크인지 확인
                if window_signal[max_idx] > 0.5 * abs(ecg_signal[last_valid_peak]):
                    validated_peaks.append(potential_peak)
        
        # 정상 R-R 간격 - 피크 추가
        validated_peaks.append(current_peak)
    
    return validated_peaks

def refine_r_peak_locations(ecg_signal, r_peak_locations, window_size=10):
    """
    R-peak 위치를 주변 윈도우에서 실제 최대값으로 정밀화
    
    Args:
        ecg_signal: ECG 신호
        r_peak_locations: 초기 R-peak 위치 목록
        window_size: 검색 윈도우 크기 (샘플 수)
    
    Returns:
        refined_peaks: 정밀화된 R-peak 위치 목록
    """
    refined_peaks = []
    
    for peak in r_peak_locations:
        # 검색 윈도우 범위 설정
        window_start = max(0, peak - window_size)
        window_end = min(len(ecg_signal), peak + window_size + 1)
        
        # 윈도우 내에서 최대값 위치 찾기
        window = ecg_signal[window_start:window_end]
        
        # 절대값 기준으로 최대값 찾기 (양수/음수 피크 모두 고려)
        local_max_idx = np.argmax(window)
        
        # 전체 신호에서의 위치로 변환
        refined_peak = window_start + local_max_idx
        refined_peaks.append(refined_peak)
    
    return refined_peaks