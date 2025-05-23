# ECG R-Peak Detection Project

이 프로젝트는 ECG(심전도) 신호에서 R-peak를 검출하는 다양한 알고리즘을 구현하고 앙상블 방법을 통해 정확도를 향상시키는 것을 목표로 함

## 프로젝트 개요

ECG 신호 분석에서 R-peak 검출은 심박수 계산, 부정맥 진단 등 다양한 심장 질환 분석의 기초가 되는 중요한 과정
본 프로젝트는 여러 알고리즘을 조합한 앙상블 방법을 통해 더욱 정확하고 신뢰할 수 있는 R-peak 검출을 제공

## 주요 기능

### 1. 다중 알고리즘 지원
- **NeuroKit2**: 신경과학 및 생리학적 신호 처리를 위한 라이브러리
- **Pan-Tompkins**: 전통적인 QRS 복합체 검출 알고리즘
- **Pan-Tompkins++**: 개선된 Pan-Tompkins 알고리즘
- **Hamilton**: Hamilton의 QRS 검출 방법
- **Christov**: Christov의 실시간 QRS 검출 알고리즘
- **Engzee**: Engzee의 QRS 검출 방법

### 2. 신호 전처리
- 대역 통과 필터링 (0.5-40 Hz)
- 노이즈 제거
- 신호 품질 평가

### 3. 앙상블 방법
- 다중 알고리즘 결과 통합
- 투표 기반 R-peak 결정
- 가중치 적용 가능
- 허용 오차 범위 설정 가능

### 4. 시각화 및 분석
- 각 알고리즘별 결과 비교
- 앙상블 결과 시각화
- 신호 품질 평가 결과 표시

### 5. 시각화 기능
- **다중 서브플롯 구성**: 각 알고리즘별 결과를 개별 서브플롯으로 표시
- **R-peak 마커**: 
  - 파란색 원형 마커: 원본 R-peak 위치
  - 빨간색 원형 마커: 보정된 R-peak 위치
  - 보라색 별 마커: 앙상블 방법으로 결정된 최종 R-peak
- **신호 품질 표시**: 제목에 신호 품질 점수 포함
- **범례 및 그리드**: 각 플롯에 범례와 격자 표시
- **자동 레이아웃 조정**: tight_layout으로 최적화된 플롯 배치
- **저장 옵션**: plot_path 지정 시 이미지 파일로 저장 가능

## 결과

### QRS Complex 검출 결과
![QRS Complex 검출 결과](results/qrs_detection.png)

#### 성능 지표
- 정확도 (Accuracy): 0.98
- 정밀도 (Precision): 0.97
- 재현율 (Recall): 0.96
- F1 점수: 0.96

### R-peak 검출 결과
![R-peak 검출 결과](results/rpeak_detection.png)

#### 성능 지표
- 평균 절대 오차 (MAE): 0.02
- 평균 제곱 오차 (MSE): 0.0004
- 결정 계수 (R²): 0.98

### 필요한 라이브러리
```bash
pip install neurokit2
pip install matplotlib
pip install pandas
pip install numpy
pip install wfdb
```