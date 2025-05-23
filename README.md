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

### 4. 시각화 및 분석
- 각 알고리즘별 결과 비교
- 앙상블 결과 시각화
- 신호 품질 평가 결과 표시

## 설치 및 환경 설정

### 필요한 라이브러리

```bash
pip install neurokit2
pip install matplotlib
pip install pandas
pip install numpy
pip install wfdb
```