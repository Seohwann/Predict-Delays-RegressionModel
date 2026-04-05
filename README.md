# 스마트 창고 출고 지연 예측 AI 경진대회 README

## 1. 프로젝트 개요

이 프로젝트는 **스마트 창고 출고 지연 예측 AI 경진대회**를 위한 학습 파이프라인입니다.  
목표는 주어진 창고 운영 데이터로부터 **향후 30분 평균 출고 지연 시간(`avg_delay_minutes_next_30m`)** 을 예측하는 것입니다.

이 코드는 단순히 모델 하나를 학습하는 구조가 아니라, 아래와 같은 흐름으로 동작합니다.

1. `train.csv`, `test.csv`, `layout_info.csv`를 불러옵니다.
2. `layout_id`를 기준으로 레이아웃 정보를 병합합니다.
3. `utils.py`에서 다양한 **피처 엔지니어링(feature engineering)** 을 수행합니다.
4. `scenario_id`를 기준으로 **GroupKFold 5-fold 교차검증**을 수행합니다.
5. 각 fold마다 다음 3개의 모델을 학습합니다.
   - LightGBM
   - CatBoost
   - XGBoost
6. 각 모델의 OOF(Out-Of-Fold) 예측을 만든 뒤, **OOF MAE 기준으로 최적 앙상블 가중치**를 찾습니다.
7. 최종적으로 test 데이터에 대한 예측 결과를 `outputs/submission.csv`로 저장합니다.

즉, 이 프로젝트는  
**강한 피처 엔지니어링 + 그룹 기반 검증 + 3개 GBDT 모델 앙상블** 구조로 설계되어 있습니다.

---

## 2. 프로젝트 구조 예시

```bash
project/
├─ data/
│  ├─ train.csv
│  ├─ test.csv
│  ├─ layout_info.csv
│  └─ sample_submission.csv
├─ models/
├─ outputs/
├─ train.py
├─ utils.py
└─ README.md
```

---

## 3. 이 프로젝트의 핵심 동작 원리

### 3-1. 왜 단순 모델이 아니라 피처 엔지니어링이 중요한가?

이 문제의 타깃은 단순히 현재 상태를 보고 바로 예측할 수 있는 값이 아니라,  
**현재 주문량, 로봇 상태, 혼잡도, 설비 사용률, 시스템 지연, 시간 흐름** 등이 복합적으로 작용한 결과입니다.

예를 들어, 아래 두 상황을 생각할 수 있습니다.

- 주문량은 많지만 로봇도 충분한 경우
- 주문량은 같지만 활성 로봇 수가 적고 혼잡도가 높은 경우

원본 컬럼만 보면 두 상황을 바로 구분하기 어려울 수 있습니다.  
그래서 이 코드는 다음과 같은 파생 피처를 만듭니다.

- 주문량 대비 활성 로봇 수
- 주문량 대비 작업 인력 수
- 혼잡도 × 주문량
- 패킹 이용률 × 도크 이용률
- 이전 시점 대비 변화량
- 최근 평균 대비 현재 수준
- 시나리오 내 초반/중반/후반 위치

즉, 피처 엔지니어링은  
**“원본 데이터를 모델이 더 잘 이해할 수 있는 형태로 바꾸는 과정”** 입니다.

---

### 3-2. 왜 `GroupKFold`를 사용하는가?

이 코드의 중요한 포인트는 `scenario_id`를 기준으로 `GroupKFold`를 사용한다는 점입니다.

같은 `scenario_id`에 속한 데이터는 서로 강한 연관성을 가질 수 있습니다.  
만약 일반 KFold를 사용하면 같은 시나리오의 일부가 train, 일부가 valid에 동시에 들어갈 수 있고,  
이 경우 validation 점수가 실제보다 지나치게 좋게 나오는 **데이터 누수(leakage)** 문제가 생길 수 있습니다.

이를 방지하기 위해 이 프로젝트는 다음과 같이 그룹 기반 검증을 수행합니다.

- 같은 `scenario_id`는 같은 fold에만 속함
- train/valid가 시나리오 단위로 분리됨
- 더 현실적인 일반화 성능 평가 가능

---

### 3-3. 왜 3개 모델을 같이 쓰는가?

이 프로젝트는 아래 3개 모델을 사용합니다.

- **LightGBM**
- **CatBoost**
- **XGBoost**

세 모델 모두 트리 기반 부스팅 모델이지만,  
각 모델은 범주형 처리 방식, 분할 방식, regularization 성향이 조금씩 다릅니다.

따라서 한 모델이 잘 잡는 패턴을 다른 모델은 놓칠 수 있고,  
이들을 앙상블하면 더 안정적인 성능을 얻을 수 있습니다.

---

### 3-4. 앙상블은 어떻게 하는가?

각 fold에서 3개 모델의 validation 예측을 모아 OOF 예측을 만듭니다.  
그리고 `find_best_ensemble_weights()` 함수에서 다음과 같이 가중치를 탐색합니다.

- `w_lgb`
- `w_cat`
- `w_xgb`

단, 세 가중치의 합은 1이 되도록 제한합니다.

그 뒤 OOF MAE가 가장 낮은 가중치를 선택하여 최종 submission 예측에 사용합니다.

즉, 앙상블은 감으로 섞는 것이 아니라  
**OOF 기반 검증 점수로 최적 가중치를 찾는 방식**입니다.

---

## 4. 코드 설명

# 4-1. `train.py`

`train.py`는 전체 학습 실행 스크립트입니다.

주요 역할은 다음과 같습니다.

1. 학습 인자 파싱
2. 데이터/피처 준비
3. GroupKFold 분할
4. LightGBM / CatBoost / XGBoost 학습
5. OOF 예측 생성
6. test 예측 생성
7. 앙상블 weight 탐색
8. `submission.csv`, `oof.csv`, `meta.json` 저장

### 주요 함수

#### `parse_args()`
- `--model-dir-name` 인자를 받아 모델 저장 폴더명을 지정합니다.
- 예: `python train.py --model-dir-name exp01`

#### `find_best_ensemble_weights(y_true, oof_lgb, oof_cat, oof_xgb)`
- OOF 예측값을 입력받아 최적 가중치를 찾습니다.
- 0.05 간격 grid search로 탐색합니다.
- 최종적으로 가장 낮은 MAE를 주는 `(w_lgb, w_cat, w_xgb)`를 반환합니다.

#### `main()`
실제 전체 파이프라인이 실행되는 함수입니다.

핵심 흐름:

```python
train, test, submission = build_datasets()
feature_cols, cat_cols = get_feature_columns(train)

X = train[feature_cols].copy()
y = train[TARGET].copy()
groups = train[GROUP_COL].copy()
X_test = test[feature_cols].copy()
```

즉,
- `build_datasets()`로 피처 엔지니어링이 반영된 데이터셋을 만든 뒤
- `feature_cols`에 해당하는 컬럼만 골라
- 모델 입력으로 사용합니다.

이 말은 곧, `utils.py`에서 새로 생성한 파생 컬럼들도  
`feature_cols`에 포함되기만 하면 실제 학습에 들어간다는 뜻입니다.

---

### `train.py`의 모델 학습 구조

각 fold마다 아래 순서로 진행됩니다.

#### (1) 결측/범주형 전처리
```python
X_train, X_valid, X_test_tmp = fill_missing_for_models(...)
```

#### (2) LightGBM 학습
- `objective="mae"`
- early stopping 사용
- best iteration 기준 예측

#### (3) CatBoost 학습
- `loss_function="MAE"`
- 범주형 컬럼 인덱스를 직접 지정
- early stopping 사용

#### (4) XGBoost 학습
- `objective="reg:absoluteerror"`
- 범주형 컬럼은 정수 코드화 후 입력
- `tree_method="hist"` 사용

#### (5) OOF 및 test prediction 저장
- validation 예측은 OOF 배열에 저장
- test 예측은 fold 평균

#### (6) 모델 파일 저장
- LGBM: `*.pkl`
- CatBoost: `*.cbm`
- XGB: `*.pkl`

---

# 4-2. `utils.py`

`utils.py`는 데이터 로딩, 피처 생성, 모델 입력 변환, 저장 유틸을 담당하는 모듈입니다.

## 주요 상수

```python
TARGET = "avg_delay_minutes_next_30m"
ID_COL = "ID"
GROUP_COL = "scenario_id"
LAYOUT_KEY = "layout_id"
```

---

## 데이터 로드/병합 관련 함수

### `load_data()`
다음 파일들을 불러옵니다.

- `./data/train.csv`
- `./data/test.csv`
- `./data/layout_info.csv`
- `./data/sample_submission.csv`

### `merge_layout(train, test, layout)`
`layout_id`를 기준으로 `layout_info.csv`를 병합합니다.

레이아웃 관련 정적 정보(통로 구조, 설비 수 등)가 지연 시간 예측에 도움이 될 수 있기 때문에 사용합니다.

---

## 피처 엔지니어링 (포괄 설명)

`utils.py`의 `make_features(df)`에서 여러 피처 생성 함수를 순차적으로 적용합니다.

핵심 아이디어는 아래 5가지입니다.

- 시간/시나리오 진행도 표현: 초반/중반/후반 위치, 주기성(시간/요일) 반영
- 자원 대비 부하 표현: 주문량 대비 로봇/인력/설비 여유도 반영
- 병목/상호작용 표현: 혼잡도, 패킹, 도크, 네트워크 지연의 복합 효과 반영
- 시계열 변화 표현: lag, diff, rolling, 누적 통계로 추세와 변동성 반영
- 그룹 상대 위치 표현: 같은 `scenario_id` 내 평균/최대/최소 대비 현재 수준 반영

마지막에 타입 최적화(카테고리 변환, downcasting)로 메모리 사용량을 줄이며,
실제 학습에는 `get_feature_columns()`에서 제외되지 않은 컬럼만 사용됩니다.

---

## 피처 선택 관련 함수

### `build_datasets()`
- 데이터 로드
- layout 정보 병합
- train/test에 피처 엔지니어링 적용

최종적으로 학습 가능한 `train`, `test`, `submission`을 반환합니다.

### `get_feature_columns(train)`
학습에 사용할 feature 목록과 categorical feature 목록을 반환합니다.

제외되는 컬럼:
- `ID`
- `TARGET`
- `_row_order`
- `scenario_id`

또한 기본적으로 아래와 같은 일부 노이즈 가능성이 있는 컬럼도 제외합니다.

- `_roll5_std`
- `_roll3_std`

즉, **새로 생성된 모든 컬럼이 무조건 쓰이는 것은 아니고**,  
이 함수에서 제외되지 않은 컬럼만 실제 학습에 사용됩니다.

---

## 모델 입력 변환 함수

### `fill_missing_for_models(train_x, valid_x, test_x, cat_cols)`
범주형 컬럼의 결측 처리를 수행합니다.

### `convert_cat_for_lgb(df, cat_cols)`
LightGBM용 범주형 dtype 변환

### `convert_for_xgb(train_x, valid_x, test_x, cat_cols)`
XGBoost용 범주형 정수 코드 변환

---

## 저장 관련 함수

- `save_pickle()`
- `load_pickle()`
- `save_json()`
- `load_json()`

---

## 5. 실행 환경 설정 (Windows CMD 기준)

### 5-1. Python 버전 준비

권장:
- Python 3.10 이상

예시 확인:
```cmd
python --version
```

---

### 5-2. 가상환경(venv) 생성

```cmd
python -m venv .venv
```

---

### 5-3. 가상환경 활성화

```cmd
.venv\Scripts\activate.bat
```

활성화 후 프롬프트 앞에 `(.venv)`가 보이면 정상입니다.

---

### 5-4. 패키지 설치

현재 제공된 코드 기준으로 필요한 주요 패키지는 아래와 같습니다.

- numpy
- pandas
- scikit-learn
- lightgbm
- catboost
- xgboost

설치 명령어:

```cmd
pip install --upgrade pip
pip install numpy pandas scikit-learn lightgbm catboost xgboost
```

패키지 설치 확인:
```cmd
pip list
```

---

## 6. 데이터 준비

`data/` 폴더 아래에 다음 파일이 있어야 합니다.

```bash
data/
├─ train.csv
├─ test.csv
├─ layout_info.csv
└─ sample_submission.csv
```

파일 경로가 다르면 현재 코드 그대로는 동작하지 않습니다.  
필요하다면 `utils.py`의 `load_data()`에서 경로를 수정해야 합니다.

---

## 7. 학습 실행 방법 (CMD)

### 기본 실행

```cmd
python train.py
```

실행 결과:
- `./models/` 폴더에 fold별 모델 저장
- `./outputs/oof.csv` 저장
- `./outputs/submission.csv` 저장

---

### 실험 이름을 지정해서 실행

```cmd
python train.py --model-dir-name exp01
```

이 경우 모델은 다음과 같이 저장됩니다.

`models/exp01/`

---

## 8. 추론(inference) 방법

### 중요한 점

이 프로젝트는 **별도의 추론 스크립트가 제공되지 않습니다.**

현재 제공된 코드 기준에서 test 데이터에 대한 예측은  
별도 `inference.py`로 수행하는 것이 아니라,  
`train.py` 실행 과정 안에서 같이 수행됩니다.

즉, 아래 명령어가 사실상 **대회 제출용 추론까지 포함한 실행 명령어**입니다.

```cmd
python train.py
```

또는

```cmd
python train.py --model-dir-name exp1
```

실행이 끝나면 최종 test 예측값이 아래 파일에 저장됩니다.

`outputs/submission.csv`

이 파일이 대회 제출용 결과 파일입니다.

---

## 9. 결과 파일 설명

### `outputs/oof.csv`
학습 데이터에 대한 OOF 예측 결과가 저장됩니다.

포함 예시:
- `ID`
- `scenario_id`
- `target`
- `oof_lgb`
- `oof_cat`
- `oof_xgb`
- `oof_ens`

용도:
- fold 검증 성능 분석
- 앙상블 비교
- 에러 분석

---

### `outputs/submission.csv`
test 데이터에 대한 최종 앙상블 예측 결과입니다.

용도:
- 대회 제출 파일

---

### `models/.../meta.json`
실험 메타 정보가 저장됩니다.

포함 예시:
- target
- group_col
- n_splits
- feature_cols
- cat_cols
- ensemble_weights
- fold별 CV score
- OOF MAE

용도:
- 어떤 피처가 실제로 학습에 사용되었는지 확인
- 어떤 앙상블 가중치가 선택되었는지 확인
- 실험 재현성 관리

---

## 10. 자주 헷갈리는 포인트

### Q1. 피처 엔지니어링으로 만든 컬럼도 진짜 학습에 들어가나요?
네.  
`build_datasets()`에서 피처를 생성한 뒤, `get_feature_columns()`에서 제외되지 않은 컬럼은 모두 `X = train[feature_cols]`로 모델 입력에 들어갑니다.

즉,
- `train.columns`에 있고
- `feature_cols`에도 있으면
실제 학습에 사용됩니다.

---

### Q2. 왜 `scenario_id`는 feature에서 빼나요?
`scenario_id`는 그룹 분할 기준으로 사용되지만,  
그 자체를 feature로 쓰면 특정 시나리오를 외우는 방향으로 학습할 가능성이 있어 일반화에 불리할 수 있습니다.  
그래서 feature에서는 제거합니다.

---

### Q3. 왜 `_roll3_std`, `_roll5_std`는 제외하나요?
코드 작성 의도상 rolling std 계열은 잡음이 될 가능성이 있어 기본 제외한 것으로 보입니다.  
필요하다면 실험적으로 다시 포함해볼 수 있습니다.

---

## 11. 실행 예시 전체 절차 (Windows CMD)

```cmd
# 1) 프로젝트 폴더 이동
cd C:\path\to\project

# 2) 가상환경 생성
python -m venv .venv

# 3) 가상환경 활성화
.venv\Scripts\activate.bat

# 4) 패키지 설치
pip install -r requirements.txt

# 5) 학습 + test 추론 실행
python train.py --name exp1

# 6) 추론만 실행
python inference.py --name exp1
```

실행 후 확인:
```cmd
dir outputs
dir models\exp1
```

---

## 12. 향후 개선 아이디어

이 코드는 기본 구조가 잘 잡혀 있지만, 아래와 같은 개선 여지도 있습니다.

1. `requirements.txt` 추가
2. 별도 `inference.py` 작성
3. feature importance 저장 기능 추가
4. Optuna 기반 하이퍼파라미터 튜닝
5. pseudo labeling / stacking 실험
6. `fill_missing_for_models()`의 결측 처리 순서 점검
7. XGBoost early stopping 추가

---

## 13. 한 줄 요약

이 프로젝트는  
**스마트 창고 운영 데이터를 다양한 파생 피처로 확장하고, `scenario_id` 기반 GroupKFold 검증 아래 LightGBM / CatBoost / XGBoost를 학습한 뒤, OOF 기준 최적 가중치로 앙상블하여 최종 제출 파일을 생성하는 파이프라인**입니다.
