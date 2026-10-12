# Stock Trading AI — 장별 더미 데이터 실습

책의 00장 빠른 시작부터 27장까지 실행 예제를 제공하며, 28장은 학습 경로 안내입니다. 장별 실행 파일은 총 33개입니다. 한국어·영어·일본어판의 공통 개념에 대응하며 안내는 한국어로 작성했습니다.

모든 가격·거래량·수익률·주문 이벤트는 코드 안에서 만들거나 고정한 **더미 데이터**입니다. 테스트와 학습을 위한 예제이며, 계정·API 키·브로커·시장 데이터 다운로드가 필요하지 않습니다. 설치에는 인터넷이 필요하고, 설치 후 실습은 로컬에서 실행합니다. 수익률 출력은 가상 실험 결과입니다.

## 1. 바로 실행

터미널에서 이 README가 있는 `examples` 폴더로 이동합니다. **Python 3.12**로 검증했습니다.

```sh
# 라이브러리 설치 없이 첫 예제 실행
python3 run.py 00
# 기대 출력: 1000.0 0.2  (최종 현금 / 최대 낙폭)

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py core
```

Windows PowerShell에서는 `python3` 대신 `py -3.12`, 가상환경 활성화는 `.venv\Scripts\Activate.ps1`을 사용합니다. 활성화 후 명령은 동일합니다. 활성화가 제한되면 `.venv\Scripts\python.exe run.py 00`처럼 직접 실행할 수 있습니다.

## 2. 장 선택과 전체 실행

```sh
python run.py --list      # 장별 파일 목록
python run.py 03          # 3장의 회계·위험지표·DSR 예제 모두 실행
python ch21_regime_zscore.py  # 파일 하나만 직접 실행

# SAC(14·23장)와 MLflow(27장)까지 포함
python -m pip install -r requirements-full.txt
python run.py all         # 성공하면 PASS 33/33
python -m unittest discover -s tests -v
python verify.py          # 외부 연결을 차단하고 전체 예제 + 테스트 검증
```

`core`는 Torch/SAC와 MLflow가 필요한 14·23·27장을 제외합니다. 모든 명령은 같은 가상환경의 Python으로 실행하세요. `all`은 누락된 의존성을 성공으로 처리하지 않고 실패 목록과 종료 코드 1을 반환합니다. 각 파일은 최대 240초까지 실행하며, 오류가 있어도 나머지 파일을 확인합니다.

SAC는 CPU에서 작은 신경망을 128스텝만 학습합니다. 처음 설치할 때 Torch 용량이 크므로 기초 실습부터 시작하려면 `requirements.txt`만 설치하세요. 난수 시드를 고정했지만 라이브러리·운영체제에 따라 마지막 소수점이나 모델 결과는 달라질 수 있습니다.

## 3. 책과 소스의 대응

본문의 독립 실행 블록과 필요한 선행 정의를 모았습니다. 4·21·22장은 설명·설계 발췌를 더미 데이터로 실행할 수 있게 보완했습니다. 28장은 코드 알고리즘 대신 파일 목록을 출력합니다. 12장은 중복 함수 정의를 정리하고, 0 이하 반감기·유효하지 않은 진입 임계값·상관·거래량을 거절하도록 보완했습니다. 생략된 대규모 서비스 전체를 복원한 패키지는 아니며, 각 장의 계산과 흐름을 짧게 확인하는 실습입니다.

`manifest.json`에 각 실행 파일과 원고 파일의 대응 및 보완 여부가 있습니다. `tiny_env.py`는 14·23장이 사용하는 13장의 환경이고, `monitor_example.py`는 18장 API가 사용하는 모니터 함수입니다. `hmm_demo.py`는 4·22장의 더미 데이터와 순방향 필터를 공유합니다.

| 장 | 예제 파일 | 확인할 내용 |
|---|---|---|
| 00 | [ch00_practice.py](ch00_practice.py) | 보유 수량·현금·최대 낙폭: 1000.0 / 0.2 |
| 01 | [ch01_pipeline.py](ch01_pipeline.py) | 작은 가격 파이프라인과 비용 차감 |
| 02 | [ch02_features.py](ch02_features.py) | 4개 피처, train/OOS 분리: (28, 4) / (12, 4) |
| 03 | [ch03_practice.py](ch03_practice.py)<br>[ch03_metrics.py](ch03_metrics.py)<br>[ch03_dsr.py](ch03_dsr.py) | 회계·MDD·DSR: MDD 0.2와 손익 정합성 |
| 04 | [ch04_hmm.py](ch04_hmm.py) | HMM 훈련·상태 분산 정렬·확률 합 1 |
| 05 | [ch05_forward.py](ch05_forward.py) | 미래 관측 없는 순방향 상태 확률 |
| 06 | [ch06_kalman.py](ch06_kalman.py) | 칼만 필터로 잡음이 있는 가격 평활 |
| 07 | [ch07_practice.py](ch07_practice.py) | 20차원 더미 관측의 공동 HMM |
| 08 | [ch08_ensemble.py](ch08_ensemble.py) | 두 계층 확률 결합과 위험 노출 |
| 09 | [ch09_kelly.py](ch09_kelly.py) | Kelly 비중과 상한: 0.05 |
| 10 | [ch10_optuna.py](ch10_optuna.py) | Optuna 훈련/검증/holdout 분리 |
| 11 | [ch11_practice.py](ch11_practice.py) | 더미 페어 공적분·상관·반감기 |
| 12 | [ch12_practice.py](ch12_practice.py) | 진입 조건별 허용/거절 이유 |
| 13 | [ch13_env.py](ch13_env.py) | 11차원 관측·왕복 비용·에피소드 종료 |
| 14 | [ch14_sac.py](ch14_sac.py) | SAC 작은 학습과 70스텝 평가 |
| 15 | [ch15_optuna.py](ch15_optuna.py) | Optuna 탐색 후 미사용 holdout 평가 |
| 16 | [ch16_metrics.py](ch16_metrics.py) | MDD·Sharpe·VaR·CVaR·기여도 |
| 17 | [ch17_order_events.py](ch17_order_events.py) | 부분체결·중복 체결 방지·취소 |
| 18 | [ch18_monitor.py](ch18_monitor.py)<br>[ch18_lab.py](ch18_lab.py) | 모니터 보고서와 HTTP /health·/report |
| 19 | [ch19_cointegration.py](ch19_cointegration.py) | OLS 헤지비율과 공적분 검정 |
| 20 | [ch20_screening.py](ch20_screening.py)<br>[ch20_regime_events.py](ch20_regime_events.py) | 페어 스크리닝과 레짐 이벤트 수 |
| 21 | [ch21_regime_zscore.py](ch21_regime_zscore.py) | 동일 Z-score의 레짐별 진입·청산·손절 |
| 22 | [ch22_pipeline.py](ch22_pipeline.py) | 더미 가격부터 비용 차감 OOS 평가까지 |
| 23 | [ch23_sac.py](ch23_sac.py) | SAC 학습·평가 흐름 재실행 |
| 24 | [ch24_validation.py](ch24_validation.py) | 시간 역전·중복·잘못된 확률 거절 |
| 25 | [ch25_per.py](ch25_per.py) | PER 샘플링 확률과 중요도 가중치 |
| 26 | [ch26_quantile_cvar.py](ch26_quantile_cvar.py) | 수익률 하위 꼬리 CVaR |
| 27 | [ch27_mlflow.py](ch27_mlflow.py) | 로컬 MLflow 파라미터·지표·artifact 검증 |
| 28 | [ch28_guide.py](ch28_guide.py) | 추천 학습 순서와 파일 목록 |

## 4. 실습의 계산 기준

- HMM은 앞 60%의 피처로만 훈련하고 스케일러도 같은 구간으로 맞춥니다. 순방향 확률은 그 시점까지의 관측만 사용합니다. 분산 순위로 붙인 상태 이름 자체가 평균회귀의 증거는 아닙니다.
- 22장은 시점 t의 종가 정보로 비중을 결정하고 t→t+1 수익에 적용하는 단순 종가 체결 모형입니다. 비중 변화에 0.1% 비용을 차감하고 마지막 청산 비용도 반영합니다. 실제 주문 체결·호가·슬리피지는 재현하지 않습니다.
- 21장은 외부 HMM 없이 **더미 확률**을 입력하여 의사결정만 확인합니다. 전환·고변동 우세 시 신규 진입을 막고, 보유 중에는 손절부터 검사합니다.
- 17장은 메모리 안의 가상 주문 상태만 변경합니다. 18장은 FastAPI TestClient로 확인하므로 포트를 열지 않습니다.
- 27장은 임시 디렉터리의 SQLite와 artifact에 기록하고 조회를 검증한 뒤 정리합니다. MLflow 서버를 별도로 띄울 필요가 없습니다.
- 25장은 PER 계산, 26장은 CVaR 계산 실습입니다. 전체 커스텀 RL 알고리즘 구현은 포함하지 않습니다.

## 5. 검증과 수정 연습

계산 예제의 `assert`가 기대 결과를 검사하며, 출력 중심 예제는 전체 실행과 추가 테스트로 확인합니다. 추가 테스트는 장 누락, 피처·HMM의 미래 참조, RL 관측·청산 비용·종료 처리, 레짐 차단·손절, HTTP 응답, 중복 체결과 초과 체결 거절을 확인합니다.

먼저 가격이나 수수료 하나를 바꿔 출력이 어떻게 달라지는지 보세요. 기대값을 고정한 `assert`가 실패하면 입력 변경에 따른 결과를 손으로 계산한 후 기대값을 함께 수정하세요. 학습 예제는 작게 유지하고 성과가 좋아지도록 더미 데이터를 고르는 방식은 피하세요.

검증 환경과 실행 결과는 `VALIDATION.json`을 참고하세요. `requirements-lock.txt`는 검증 환경의 전체 의존성 기록이며, 일반 설치는 위의 core/full 파일을 사용합니다. 외부 라이브러리의 deprecation 경고는 환경에 따라 표시될 수 있습니다.

## 공개 저장소

[GitHub: tedpark/stock-trading-ai-examples](https://github.com/tedpark/stock-trading-ai-examples)

이 저장소는 학습용 소스·설치 안내·테스트만 포함합니다. 책 원고/PDF, 판매·배포 설정, 계정정보, 실제 거래 데이터는 포함하지 않습니다. 본문 대응 검토 내용은 [BOOK_COVERAGE.md](BOOK_COVERAGE.md)를 참고하세요.
