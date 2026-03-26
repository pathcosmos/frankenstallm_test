# FRANKENSTALLM 3B 평가 프레임워크

한국어 특화 커스텀 LLM인 **FRANKENSTALLM 3B**를 5개 비교 모델과 함께 7개 트랙으로 종합 평가하는 프레임워크.

## 빠른 시작 (Quick Start)

```bash
# 1. 리포지토리 클론
git clone https://github.com/lanco/frankenstallm_test.git
cd frankenstallm_test

# 2. Ollama 설치 (이미 설치되어 있으면 건너뛰기)
curl -fsSL https://ollama.com/install.sh | sh

# 3. Python 패키지 설치
pip install -r requirements.txt
sudo apt install -y fonts-nanum  # 한국어 폰트 (리포트 차트용)

# 4. 비교 모델 다운로드
ollama pull qwen2.5:3b
ollama pull gemma3:4b
ollama pull phi4-mini
ollama pull exaone3.5:2.4b
ollama pull llama3.2:3b

# 5. 평가 실행
python run_evaluation.py
```

---

## 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |
| Python | 3.12+ | 3.12 |
| RAM | 16GB | 32GB |
| GPU | 없음 (CPU 가능) | NVIDIA 16GB+ VRAM |
| 디스크 | 20GB 여유 | 50GB 여유 |

---

## 환경 구성 상세

### Ollama 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Ollama 모델 저장 경로 설정 (선택)

기본 경로(`~/.ollama/models`)를 변경하려면:

```bash
export OLLAMA_MODELS=/var/ollama/models
```

> `eval_framework/config.py`에서 `os.environ.setdefault("OLLAMA_MODELS", "/var/ollama/models")`로 설정되어 있음. 다른 경로를 사용한다면 `config.py`도 수정.

### Python 패키지

```bash
pip install -r requirements.txt
```

필수 패키지: `matplotlib`, `requests`, `numpy`, `scipy`

### 한국어 폰트 (리포트 차트용)

```bash
sudo apt install -y fonts-nanum
```

### 시스템 튜닝 (선택)

대용량 모델 로딩 시 swap thrashing 방지:

```bash
sudo sysctl -w vm.swappiness=10
```

---

## 모델 설치

### 비교 모델 (5개)

```bash
ollama pull qwen2.5:3b       # Alibaba Qwen 2.5, 3.1B, Q4_K_M
ollama pull gemma3:4b         # Google Gemma 3, 4.3B, Q4_K_M
ollama pull phi4-mini          # Microsoft Phi-4 Mini, 3.8B, Q4_K_M
ollama pull exaone3.5:2.4b    # LG AI EXAONE 3.5, 2.7B, Q4_K_M
ollama pull llama3.2:3b       # Meta LLaMA 3.2, 3.2B, Q4_K_M
```

### FRANKENSTALLM v2 커스텀 모델 (3개)

FRANKENSTALLM v2 GGUF 파일을 준비한 후 Modelfile 템플릿으로 등록:

```bash
# 1. GGUF 파일을 적절한 위치에 배치
# 2. modelfiles/ 의 Modelfile에서 <PATH_TO_GGUF>를 실제 경로로 수정
# 3. Ollama에 등록

ollama create frankenstallm-3b-v2-Q4_K_M -f modelfiles/Modelfile.v2-Q4_K_M
ollama create frankenstallm-3b-v2-Q8_0   -f modelfiles/Modelfile.v2-Q8_0
ollama create frankenstallm-3b-v2-f16    -f modelfiles/Modelfile.v2-f16
```

각 모델의 양자화별 파일 크기:

| 양자화 | 파일 크기 | 파라미터 수 | 비고 |
|--------|-----------|-------------|------|
| Q4_K_M | 757 MB | 1.2B | 가장 빠름, 기본 권장 |
| Q8_0 | 1.2 GB | 1.2B | 균형 |
| F16 | 2.3 GB | 1.2B | 최고 품질, GPU 권장 |

> **v1 모델 주의사항**: v1 GGUF는 SPM 토크나이저의 `byte_to_token` 매핑 결함으로 llama.cpp 계열 엔진에서 SIGABRT 크래시 발생. **v2만 사용 가능.**

---

## GPU vs CPU 모드

### GPU 모드 (권장)

기본 동작. Ollama가 자동으로 NVIDIA GPU를 감지하여 사용.

```bash
ollama serve  # GPU 자동 감지
```

`config.py`가 `nvidia-smi`로 GPU 가용성을 확인하고 타임아웃을 자동 조정:
- GPU 모드: 기본 타임아웃 (Q4_K_M: 120초, Q8_0: 180초, F16: 300초)
- CPU 모드: 타임아웃 2배 자동 적용

### CPU 모드

GPU 없이 실행하려면:

```bash
CUDA_VISIBLE_DEVICES="" ollama serve
```

> CPU 모드는 상당히 느림. F16 모델은 응답 하나에 수 분 소요 가능.

---

## 평가 실행

### 기본 실행 (전체 7트랙)

```bash
python run_evaluation.py
```

### 특정 트랙만 실행

```bash
python run_evaluation.py --tracks 1 4 5
python run_evaluation.py --tracks 6      # Track 6만 (성능 벤치마크)
```

### 특정 모델만 실행

```bash
python run_evaluation.py --models frankenstallm-3b-v2-Q4_K_M qwen2.5:3b
```

### 기존 결과로 리포트만 생성

```bash
python run_evaluation.py --report-only
```

### 7개 평가 트랙

| 트랙 | 이름 | 설명 | LLM-as-Judge |
|------|------|------|:---:|
| 1 | Korean Bench | KoBEST 4개 태스크 (BoolQ, COPA, SentiNeg, HellaSwag) | |
| 2 | KO-Bench | 8개 카테고리 한국어 생성 품질 | O |
| 3 | Korean Deep | 심층 한국어 이해력 | O |
| 4 | Code & Math | 코딩/수학 문제 해결 | |
| 5 | Consistency | 응답 일관성 테스트 | |
| 6 | Performance | 토큰 속도, 레이턴시, 동시성 | |
| 7 | Pairwise | 모델 쌍대비교 | O |

### 3단계 분할 실행 전략

전체 실행이 오래 걸리므로 단계별 분할 권장:

```bash
# Stage 1: 자동 채점 트랙 (LLM Judge 불필요)
python run_evaluation.py --tracks 1 4 5 6

# Stage 2: LLM-as-Judge 트랙 (Claude CLI 필요)
python run_evaluation.py --tracks 2 3 7

# Stage 3: 리포트 생성
python run_evaluation.py --report-only
```

---

## 체크포인트 & 이어하기

평가가 중단되어도 자동으로 체크포인트가 저장됨 (`results/*_checkpoint.json`).

재실행 시 체크포인트를 자동 로드하여 이어서 진행:

```bash
# 중단 후 동일 명령으로 재실행하면 자동 이어하기
python run_evaluation.py --tracks 1 4 5
```

---

## Claude CLI 설정

Track 2, 3, 7은 **LLM-as-Judge**로 Claude를 사용. `claude` CLI가 PATH에 있어야 함.

```bash
# Claude CLI 설치 확인
which claude

# 테스트
claude -p "Hello"
```

> Claude CLI가 없으면 Track 2, 3, 7은 건너뛰고 나머지 트랙만 실행하면 됨.

---

## Ollama Watchdog

Ollama가 대용량 모델 로딩 중 크래시할 수 있음. 자동 재시작 스크립트:

```bash
chmod +x ollama_watchdog.sh
./ollama_watchdog.sh &
```

---

## 트러블슈팅

### Ollama GPU 크래시

GPU VRAM 부족 시 Ollama가 크래시할 수 있음:

```bash
# GPU 상태 확인
nvidia-smi

# GPU 메모리 비우기 — 로드된 모델 모두 해제
curl http://localhost:11434/api/ps  # 현재 로드된 모델 확인
```

### Ollama 반복 재시작

```bash
# Ollama 프로세스 정리 후 재시작
pkill -9 -f ollama
sleep 3
ollama serve &
```

### v1 모델 SIGABRT

v1 모델(`frankenstallm-3b-Q4_K_M` 등)은 SPM 토크나이저 결함으로 실행 불가. **v2 모델만 사용.**

### "모델을 찾을 수 없음" 오류

```bash
# 설치된 모델 목록 확인
ollama list

# Ollama 모델 저장 경로 확인
echo $OLLAMA_MODELS
```

### 한국어 차트 깨짐

```bash
sudo apt install -y fonts-nanum
# matplotlib 캐시 삭제
python -c "import matplotlib; print(matplotlib.get_cachedir())"
rm -rf ~/.cache/matplotlib
```

---

## 테스트 (pytest)

### 테스트 설치 및 실행

```bash
# 1. 개발 의존성 설치
pip install -r requirements-dev.txt

# 2. 전체 테스트 실행
pytest tests/ -v

# 3. 단위 테스트만
pytest tests/unit/ -v

# 4. 통합 테스트만
pytest tests/integration/ -v

# 5. 커버리지 리포트
pytest tests/ --cov=eval_framework --cov-report=term-missing
```

### 테스트 구조

**116개 테스트** — 모두 Ollama 서버나 GPU 없이 실행 가능 (mock 기반)

| 파일 | 테스트 수 | 대상 모듈 |
|------|----------|----------|
| `tests/unit/test_judge.py` | 31 | `_call_judge`, `_extract_json`, `score_response`, `score_pairwise`, `score_with_criteria` |
| `tests/unit/test_runner.py` | 31 | `generate`, `chat`, `switch_model`, `wait_for_ollama`, health check, checkpoint I/O |
| `tests/unit/test_scoring.py` | 17 | `aggregate_accuracy`, `aggregate_judge_scores`, `fit_bradley_terry`, `build_scorecard` |
| `tests/unit/test_config.py` | 11 | `_gpu_available`, 타임아웃 계산, 모델 리스트 일관성 |
| `tests/unit/test_evafrill_runner.py` | 9 | `is_evafrill`, `_top_p_filtering` (torch CPU) |
| `tests/unit/test_data_externalization.py` | 9 | Track 2/7 JSON 로딩, 스키마 검증, fallback |
| `tests/integration/test_judge_pipeline.py` | 3 | score → aggregate → Elo 파이프라인 |
| `tests/integration/test_model_lifecycle.py` | 3 | 모델 전환 A→B→C, 서버 재시작, evafrill↔ollama |
| `tests/integration/test_track_execution.py` | 2 | Track 7 최소 실행, 체크포인트 이어하기 |

### 커버리지 현황

| 모듈 | 커버리지 |
|------|---------|
| `judge.py` | 97% |
| `scoring.py` | 98% |
| `config.py` | 98% |
| `runner.py` | 83% |

---

## 프로젝트 구조

```
frankenstallm_test/
├── run_evaluation.py          # 메인 실행 스크립트
├── benchmark.py               # 단독 벤치마크
├── ollama_watchdog.sh         # Ollama 자동 재시작
├── requirements.txt           # Python 의존성 (런타임)
├── requirements-dev.txt       # Python 의존성 (개발/테스트)
├── pytest.ini                 # pytest 설정
├── eval_framework/            # 평가 프레임워크 코어
│   ├── config.py              # 설정 (모델, 타임아웃, 파라미터)
│   ├── runner.py              # Ollama API 실행 엔진
│   ├── judge.py               # LLM-as-Judge (Ollama gemma3:12b)
│   ├── evafrill_runner.py     # EVAFRILL-Mo-3B PyTorch 직접 추론
│   ├── scoring.py             # 스코어카드 계산 + Bradley-Terry Elo
│   ├── report.py              # HTML/Markdown 리포트 생성
│   └── tracks/                # 7개 평가 트랙
│       ├── track1_korean_bench.py
│       ├── track2_ko_bench.py
│       ├── track3_korean_deep.py
│       ├── track4_code_math.py
│       ├── track5_consistency.py
│       ├── track6_performance.py
│       └── track7_pairwise.py
├── tests/                     # pytest 테스트 스위트
│   ├── conftest.py            # 공유 fixtures (Ollama mock, 샘플 데이터)
│   ├── unit/                  # 단위 테스트 (6 파일, 108개)
│   │   ├── test_judge.py
│   │   ├── test_runner.py
│   │   ├── test_scoring.py
│   │   ├── test_config.py
│   │   ├── test_evafrill_runner.py
│   │   └── test_data_externalization.py
│   └── integration/           # 통합 테스트 (3 파일, 8개)
│       ├── test_judge_pipeline.py
│       ├── test_model_lifecycle.py
│       └── test_track_execution.py
├── data/                      # 벤치마크 데이터셋
│   ├── code_problems/
│   ├── ko_bench/
│   │   └── questions.json     # Track 2 질문 (80개, 외부화)
│   ├── korean_deep/
│   ├── math_problems/
│   └── track7_prompts.json    # Track 7 프롬프트 (20개, 외부화)
├── results/                   # 평가 결과 (체크포인트 포함)
├── reports/                   # 생성된 리포트
├── modelfiles/                # FRANKENSTALLM Modelfile 템플릿
│   ├── Modelfile.v2-Q4_K_M
│   ├── Modelfile.v2-Q8_0
│   └── Modelfile.v2-f16
├── MODEL_DETAILS.md           # 전체 11개 모델 상세 스펙
└── TEST_LOG.md                # 테스트 진행 기록
```

---

## 라이선스

Private research project.
