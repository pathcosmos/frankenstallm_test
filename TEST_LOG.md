# FRANKENSTALLM 3B 종합 평가 프레임워크 — 테스트 로그

> **문서 작성일:** 2026-03-11
> **테스트 시작일:** 2026-03-10
> **작성 목적:** FRANKENSTALLM 3B 커스텀 LLM의 7-트랙 종합 벤치마크 평가 과정 전체를 기록
> **시스템 환경:** Ubuntu Linux 6.8.0-101-generic, Python 3.12, Ollama, NVIDIA GPU (장애 발생)

---

## 목차

1. [개요](#1-개요)
2. [테스트 대상 모델 목록](#2-테스트-대상-모델-목록)
3. [평가 트랙 구성](#3-평가-트랙-구성)
4. [환경 문제 및 해결 과정](#4-환경-문제-및-해결-과정)
5. [현재 상태](#5-현재-상태)
6. [핵심 교훈](#6-핵심-교훈)
7. [파일 구조](#7-파일-구조)

---

## 1. 개요

### FRANKENSTALLM 3B란?

FRANKENSTALLM 3B는 커스텀 빌드된 3B 파라미터 규모의 한국어/영어 이중언어 대규모 언어 모델(LLM)이다. "Frankenstein"과 "STALLM"(Small Talk And Language Learning Model)의 합성어로, 여러 기술을 조합하여 만들어진 실험적 모델임을 나타낸다. ORPO(Odds Ratio Preference Optimization) 기법으로 파인튜닝되었으며, SPM(SentencePiece Model) 토크나이저를 사용한다.

### 평가 프레임워크의 목적

본 평가 프레임워크는 FRANKENSTALLM 3B의 성능을 **7개 트랙**으로 나눈 종합 벤치마크를 통해 측정한다. 단순 정확도뿐 아니라 코드 생성, 일관성, 성능 메트릭, 그리고 동급 오픈소스 모델과의 쌍대비교까지 포함하는 포괄적 평가 체계이다.

핵심 목표:
- FRANKENSTALLM 3B v2의 한국어 벤치마크 성능 정량 평가
- 동급(3B 전후) 오픈소스 모델 5종과의 직접 비교
- 양자화 수준(Q4_K_M, Q8_0, f16)별 성능 차이 분석
- v1 → v2 모델 개선 효과 검증

### 테스트 시작 시점

- **2026-03-10 13:11** — 최초 벤치마크 실행 (Track 6 성능 측정)
- **2026-03-10 15:52** — Track 6 완료
- **2026-03-11 00:29** — 환경 문제 해결 후 본격 Stage 1 시작
- **2026-03-11 02:42** — Track 1 완료
- **2026-03-11 05:52** — Track 4 완료
- **2026-03-11 10:05** — Track 5 진행 중 (체크포인트 저장)

---

## 2. 테스트 대상 모델 목록

### 전체 모델 매트릭스 (11개)

| # | 모델명 | 파일 크기 | 양자화 | 아키텍처 | Vocab 크기 | 상태 | 백엔드 | 비고 |
|---|--------|-----------|--------|----------|------------|------|--------|------|
| 1 | `frankenstallm-3b-Q4_K_M` | 1.9 GB | Q4_K_M | Custom 3B | 64,000 | **CRASH** | Ollama | v1, 토크나이저 결함 |
| 2 | `frankenstallm-3b-Q8_0` | 3.2 GB | Q8_0 | Custom 3B | 64,000 | **CRASH** | Ollama | v1, 토크나이저 결함 |
| 3 | `frankenstallm-3b-f16` | 6.0 GB | F16 (무압축) | Custom 3B | 64,000 | **CRASH** | Ollama | v1, 토크나이저 결함 |
| 4 | `frankenstallm-3b-v2-Q4_K_M` | 757 MB | Q4_K_M | Custom 3B | 64,256 | **OK** | Ollama (CPU) | v2, 토크나이저 수정 |
| 5 | `frankenstallm-3b-v2-Q8_0` | 1.2 GB | Q8_0 | Custom 3B | 64,256 | **OK** | Ollama (CPU) | v2, 토크나이저 수정 |
| 6 | `frankenstallm-3b-v2-f16` | 2.3 GB | F16 (무압축) | Custom 3B | 64,256 | **OK** | Ollama (CPU) | v2, 풀 정밀도 |
| 7 | `qwen2.5:3b` | 1.9 GB | Q4_K_M (기본) | Qwen 2.5 | 151,936 | **OK** | Ollama (CPU) | Alibaba |
| 8 | `gemma3:4b` | 3.3 GB | Q4_K_M (기본) | Gemma 3 | 262,144 | **OK** | Ollama (CPU) | Google |
| 9 | `phi4-mini` | 2.5 GB | Q4_K_M (기본) | Phi-4 | 100,352 | **OK** | Ollama (CPU) | Microsoft |
| 10 | `exaone3.5:2.4b` | 1.6 GB | Q4_K_M (기본) | EXAONE 3.5 | 102,400 | **OK** | Ollama (CPU) | LG AI Research |
| 11 | `llama3.2:3b` | 2.0 GB | Q4_K_M (기본) | Llama 3.2 | 128,256 | **OK** | Ollama (CPU) | Meta |

### v1 모델 — 치명적 결함 상세

FRANKENSTALLM 3B v1의 GGUF 파일에는 **SPM 토크나이저에 치명적 결함**이 존재한다:

```
오류 메시지: llama_vocab::byte_to_token → std::unordered_map::at → std::out_of_range
```

구체적 결함 내용:
- **byte_to_token 매핑 누락**: SPM 토크나이저의 바이트 폴백(byte fallback) 토큰이 GGUF 메타데이터에 기록되지 않음
- **줄바꿈 토큰 누락**: `\n` (newline) 토큰이 토크나이저에 정의되지 않아 첫 번째 줄바꿈 발생 시 즉시 크래시
- **`<s>` EOG 미설정**: 시작 토큰(`<s>`)이 End-of-Generation 토큰으로 표시되지 않아 무한 생성 루프 발생 가능
- **채팅 템플릿 미설정**: `{{ .Prompt }}`만 있는 raw passthrough로, 실질적인 대화 포맷팅 불가

이 결함은 **양자화 수준과 무관하게** Q4_K_M, Q8_0, f16 세 가지 모두에서 동일하게 발생한다. 이는 변환(conversion) 단계에서의 토크나이저 직렬화 오류이므로, 원본 체크포인트에서 재변환하지 않는 한 수정 불가능하다.

### v2 모델 — 수정 사항

v1 → v2에서 수정된 핵심 사항:
- Vocab 크기 64,000 → 64,256 (256개 바이트 폴백 토큰 추가)
- SPM byte_to_token 매핑 완전 수록
- 줄바꿈 토큰 정상 등록
- EOG 토큰 올바르게 설정
- 채팅 템플릿 추가

### 비교 모델 선정 기준

비교 모델 5종은 다음 기준으로 선정되었다:
- **파라미터 규모**: 2.4B ~ 4B 범위 (FRANKENSTALLM 3B와 동급)
- **다양한 출처**: Alibaba, Google, Microsoft, LG AI, Meta — 5개 주요 AI 연구 기관
- **한국어 지원**: 모두 한국어 토큰을 포함하며, 특히 `exaone3.5:2.4b`는 한국어 특화 모델
- **공개 가용성**: Ollama 레지스트리에서 바로 사용 가능

---

## 3. 평가 트랙 구성

### 7-트랙 평가 체계 개요

```
┌────────────────────────────────────────────────────────────────┐
│                    FRANKENSTALLM 3B 평가 체계                   │
├──────────┬─────────────────────────────────────────────────────┤
│ Stage 1  │ Track 1: 한국어 표준 벤치마크 (자동 채점)            │
│ (자동)   │ Track 4: 코드/수학 (자동 채점)                       │
│          │ Track 5: 일관성/강건성 (자동 채점)                    │
├──────────┼─────────────────────────────────────────────────────┤
│ Stage 2  │ Track 2: Ko-Bench (Claude 심사)                     │
│ (Claude) │ Track 3: 한국어 심층 (하이브리드 채점)               │
├──────────┼─────────────────────────────────────────────────────┤
│ Stage 3  │ Track 7: 쌍대비교 (Claude 심사, 대규모)              │
│ (대규모) │                                                     │
├──────────┼─────────────────────────────────────────────────────┤
│ 완료     │ Track 6: 성능 측정 (레이턴시, 처리량, VRAM)          │
└──────────┴─────────────────────────────────────────────────────┘
```

### Track 1: 한국어 표준 벤치마크 (Korean Standard Benchmark)

| 항목 | 내용 |
|------|------|
| **데이터셋** | KoBEST (4개 하위 과제) + KMMLU |
| **질문 수** | 모델당 130문항 |
| **채점 방식** | 자동 (정확도 기반) |
| **측정 지표** | Accuracy, F1 |

KoBEST 하위 과제:
- `kobest_boolq` — 예/아니오 질의응답 (불리언 이해력)
- `kobest_copa` — 인과관계 추론 (원인/결과 선택)
- `kobest_sentineg` — 감성 부정 인식 (부정 표현 이해)
- `kobest_hellaswag` — 문장 완성 (상식 추론)

샘플링 파라미터 (벤치마크용 — greedy decoding):
```python
BENCHMARK_SAMPLING = {
    "temperature": 0.0,    # 결정론적 출력
    "top_p": 1.0,          # 전체 확률 분포 사용
    "repeat_penalty": 1.0, # 반복 페널티 없음
    "num_predict": 256,    # 최대 256 토큰
    "num_ctx": 4096,       # 컨텍스트 길이 4096
}
```

### Track 2: Ko-Bench

| 항목 | 내용 |
|------|------|
| **카테고리** | 8개 (writing, roleplay, reasoning, math, coding, extraction, stem, humanities) |
| **질문 수** | 카테고리당 10문항 = 총 80문항/모델 |
| **채점 방식** | Claude CLI(`claude -p`)를 LLM-as-Judge로 사용 |
| **측정 지표** | 1~10점 척도 (카테고리별 평균 + 전체 평균) |

### Track 3: 한국어 심층 평가 (Korean Deep)

| 항목 | 내용 |
|------|------|
| **질문 수** | 100문항/모델 |
| **채점 방식** | 하이브리드 (자동 키워드 매칭 + Claude 심사) |
| **측정 지표** | 복합 점수 |
| **특징** | 한국 문화, 역사, 사회에 대한 심층 이해도 평가 |

### Track 4: 코드/수학 (Code & Math)

| 항목 | 내용 |
|------|------|
| **영역** | Pass@1 코드 생성, SQL 쿼리, 디버깅, 수학 문제 |
| **채점 방식** | 자동 (실행 결과 + 정답 비교) |
| **측정 지표** | Pass@1 정확도, 코드 실행 성공률 |

### Track 5: 일관성/강건성 (Consistency & Robustness)

| 항목 | 내용 |
|------|------|
| **테스트 영역** | 반복 일관성, 패러프레이즈 불변성, 환각(hallucination) 검출 |
| **채점 방식** | 자동 (동일 프롬프트 반복 → 응답 유사도 계산) |
| **측정 지표** | 일관성 점수, 환각률 |

### Track 6: 성능 측정 (Performance) — 완료됨

| 항목 | 내용 |
|------|------|
| **측정 항목** | 레이턴시(latency), 처리량(throughput), VRAM 사용량 |
| **입력 길이** | 100, 500, 1000, 2000 토큰 |
| **동시 요청** | 1, 2, 4 레벨 |
| **측정 지표** | tokens/sec, TTFT(Time to First Token), VRAM(MB) |

### Track 7: 쌍대비교 (Pairwise Comparison)

| 항목 | 내용 |
|------|------|
| **비교 쌍** | 8개 모델에서 2개씩 선택 = C(8,2) = 28쌍 |
| **프롬프트** | 20개 대표 프롬프트 |
| **방향** | 양방향 (A vs B + B vs A) = 쌍당 40회 |
| **총 Claude 호출** | 28 × 20 × 2 = 약 1,120회 (v1 제외 시) |
| **채점 방식** | Claude CLI가 "승/무/패" 판정 |
| **측정 지표** | 승률, ELO 레이팅, Bradley-Terry 점수 |

---

## 4. 환경 문제 및 해결 과정

이 섹션은 평가 과정에서 발생한 모든 기술적 문제를 시간순으로 기록한다. **실제 ML 평가 인프라 운영에서 발생하는 문제의 전형적인 사례**로, 교육적 가치가 높다.

---

### Phase 1: 환경 준비 문제

#### 문제 1-1: matplotlib 미설치

리포트 생성(`eval_framework/report.py`)에서 차트를 그리기 위해 matplotlib이 필요했으나 시스템에 설치되어 있지 않았다.

```bash
$ python -c "import matplotlib"
ModuleNotFoundError: No module named 'matplotlib'
```

**해결:**
```bash
pip install --break-system-packages matplotlib
```

#### 문제 1-2: 한글 폰트 미설치

matplotlib으로 한국어 텍스트가 포함된 차트를 생성할 때, 한글 글리프가 모두 □(두부 문자)로 표시되었다.

```bash
$ fc-list | grep -i nanum
# (출력 없음)
```

**해결:**
```bash
sudo apt install -y fonts-nanum
# 설치 후 matplotlib 폰트 캐시 재생성 필요
python -c "import matplotlib.font_manager; matplotlib.font_manager._load_fontmanager(try_read_cache=False)"
```

#### 문제 1-3: PEP 668 — externally-managed-environment 제한

Ubuntu 24.04+에서 시스템 Python에 `pip install`을 직접 실행하면 PEP 668 에러가 발생한다:

```
error: externally-managed-environment
× This environment is externally managed
```

이는 시스템 패키지 관리자(apt)와의 충돌을 방지하기 위한 보호 기능이다.

**해결:** 프로젝트 내 venv를 생성하거나 `--break-system-packages` 플래그를 사용한다.

```bash
# 방법 1: venv 사용 (권장)
python3 -m venv .venv && source .venv/bin/activate

# 방법 2: 시스템 제한 우회 (긴급 시)
pip install --break-system-packages matplotlib
```

#### 문제 1-4: swappiness 값 과다

기본 `vm.swappiness=60`은 일반 데스크톱에 적합하나, ML 워크로드에서는 불필요한 스왑 발생으로 추론 성능이 저하된다.

**해결:**
```bash
# 현재 값 확인
cat /proc/sys/vm/swappiness
# 60

# ML 워크로드에 적합한 값으로 변경
sudo sysctl vm.swappiness=10

# 영구 적용 (재부팅 후에도 유지)
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
```

`swappiness=10`으로 설정하면 커널이 가능한 한 물리 메모리를 사용하고, 실제로 메모리 압박이 심할 때만 스왑을 사용하게 된다.

---

### Phase 2: Ollama 서버 문제

#### 문제 2-1: Ollama 서버 미실행

평가 스크립트 실행 시 Ollama API에 연결할 수 없었다.

```
requests.exceptions.ConnectionError: Connection refused (localhost:11434)
```

**해결:** 워치독 스크립트(`ollama_watchdog.sh`)를 통해 Ollama를 백그라운드에서 시작:

```bash
./ollama_watchdog.sh &
```

워치독은 10초마다 `curl -sf http://localhost:11434/` 로 헬스체크를 수행하고, 무응답 시 자동으로 Ollama를 재시작한다.

#### 문제 2-2: 모델 존재 확인

11개 모델이 모두 `/var/ollama/models` 경로에 정상적으로 저장되어 있는지 확인이 필요했다.

```bash
$ ollama list
NAME                            SIZE
frankenstallm-3b-Q4_K_M         1.9 GB
frankenstallm-3b-Q8_0           3.2 GB
frankenstallm-3b-f16            6.0 GB
frankenstallm-3b-v2-Q4_K_M     757 MB
frankenstallm-3b-v2-Q8_0       1.2 GB
frankenstallm-3b-v2-f16        2.3 GB
qwen2.5:3b                     1.9 GB
gemma3:4b                      3.3 GB
phi4-mini                      2.5 GB
exaone3.5:2.4b                 1.6 GB
llama3.2:3b                    2.0 GB
```

**결과:** 11개 모델 모두 확인. 스모크 테스트(간단한 프롬프트 실행)도 초기에는 통과했다.

---

### Phase 3: 첫 번째 실행 실패 — 출력 버퍼링

#### 증상

백그라운드로 실행한 평가 프로세스의 로그 파일이 0바이트를 유지했다:

```bash
$ PYTHONUNBUFFERED=0 python run_evaluation.py --tracks 1 4 5 > eval.log 2>&1 &
$ ls -la eval.log
-rw-rw-r-- 1 lanco lanco 0 Mar 10 23:52 eval.log
# 수분이 지나도 0바이트
```

프로세스는 `ps aux`로 확인하면 실행 중(Running)이지만, 로그 파일에 아무 것도 기록되지 않았다.

#### 원인

Python의 `stdout`은 기본적으로 **라인 버퍼링**(터미널 출력 시) 또는 **블록 버퍼링**(파일/파이프 리다이렉트 시)을 사용한다. 백그라운드 프로세스에서 파일로 리다이렉트할 때 블록 버퍼링이 적용되어, 버퍼(보통 8KB)가 가득 차기 전까지는 디스크에 기록되지 않는다.

#### 해결

```bash
# 방법 1: 환경변수로 버퍼링 비활성화
PYTHONUNBUFFERED=1 python run_evaluation.py --tracks 1 4 5 > eval.log 2>&1 &

# 방법 2: Python -u 플래그
python -u run_evaluation.py --tracks 1 4 5 > eval.log 2>&1 &

# 방법 3: stdbuf 유틸리티
stdbuf -oL python run_evaluation.py --tracks 1 4 5 > eval.log 2>&1 &
```

`PYTHONUNBUFFERED=1`을 설정하면 `stdout`과 `stderr` 모두 즉시 플러시되어, 실시간으로 로그를 `tail -f`로 모니터링할 수 있다.

---

### Phase 4: GPU Fullchip Reset (근본 원인)

이 문제는 이후 모든 장애의 **근본 원인(root cause)**이었다.

#### 증상

```bash
$ nvidia-smi
Unable to determine the device handle for GPU 0000:01:00.0: Unknown Error
```

정상적이라면 GPU 이름, 온도, 메모리 사용량 등이 표시되어야 하지만, 드라이버가 GPU 핸들 자체를 얻을 수 없는 상태였다.

#### dmesg 커널 로그 분석

```
[12345.678] NVRM: GPU at PCI:0000:01:00: GPU_IN_FULLCHIP_RESET
[12345.679] NVRM: GPU 0000:01:00.0: RmInitAdapter failed! (0x24:0x65)
[12345.680] NVRM: GPU 0000:01:00.0: rm_init_adapter failed, device minor number 0
[12346.123] nvidia-modeset: Flip event timeout on head 0 of /dev/nvidia0
[12347.456] NVRM: Unloading NVIDIA UNIX driver
```

#### 분석

- `GPU_IN_FULLCHIP_RESET`: GPU 하드웨어가 내부적으로 전체 칩 리셋 상태에 진입함
- 이는 **열 과부하(thermal throttling)**, **전력 초과**, 또는 **하드웨어 오류**로 인해 발생
- 이전의 무거운 추론 워크로드(11개 모델 순차 로딩/언로딩)가 GPU에 과부하를 줌
- `Flip event timeout`: 디스플레이 출력 관련 타임아웃 (서버에서도 발생 가능)
- `Unloading driver`: NVIDIA 커널 모듈이 스스로 언로드됨

#### 해결 시도

```bash
# 시도 1: 드라이버 재로드 (실패)
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
# rmmod: ERROR: Module nvidia is in use by: nvidia_modeset nvidia_uvm

# 시도 2: 강제 언로드 (실패)
sudo modprobe -r nvidia
# modprobe: FATAL: Module nvidia is in use

# 시도 3: nvidia-smi 리셋 (실패)
nvidia-smi --gpu-reset -i 0
# Unable to determine the device handle for GPU 0000:01:00.0: Unknown Error
```

**결론:** GPU fullchip reset 상태에서는 **소프트웨어 수준의 드라이버 재로드가 불가능**하다. 유일한 해결 방법은 **물리적 시스템 재부팅**이다. 그러나 진행 중인 평가를 중단하지 않기 위해, **CPU-only 모드로 전환**하여 평가를 계속하기로 결정했다.

---

### Phase 5: Ollama 반복 크래시 — Race Condition

#### 증상

Ollama가 시작된 후 수초 ~ 수십 초 내에 반복적으로 종료되었다. 로그를 확인하면 시작과 종료가 무한 반복되는 패턴이 관찰되었다:

```
[2026-03-10 22:15:01] Ollama 시작 중...
[2026-03-10 22:15:06] Ollama 시작 완료 (PID: 12345)
[2026-03-10 22:15:16] ⚠ Ollama 무응답 감지
[2026-03-10 22:15:19] Ollama 시작 중...
[2026-03-10 22:15:24] Ollama 시작 완료 (PID: 12350)
[2026-03-10 22:15:34] ⚠ Ollama 무응답 감지
# ... 무한 반복
```

#### 원인 분석 — 이중 재시작 메커니즘 충돌

**두 개의 독립적인 재시작 메커니즘**이 동시에 Ollama를 관리하려고 했다:

**메커니즘 1 — `ollama_watchdog.sh` (외부 스크립트)**
```bash
# 10초마다 체크, 무응답 시 kill + restart
while true; do
    if ! check_ollama; then
        pkill -f "ollama serve"    # ← Ollama 프로세스 종료
        sleep 3
        start_ollama               # ← 새 Ollama 시작
    fi
    sleep 10                       # ← 10초 대기
done
```

**메커니즘 2 — `runner.py._restart_ollama()` (내부 Python)**
```python
def _restart_ollama():
    subprocess.run(["pkill", "-f", "ollama serve"])   # ← SIGTERM
    time.sleep(5)
    subprocess.run(["pkill", "-9", "-f", "ollama serve"])  # ← SIGKILL
    time.sleep(3)
    subprocess.Popen(["ollama", "serve"], ...)         # ← 새 인스턴스 시작
```

#### 충돌 시나리오 (타임라인)

```
T+0s:  워치독이 Ollama 시작 (PID A)
T+3s:  Runner가 추론 요청 → Ollama 크래시 감지
T+3s:  Runner가 pkill → PID A 종료
T+8s:  Runner가 Ollama 시작 (PID B)
T+10s: 워치독 체크 → 이전 PID A가 없음 → "무응답"으로 판단
T+10s: 워치독이 pkill → PID B 종료 ← Runner가 방금 시작한 것!
T+13s: 워치독이 Ollama 시작 (PID C)
T+16s: Runner가 웜업 시도 → 아직 로딩 중 → 타임아웃 → 재시작 시도
T+16s: Runner가 pkill → PID C 종료
# → 무한 루프
```

#### 해결

워치독 스크립트를 비활성화하고, `runner.py` 내장 재시작 로직만 사용하도록 했다:

```bash
# 워치독 프로세스 종료
pkill -f "ollama_watchdog.sh"

# runner.py의 wait_for_ollama()와 _restart_ollama()만으로 관리
# → 단일 제어 지점, race condition 해소
```

**교훈:** 동일 서비스에 대해 **복수의 관리 메커니즘**이 동시에 동작하면 반드시 race condition이 발생한다. 항상 **단일 책임 원칙(Single Responsibility)**을 적용해야 한다.

---

### Phase 6: Ollama SIGABRT 크래시

#### 증상

race condition을 해결한 후에도, Ollama가 모델 추론 중 SIGABRT 시그널로 크래시했다:

```
signal: abort trap
signal arrived during cgo execution

goroutine 1 [syscall]:
runtime.cgocall(...)
    /usr/local/go/src/runtime/cgocall.go:157
```

이 오류는 Go 런타임에서 CGO(C-Go 인터페이스)를 통해 호출된 llama.cpp C++ 코드에서 `abort()`가 발생했음을 의미한다.

#### 초기 가설 — GPU 관련 초기화 실패

GPU가 fullchip reset 상태이므로, Ollama가 CUDA 초기화를 시도하다가 크래시하는 것으로 추정했다.

```bash
# GPU 비활성화하여 CPU-only 모드 강제
export CUDA_VISIBLE_DEVICES=""
ollama serve
```

#### 결과 — 여전히 크래시

`CUDA_VISIBLE_DEVICES=""`로 GPU를 완전히 비활성화했음에도 동일한 SIGABRT가 발생했다. 이는 GPU와 무관한 문제임을 확인했다.

#### 대응 — 3개 병렬 에이전트 투입

문제의 근본 원인을 찾기 위해 **3개의 병렬 조사 에이전트**를 동시에 투입했다.

---

### Phase 7: 병렬 에이전트 심층 조사

#### Agent 1 — 모델별 크래시 테스트

11개 모델을 하나씩 개별적으로 로딩하여 간단한 추론("hello"에 대한 응답)을 시도했다.

| 모델 그룹 | 테스트 대상 | 결과 |
|-----------|-------------|------|
| v1 모델 3종 | `frankenstallm-3b-Q4_K_M` | **CRASH** (SIGABRT) |
| | `frankenstallm-3b-Q8_0` | **CRASH** (SIGABRT) |
| | `frankenstallm-3b-f16` | **CRASH** (SIGABRT) |
| v2 모델 3종 | `frankenstallm-3b-v2-Q4_K_M` | **OK** |
| | `frankenstallm-3b-v2-Q8_0` | **OK** |
| | `frankenstallm-3b-v2-f16` | **OK** |
| 비교 모델 5종 | `qwen2.5:3b` | **OK** |
| | `gemma3:4b` | **OK** |
| | `phi4-mini` | **OK** |
| | `exaone3.5:2.4b` | **OK** |
| | `llama3.2:3b` | **OK** |

**핵심 발견:** 크래시는 **v1 모델에서만** 발생하며, 양자화 수준(Q4_K_M, Q8_0, f16)과는 무관하다. 이는 모델 가중치가 아닌 **메타데이터(토크나이저)** 문제임을 강하게 시사한다.

#### Agent 2 — GGUF 파일 구조 분석

Ollama의 모델 블롭(blob) 디렉토리에서 v1과 v2의 GGUF 메타데이터를 비교 분석했다.

**v1 GGUF 토크나이저 문제점:**

```
1. byte_to_token 매핑 누락
   - SPM 토크나이저는 알 수 없는 바이트를 <0x00>~<0xFF> 형태의 바이트 토큰으로 변환해야 함
   - v1에서는 이 매핑이 GGUF 메타데이터에 기록되지 않음
   - llama.cpp가 `std::unordered_map::at()`으로 조회 시 키가 없으므로 std::out_of_range 예외 발생

2. 줄바꿈 토큰 미등록
   - "\n" (0x0A)에 대응하는 토큰이 없음
   - 모델이 첫 번째 줄바꿈을 생성하려 하면 byte_to_token 실패 → 즉시 크래시

3. <s> 토큰 EOG 미설정
   - BOS(Beginning of Sequence) 토큰인 <s>가 EOG(End of Generation)로 표시되지 않음
   - 생성 종료 조건이 없어 무한 루프 가능 (크래시가 없었다면)

4. 채팅 템플릿 없음
   - template: "{{ .Prompt }}" (Ollama 기본 raw passthrough)
   - system/user/assistant 역할 구분 없이 프롬프트를 그대로 전달
```

**v2 GGUF 수정 사항:**

```
1. vocab_size: 64,000 → 64,256 (+256 바이트 폴백 토큰)
2. byte_to_token 매핑 완전 수록 (0x00~0xFF 전체)
3. 줄바꿈 토큰 정상 등록
4. <s>, </s> 토큰의 EOG 플래그 정상 설정
5. 적절한 채팅 템플릿 포함
```

#### Agent 3 — 대안 추론 엔진 조사

v1 모델을 Ollama 외의 다른 llama.cpp 기반 엔진으로 실행할 수 있는지 조사했다.

조사된 대안:
- **llama-cpp-python**: Python 바인딩으로 llama.cpp를 직접 호출
- **llama.cpp 서버**: `llama-server` 바이너리를 직접 실행
- **Ollama 다운그레이드**: 이전 버전에서 다른 토크나이저 처리 로직을 사용했을 가능성

**핵심 발견:** Ollama의 블롭(`/var/ollama/models/blobs/`)에 저장된 파일은 **순수 GGUF 파일**과 동일하다. 따라서 이 파일을 llama-cpp-python이나 llama.cpp 서버에서 직접 사용할 수 있다.

---

### Phase 8: llama-cpp-python 시도 — 동일 크래시

Agent 3의 발견을 바탕으로 llama-cpp-python으로 v1 모델 실행을 시도했다.

#### 설치

```bash
cd /home/lanco/cursor/temp_git/frankenstallm_test
python3 -m venv .venv
source .venv/bin/activate
pip install llama-cpp-python
```

#### 테스트 코드

```python
from llama_cpp import Llama

# v1 GGUF 파일 경로 (Ollama blob에서 직접 참조)
model_path = "/var/ollama/models/blobs/sha256-xxxx"  # v1 Q4_K_M
llm = Llama(model_path=model_path)
output = llm("Hello, how are you?", max_tokens=32)
```

#### 결과 — 동일한 크래시

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  _Map_base::at
```

**전체 스택 트레이스:**

```
llama_vocab::byte_to_token(uint8_t) const
  → std::unordered_map<uint8_t, llama_token>::at()
    → std::out_of_range: _Map_base::at
```

이로써 v1 GGUF의 토크나이저 결함이 **Ollama 고유 문제가 아닌 llama.cpp 라이브러리 수준의 문제**임이 확정되었다. llama.cpp를 사용하는 어떤 엔진(Ollama, llama-cpp-python, llama.cpp server, text-generation-webui 등)에서도 동일하게 크래시한다.

**결론:** v1 GGUF 파일은 원본 체크포인트(PyTorch/Safetensors)에서 올바른 토크나이저 설정으로 재변환하지 않는 한 사용 불가능하다. 원본 체크포인트가 현재 사용 가능하지 않으므로, v1은 평가에서 **완전 제외**한다.

---

### Phase 9: 최종 해결 — v1 제외 + CPU-only 모드

모든 조사 결과를 종합하여 최종 해결책을 적용했다.

#### 적용된 변경 사항

**1. config.py 수정 — v1 모델 제외 + CPU-only 설정**

```python
# v1 모델은 별도 리스트로 분리 (참조용)
FRANKENSTALLM_V1_MODELS = [
    "frankenstallm-3b-Q4_K_M",
    "frankenstallm-3b-Q8_0",
    "frankenstallm-3b-f16",
]

# 실제 평가 대상은 v2만 포함
FRANKENSTALLM_MODELS = [
    "frankenstallm-3b-v2-Q4_K_M",
    "frankenstallm-3b-v2-Q8_0",
    "frankenstallm-3b-v2-f16",
]

ALL_MODELS = FRANKENSTALLM_MODELS + COMPARISON_MODELS  # 8개
```

**2. config.py — CPU 모드 타임아웃 조정**

```python
# GPU 가용 여부에 따라 타임아웃 자동 조정
GPU_AVAILABLE = _gpu_available()
_TIMEOUT_MULTIPLIER = 1 if GPU_AVAILABLE else 2  # CPU 모드 시 2배

MODEL_TIMEOUTS = {name: 120 * _TIMEOUT_MULTIPLIER for name in ALL_MODELS}
# Q4_K_M: 240초 (CPU), Q8_0: 360초 (CPU), f16: 600초 (CPU)
```

**3. runner.py — graceful shutdown으로 변경**

```python
def _restart_ollama():
    # 변경 전: pkill -9 -f ollama (즉시 강제 종료)
    # 변경 후: SIGTERM 먼저 → 5초 대기 → 필요 시 SIGKILL
    subprocess.run(["pkill", "-f", "ollama serve"])  # SIGTERM
    time.sleep(5)
    subprocess.run(["pkill", "-9", "-f", "ollama serve"])  # SIGKILL (보험)
    time.sleep(3)
```

**4. runner.py — CPU-only Ollama 시작**

```python
env = os.environ.copy()
env["OLLAMA_MODELS"] = "/var/ollama/models"
if not config.GPU_AVAILABLE:
    env["CUDA_VISIBLE_DEVICES"] = ""  # GPU 완전 비활성화
subprocess.Popen(["ollama", "serve"], env=env, ...)
```

#### 최종 결과

위 변경 사항 적용 후 평가를 재시작한 결과:

```
✅ Track 1 완료: 8 모델 × 130 문항 = 1,040 추론 — 0 errors, 0 retries
✅ Track 4 완료: 8 모델 × code/math tasks — 0 errors, 0 retries
🔄 Track 5 진행 중: 3/8 모델 완료 — 0 errors, 0 retries
```

**수정 이후 에러 0건, 재시도 0건** — 완벽히 안정적으로 동작하고 있다.

---

## 5. 현재 상태

### 스테이지별 진행 상황

| 스테이지 | 트랙 | 상태 | 진행률 | 비고 |
|----------|------|------|--------|------|
| Stage 1 | Track 1: 한국어 표준 벤치마크 | **완료** | 100% | 8모델 × 130문항, 2026-03-11 02:42 완료 |
| Stage 1 | Track 4: 코드/수학 | **완료** | 100% | 8모델, 2026-03-11 05:52 완료 |
| Stage 1 | Track 5: 일관성/강건성 | **진행 중** | ~38% (3/8 모델) | 2026-03-11 10:05 체크포인트 |
| Stage 2 | Track 2: Ko-Bench | 대기 중 | 0% | Claude CLI 심사 필요 |
| Stage 2 | Track 3: 한국어 심층 | 대기 중 | 0% | 하이브리드 채점 (자동 + Claude) |
| 완료 | Track 6: 성능 측정 | **완료** | 100% | 2026-03-10 15:52 완료 |
| Stage 3 | Track 7: 쌍대비교 | 대기 중 | 0% | ~1,120 Claude API 호출 필요 |

### 결과 파일 현황

```
results/
├── benchmark_20260310_131110.json          # 초기 벤치마크 (참고용)
├── benchmark_summary_20260310_131110.txt   # 초기 벤치마크 요약
├── track1_20260311_024226.json             # Track 1 최종 결과
├── track1_korean_bench_20260311_024226.json # Track 1 상세
├── track1_korean_bench_checkpoint.json     # Track 1 체크포인트
├── track4_20260311_055248.json             # Track 4 최종 결과
├── track4_code_math_20260311_055248.json   # Track 4 상세
├── track4_code_math_checkpoint.json        # Track 4 체크포인트
├── track5_consistency_checkpoint.json      # Track 5 진행 중 체크포인트
├── track6_20260310_155213.json             # Track 6 최종 결과
├── track6_performance_20260310_134052.json # Track 6 초기 실행
├── track6_performance_20260310_155213.json # Track 6 완전 실행
└── track6_performance_checkpoint.json      # Track 6 체크포인트
```

### 예상 남은 시간

| 작업 | 예상 시간 | 의존성 |
|------|-----------|--------|
| Track 5 완료 (5/8 모델 남음) | ~4시간 | CPU-only 추론 |
| Track 2 (Ko-Bench) | ~3시간 | Claude CLI 심사 |
| Track 3 (한국어 심층) | ~5시간 | 하이브리드 채점 |
| Track 7 (쌍대비교) | ~12시간 | ~1,120 Claude API 호출 |
| 리포트 생성 | ~30분 | 모든 트랙 완료 |
| **합계** | **~25시간** | |

---

## 6. 핵심 교훈

### 교훈 1: GPU Fullchip Reset은 소프트웨어로 복구 불가

```
문제: nvidia-smi → "Unable to determine the device handle"
시도: rmmod nvidia, modprobe -r, nvidia-smi --gpu-reset
결과: 모두 실패
해결: 물리적 시스템 재부팅 필요 (또는 CPU 폴백)
```

**GPU가 fullchip reset 상태에 들어가면 커널 드라이버 수준의 재로드로도 복구되지 않는다.** 이는 NVIDIA GPU의 하드웨어 보호 메커니즘으로, PCIe 버스 리셋이 필요하며 이는 시스템 재부팅을 통해서만 가능하다. ML 워크로드에서 GPU 모니터링(온도, 전력)을 상시 수행하고, 위험 수준에 도달하기 전에 워크로드를 줄이는 것이 예방적 대책이다.

### 교훈 2: 복수 재시작 메커니즘은 Race Condition을 유발한다

```
문제: watchdog.sh + runner.py._restart_ollama()가 동시에 Ollama 관리
결과: 서로의 인스턴스를 kill → 무한 재시작 루프
해결: 단일 관리 지점으로 통합 (runner.py만 사용)
```

**하나의 서비스에 대해 반드시 하나의 관리 메커니즘만** 동작해야 한다. systemd 서비스 + 커스텀 워치독 + 애플리케이션 레벨 재시작이 모두 활성화되면 예측 불가능한 동작이 발생한다.

### 교훈 3: GGUF 토크나이저 결함은 모든 llama.cpp 계열을 크래시시킨다

```
문제: v1 GGUF의 SPM byte_to_token 매핑 누락
영향: Ollama, llama-cpp-python, llama.cpp server 모두 동일 크래시
원인: 모델 변환(conversion) 단계의 토크나이저 직렬화 오류
해결: 원본 체크포인트에서 재변환 (불가 시 모델 제외)
```

GGUF 파일 형식은 모델 가중치와 메타데이터(토크나이저, 채팅 템플릿 등)를 하나의 파일에 담는다. 메타데이터에 결함이 있으면 가중치가 완벽하더라도 추론이 불가능하다. **모델 변환 후 반드시 토크나이저 무결성 검증**을 수행해야 한다.

### 교훈 4: CUDA_VISIBLE_DEVICES=""는 CPU-only Ollama의 핵심

```
문제: GPU 장애 상태에서 Ollama가 CUDA 초기화 시도 → 크래시
해결: CUDA_VISIBLE_DEVICES="" 환경변수로 GPU 완전 비활성화
```

`CUDA_VISIBLE_DEVICES=""`를 설정하면 CUDA 런타임이 GPU를 아예 탐색하지 않으므로, 장애 GPU로 인한 초기화 실패를 원천 차단할 수 있다. GPU 장애가 감지되면 즉시 이 변수를 설정하는 것이 최선이다.

### 교훈 5: PYTHONUNBUFFERED=1은 백그라운드 프로세스 모니터링에 필수

```
문제: 백그라운드 Python 프로세스의 로그가 0바이트
원인: 파일 리다이렉트 시 블록 버퍼링 (8KB 단위)
해결: PYTHONUNBUFFERED=1로 즉시 플러시
```

장시간 실행되는 ML 파이프라인에서는 실시간 로그 모니터링이 디버깅의 핵심이다. `PYTHONUNBUFFERED=1`을 항상 설정하거나, Python 스크립트 내부에서 `sys.stdout.reconfigure(line_buffering=True)`를 사용하자.

### 교훈 6: v1 → v2 모델 개선이 검증한 것

| 항목 | v1 | v2 | 영향 |
|------|-----|-----|------|
| Vocab 크기 | 64,000 | 64,256 | 256개 바이트 폴백 토큰 추가 |
| byte_to_token | 누락 | 완전 수록 | llama.cpp 크래시 해결 |
| 줄바꿈 토큰 | 미등록 | 등록 | 멀티라인 출력 가능 |
| EOG 설정 | 미설정 | 설정 | 무한 생성 방지 |
| 채팅 템플릿 | 없음 | 있음 | 대화형 사용 가능 |
| 추론 가능 여부 | **불가** | **정상** | 전체 평가 가능 |

---

## 7. 파일 구조

### 프로젝트 디렉토리 구조

```
/home/lanco/cursor/temp_git/frankenstallm_test/
├── run_evaluation.py              # 메인 실행 스크립트 (7-트랙 오케스트레이터)
├── benchmark.py                   # 초기 벤치마크 스크립트 (레거시)
├── ollama_watchdog.sh             # Ollama 워치독 (비활성화됨 — race condition)
├── TEST_LOG.md                    # 본 문서
│
├── eval_framework/                # 평가 프레임워크 코어
│   ├── __init__.py               # 패키지 초기화
│   ├── config.py                 # 설정 — 모델 목록, API, 타임아웃, 샘플링
│   ├── runner.py                 # Ollama API 러너 — 재시도, 재시작, 체크포인트
│   ├── scoring.py                # 스코어카드 빌드 및 점수 계산
│   ├── judge.py                  # Claude CLI LLM-as-Judge 인터페이스
│   ├── report.py                 # HTML/Markdown 리포트 생성 (matplotlib 차트)
│   │
│   └── tracks/                   # 7개 평가 트랙 모듈
│       ├── __init__.py
│       ├── track1_korean_bench.py  # KoBEST + KMMLU (한국어 표준)
│       ├── track2_ko_bench.py      # Ko-Bench (8카테고리, Claude 심사)
│       ├── track3_korean_deep.py   # 한국어 심층 (하이브리드 채점)
│       ├── track4_code_math.py     # 코드/수학 (Pass@1, SQL)
│       ├── track5_consistency.py   # 일관성/강건성 (반복, 패러프레이즈)
│       ├── track6_performance.py   # 성능 측정 (레이턴시, 처리량, VRAM)
│       └── track7_pairwise.py      # 쌍대비교 (Claude 심사, ELO)
│
├── data/                          # 평가 데이터셋 (KoBEST, Ko-Bench 등)
│
├── results/                       # 평가 결과 (JSON)
│   ├── track1_korean_bench_*.json  # Track 1 결과
│   ├── track4_code_math_*.json     # Track 4 결과
│   ├── track5_consistency_*.json   # Track 5 체크포인트
│   ├── track6_performance_*.json   # Track 6 결과
│   └── *_checkpoint.json           # 각 트랙 체크포인트 (중간 저장)
│
├── reports/                       # 생성된 리포트 (HTML, Markdown)
│
└── .venv/                         # Python 가상환경 (llama-cpp-python 등)
```

### 핵심 파일 역할 상세

| 파일 | 크기 | 역할 | 핵심 기능 |
|------|------|------|-----------|
| `config.py` | 5.2 KB | 중앙 설정 | 모델 목록, GPU 감지, 타임아웃 자동 조정, 샘플링 파라미터 |
| `runner.py` | 11.1 KB | API 래퍼 | Ollama 헬스체크, 자동 재시작, 모델 전환, 체크포인트 저장 |
| `scoring.py` | 8.1 KB | 채점 | 스코어카드 빌드, 트랙별 가중치 적용, 종합 순위 산출 |
| `judge.py` | 7.4 KB | LLM 심사 | `claude -p` CLI 호출, 프롬프트 템플릿, 점수 파싱 |
| `report.py` | 40.5 KB | 리포트 | HTML/Markdown 리포트, matplotlib 차트, 레이더 차트 |
| `run_evaluation.py` | 4.5 KB | 오케스트레이터 | CLI 인터페이스, 트랙 순차 실행, 쿨다운 관리 |

### 실행 환경 요약

```
OS:          Ubuntu Linux 6.8.0-101-generic
Python:      3.12
Ollama:      설치됨 (CPU-only 모드로 동작)
GPU:         NVIDIA (fullchip reset 상태 — 사용 불가)
모델 저장:    /var/ollama/models
CPU 추론:    CUDA_VISIBLE_DEVICES="" 강제 설정
swappiness:  10 (ML 최적화)
```

---

## 부록: 주요 명령어 레퍼런스

### 평가 실행

```bash
# 전체 7트랙 실행
PYTHONUNBUFFERED=1 python run_evaluation.py > eval.log 2>&1 &

# 특정 트랙만 실행
PYTHONUNBUFFERED=1 python run_evaluation.py --tracks 1 4 5 > eval.log 2>&1 &

# 특정 모델만 평가
python run_evaluation.py --models frankenstallm-3b-v2-Q8_0 qwen2.5:3b

# 기존 결과로 리포트만 생성
python run_evaluation.py --report-only
```

### Ollama 관리

```bash
# CPU-only 모드로 Ollama 시작
CUDA_VISIBLE_DEVICES="" OLLAMA_MODELS=/var/ollama/models ollama serve

# 현재 로딩된 모델 확인
curl -s http://localhost:11434/api/ps | python -m json.tool

# 모델 수동 언로드
curl -X POST http://localhost:11434/api/generate -d '{"model":"qwen2.5:3b","keep_alive":0}'

# 모델 목록 확인
ollama list
```

### 디버깅

```bash
# Ollama 서버 로그 확인
tail -f /tmp/ollama_serve.log

# GPU 상태 확인
nvidia-smi

# 커널 GPU 로그 확인
dmesg | grep -i nvidia | tail -20

# 프로세스 상태 확인
ps aux | grep -E "(ollama|python)" | grep -v grep

# 결과 파일 크기 확인
ls -lh results/
```

---

*본 문서는 FRANKENSTALLM 3B 평가 과정에서 발생한 모든 기술적 사안을 기록한 운영 로그이다.*
*최종 평가 결과는 모든 7개 트랙이 완료된 후 별도 리포트로 생성될 예정이다.*
