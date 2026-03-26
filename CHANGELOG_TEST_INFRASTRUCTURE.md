# 테스트 인프라 구축 변경사항 상세

> 작성일: 2026-03-26

## 개요

프로젝트에 pytest 기반 테스트 인프라를 전면 구축. 기존에는 LLM 평가 벤치마크(7개 트랙, 560+ 케이스)만 있었고, 프레임워크 코드 자체의 소프트웨어 품질 보증 테스트는 전무했음. 이번 작업으로 **116개 테스트**를 추가하고, 버그 1건 수정, 데이터 외부화로 확장성을 개선함.

---

## 1. 버그 수정

### `eval_framework/judge.py` — `UnboundLocalError` 수정

**문제 위치**: `score_response()` 함수 (기존 line 90~101)

**증상**: `_call_judge()` 내부에서 `resp.json()`이 `json.JSONDecodeError`를 발생시키면, `text` 변수가 할당되지 않은 상태에서 `except json.JSONDecodeError` 블록의 `re.findall(r"\b(\d{1,2})\b", text)`가 `text`를 참조하여 `UnboundLocalError` 발생 가능.

**원인 분석**:
```python
# 기존 코드 (문제)
for attempt in range(max_retries):
    try:
        text = _call_judge(judge_prompt)      # ← _call_judge 내부에서 JSONDecodeError 발생 시 text 미할당
        result = _extract_json(text)
        ...
    except json.JSONDecodeError:
        nums = re.findall(r"\b(\d{1,2})\b", text)  # ← text가 정의되지 않음!
```

**수정 내용**:
```python
# 수정된 코드
text = ""  # ← 루프 전 초기화 추가
for attempt in range(max_retries):
    try:
        text = _call_judge(judge_prompt)
        ...
```

**검증 테스트**: `tests/unit/test_judge.py::TestScoreResponse::test_bug_fix_text_initialized`

---

## 2. 신규 파일 목록

### 테스트 인프라 (신규 생성)

| 파일 | 설명 |
|------|------|
| `pytest.ini` | pytest 설정 — testpaths, markers (unit/integration/slow), addopts |
| `requirements-dev.txt` | 테스트 의존성 — pytest>=8.0, pytest-cov>=5.0, pytest-mock>=3.12 |
| `tests/__init__.py` | 테스트 패키지 초기화 |
| `tests/conftest.py` | 공유 fixtures — EVAFRILL mock, Ollama API mock, 샘플 데이터, 임시 디렉토리 |
| `tests/unit/__init__.py` | 단위 테스트 패키지 |
| `tests/integration/__init__.py` | 통합 테스트 패키지 |

### 단위 테스트 파일 (6개, 108 테스트)

| 파일 | 테스트 수 | 대상 |
|------|----------|------|
| `tests/unit/test_judge.py` | 31 | `_call_judge` (3), `_extract_json` (12), `score_response` (7), `score_pairwise` (6), `score_with_criteria` (3) |
| `tests/unit/test_runner.py` | 31 | `generate` (7), `chat` (5), `switch_model` (5), `wait_for_ollama` (3), `unload_model` (2), health/utility (5), checkpoint (4) |
| `tests/unit/test_scoring.py` | 17 | `aggregate_accuracy` (4), `aggregate_judge_scores` (3), `aggregate_performance` (2), `fit_bradley_terry` (5), `build_scorecard` (2), `save_scorecard` (1) |
| `tests/unit/test_config.py` | 11 | `_gpu_available` (4), 타임아웃 계산 (5), 모델 리스트 일관성 (2) |
| `tests/unit/test_evafrill_runner.py` | 9 | `is_evafrill` (4), `_top_p_filtering` (5) |
| `tests/unit/test_data_externalization.py` | 9 | Track 2 JSON (3), Track 7 JSON (4), fallback (2) |

### 통합 테스트 파일 (3개, 8 테스트)

| 파일 | 테스트 수 | 대상 |
|------|----------|------|
| `tests/integration/test_judge_pipeline.py` | 3 | score→aggregate, pairwise→BT Elo, criteria→카테고리별 집계 |
| `tests/integration/test_model_lifecycle.py` | 3 | 순차 전환 A→B→C, 서버 재시작 복구, evafrill↔ollama 전환 |
| `tests/integration/test_track_execution.py` | 2 | Track 7 최소 실행 (2모델×2프롬프트), 체크포인트 이어하기 |

### 외부화된 데이터 파일 (신규 생성)

| 파일 | 내용 |
|------|------|
| `data/ko_bench/questions.json` | Track 2 질문 80개 (8 카테고리 × 10 질문, 2-turn 형식) |
| `data/track7_prompts.json` | Track 7 프롬프트 20개 (7 카테고리) |

---

## 3. 수정된 기존 파일

### `eval_framework/judge.py`
- **변경**: `score_response()` 함수에 `text = ""` 초기화 추가 (line 91)
- **이유**: `UnboundLocalError` 버그 수정

### `eval_framework/tracks/track2_ko_bench.py`
- **추가**: `import json`, `from pathlib import Path`
- **추가**: `_load_questions()` 함수 — `data/ko_bench/questions.json`에서 질문 로드, 파일 없으면 `None` 반환
- **변경**: `run()` 함수 내 `questions = QUESTIONS[category]` → `_questions = _load_questions() or QUESTIONS` 후 `questions = _questions[category]`
- **이유**: 테스트 데이터를 JSON으로 외부화하여 코드 수정 없이 질문 추가/변경 가능

### `eval_framework/tracks/track7_pairwise.py`
- **추가**: `import json`
- **추가**: `_load_prompts()` 함수 — `data/track7_prompts.json`에서 프롬프트 로드, 파일 없으면 `None` 반환
- **추가**: `PROMPTS = _load_prompts() or PROMPTS` (인라인 데이터와 JSON 파일 간 자동 선택)
- **이유**: 프롬프트를 JSON으로 외부화하여 확장성 확보

### `README.md`
- **추가**: "테스트 (pytest)" 섹션 — 설치, 실행, 테스트 구조, 커버리지 현황
- **변경**: "프로젝트 구조" 섹션 — `tests/`, `requirements-dev.txt`, `pytest.ini`, 외부화된 데이터 파일 반영

---

## 4. 테스트 상세 내역

### 4.1 `tests/conftest.py` — 공유 Fixtures

```
EVAFRILL 모듈 mock      — sys.modules 패치로 torch/yaml/tokenizers 없이 import 가능
mock_ollama_post        — requests.post 패치, Ollama API 응답 반환
mock_ollama_get         — requests.get 패치, health check/ps 응답
sample_generate_response — runner.generate() 9-key dict 샘플
sample_judge_json       — judge 정상 JSON 응답 문자열
sample_pairwise_json    — pairwise judge 정상 JSON 응답 문자열
sample_criteria_json    — multi-criteria judge 정상 JSON 응답 문자열
sample_comparisons      — Bradley-Terry 피팅용 비교 데이터 (3모델 6비교)
sample_accuracy_results — aggregate_accuracy용 데이터 (2모델 5건)
sample_judge_results    — aggregate_judge_scores용 데이터 (2모델 5건)
tmp_results_dir         — config.RESULTS_DIR 임시 경로 패치
mock_time_sleep         — time.sleep 패치 (테스트 속도)
```

### 4.2 `test_judge.py` — 31 테스트

**TestCallJudge** (3):
- `test_success` — Ollama API 호출 → 응답 strip 후 반환
- `test_custom_timeout` — timeout 파라미터 전달 확인
- `test_http_error` — raise_for_status()의 HTTPError 전파

**TestExtractJson** (12):
- `test_plain_json` — `'{"score": 8, "reasoning": "good"}'`
- `test_markdown_json_block` — ` ```json\n{...}\n``` `
- `test_markdown_no_lang_tag` — ` ```\n{...}\n``` `
- `test_with_preamble` — `'Here is: {"score": 9}'`
- `test_with_trailing_text` — `'{"score": 5} hope this helps'`
- `test_nested_braces` — `'{"scores": {"a": 1, "b": 2}}'`
- `test_unicode_korean` — `'{"reasoning": "정확한 답변"}'`
- `test_no_json_raises` — JSON 없는 텍스트 → JSONDecodeError
- `test_empty_string_raises` — 빈 문자열 → 예외
- `test_malformed_json_raises` — 깨진 JSON → JSONDecodeError
- `test_multiple_json_objects_outermost` — 여러 JSON → outermost 추출 시도
- `test_whitespace_in_markdown_block` — 마크다운 블록 내 공백 처리

**TestScoreResponse** (7):
- `test_valid_json_response` — 정상 JSON → score/reasoning 반환
- `test_fallback_to_number_extraction` — JSON 파싱 실패 → 숫자 fallback
- `test_timeout_then_success` — Timeout 1회 → 재시도 성공
- `test_all_retries_fail` — Exception 3회 → error 반환
- `test_connection_error_then_success` — ConnectionError → 재시도 성공
- `test_bug_fix_text_initialized` — JSONDecodeError 시 text="" 안전 처리 (버그 수정 검증)
- `test_json_fallback_no_valid_number` — 유효 숫자 없으면 "채점 실패" 반환

**TestScorePairwise** (6):
- `test_winner_a/b/tie` — 정상 winner 파싱
- `test_invalid_winner_defaults_tie` — winner="C" → TIE
- `test_timeout_then_success` — 재시도 후 성공
- `test_all_retries_fail` — 전체 실패 → TIE + error

**TestScoreWithCriteria** (3):
- `test_valid_multi_criteria` — 다면적 점수 반환
- `test_timeout_retry` — 재시도 성공
- `test_all_fail` — 전체 실패 → 빈 scores

### 4.3 `test_runner.py` — 31 테스트

**TestGenerate** (7):
- `test_success` — 나노초→초 변환, tokens_per_sec 계산 검증
- `test_timeout_then_success` — 재시도 성공
- `test_connection_error` — error dict 반환
- `test_all_retries_exhausted` — tokens_per_sec=0
- `test_custom_options` — options dict payload 전달 확인
- `test_system_prompt` — system key payload 포함 확인
- `test_evafrill_delegation` — evafrill_runner.generate 위임

**TestChat** (5):
- `test_success` — message.content 추출
- `test_retry_on_timeout` — 재시도 성공
- `test_evafrill_delegation` — 메시지→프롬프트 변환 위임
- `test_evafrill_system_extraction` — system role 분리 전달
- `test_all_retries_fail` — error 반환

**TestSwitchModel** (5):
- `test_success` — unload→warmup 순서
- `test_same_model_no_unload` — 동일 모델 전환 시 unload 생략
- `test_warmup_fail_then_restart` — warmup 실패→재시작→재시도
- `test_evafrill_to_ollama` — EVAFRILL unload + Ollama warmup
- `test_ollama_to_evafrill` — Ollama unload + EVAFRILL load

**TestWaitForOllama** (3):
- `test_immediate_success` — 즉시 응답
- `test_timeout_then_auto_restart` — 타임아웃 후 자동 재시작 성공
- `test_all_restart_attempts_fail` — 재시작 3회 실패 → False

**TestUnloadModel** (2):
- `test_unload_success` — keep_alive=0 전송
- `test_unload_error_silent` — 예외 무시 확인

**TestHealthAndUtility** (5):
- `test_health_check_success/failure` — 200/ConnectionError
- `test_get_loaded_models` — /api/ps 파싱
- `test_get_vram_usage_success` — nvidia-smi 출력 파싱
- `test_get_vram_usage_no_gpu` — FileNotFoundError → 0 반환

**TestCheckpoint** (4):
- `test_save_checkpoint` — JSON 저장 및 내용 검증
- `test_load_checkpoint_exists` — 저장→로드 round-trip
- `test_load_checkpoint_missing` — 파일 없음 → None
- `test_save_results_incremental` — 타임스탬프 포함 파일명

### 4.4 `test_scoring.py` — 17 테스트

**TestAggregateAccuracy** (4): 기본/빈 리스트/전부 정답/custom model_key
**TestAggregateJudgeScores** (3): 기본(mean/std/median/n)/카테고리별/score=0 제외
**TestAggregatePerformance** (2): 기본(mean/std/min/max)/누락 키 graceful
**TestFitBradleyTerry** (5): 명확한 승자(elo 대소), 동률(elo 근접), 단일 모델(1000), 3모델 추이, 빈 비교
**TestBuildScorecard** (2): 트랙 통합(dict/float 모두 처리), 누락 모델
**TestSaveScorecard** (1): JSON 저장 round-trip

### 4.5 `test_config.py` — 11 테스트

**TestGpuAvailable** (4): nvidia-smi 성공/FileNotFoundError/비정상 returncode/빈 출력
**TestTimeoutCalculation** (5): 전체 모델 존재 확인, Q8_0/8b/evafrill/deepseek-r1별 타임아웃
**TestModelListConsistency** (2): ALL_MODELS = 서브리스트 합, FRANKENSTALLM = V1+V2

### 4.6 `test_evafrill_runner.py` — 9 테스트

**TestIsEvafrill** (4): 정확한 이름/대문자/비evafrill/부분매칭
**TestTopPFiltering** (5): top_k only/top_p only/양쪽/1D→2D/shape 보존

### 4.7 `test_data_externalization.py` — 9 테스트

**TestTrack2QuestionsJson** (3): 파일 존재, 스키마 검증(8카테고리×10질문×turn1+turn2), _load_questions 동작
**TestTrack7PromptsJson** (4): 파일 존재, 스키마 검증(20항목×id+category+prompt), _load_prompts 동작, 7카테고리 커버리지
**TestFallbackWhenJsonMissing** (2): DATA_DIR에 파일 없을 때 None 반환 (inline fallback)

### 4.8 통합 테스트 — 8 테스트

**test_judge_pipeline.py** (3):
- `test_score_then_aggregate` — score_response 5회 → aggregate_judge_scores → summary 검증
- `test_pairwise_to_elo` — score_pairwise 6쌍 → fit_bradley_terry → 승자 elo 최고
- `test_multi_category_scoring` — score_with_criteria 3회 → 카테고리별 집계

**test_model_lifecycle.py** (3):
- `test_sequential_switch` — A→B→C unload/warmup 순서 검증
- `test_restart_recovery` — health check 실패 → 재시작 → 복구
- `test_full_cycle` — Ollama→EVAFRILL→Ollama 전환 전체 사이클

**test_track_execution.py** (2):
- `test_minimal_pairwise_to_elo` — 2모델 2프롬프트 → position bias 제거 → Elo 산출
- `test_checkpoint_save_and_resume` — 부분 저장 → 로드 → 병합 → 재저장

---

## 5. 커버리지 결과

```
Name                        Stmts   Miss  Cover
────────────────────────────────────────────────
eval_framework/judge.py        75      2    97%
eval_framework/scoring.py     141      3    98%
eval_framework/config.py       60      1    98%
eval_framework/runner.py      206     35    83%
eval_framework/evafrill_runner 113     72    36%
```

- `evafrill_runner.py`의 36%는 실제 모델 로딩/추론 코드(하드웨어 의존)가 테스트 범위 밖이므로 정상
- `runner.py`의 미커버 라인은 `_restart_ollama()` 내부의 `subprocess.Popen` 호출과 `wait_for_ollama`의 일부 분기

---

## 6. 데이터 외부화 상세

### Track 2 (`data/ko_bench/questions.json`)

**이전**: `track2_ko_bench.py`에 QUESTIONS dict 하드코딩 (80 질문)

**이후**: JSON 파일로 분리, `_load_questions()` 함수가 JSON 로드 → 실패 시 inline fallback

```json
{
  "writing": [
    {"turn1": "봄을 주제로 짧은 수필을 써주세요.", "turn2": "위 수필을 가을 버전으로 바꿔 써주세요."},
    ...
  ],
  "roleplay": [...],
  "reasoning": [...],
  "math": [...],
  "coding": [...],
  "extraction": [...],
  "stem": [...],
  "humanities": [...]
}
```

### Track 7 (`data/track7_prompts.json`)

**이전**: `track7_pairwise.py`에 PROMPTS 리스트 하드코딩 (20 프롬프트)

**이후**: JSON 파일로 분리, `_load_prompts()` 함수가 JSON 로드 → 실패 시 inline fallback

```json
[
  {"id": "kr_history_1", "category": "korean_knowledge", "prompt": "조선 시대 세종대왕의..."},
  ...
]
```

**확장 방법**: JSON 파일에 항목 추가만으로 새 테스트 케이스 반영. `config.TRACK7_NUM_PROMPTS` 값도 함께 업데이트 필요.

---

## 7. 기술적 결정사항

### EVAFRILL import 문제 해결

`evafrill_runner.py`가 모듈 로드 시 `torch`, `/home/lanco/models/EVAFRILL-Mo`의 커스텀 모듈, `yaml`, `tokenizers`, `safetensors`를 import. 테스트 환경에서는 `tests/conftest.py` 최상단에서 `sys.modules`에 MagicMock을 주입하여 해결:

```python
_EVAFRILL_MOCK_MODULES = [
    "model", "model.config", "model.transformer",
    "tokenizers", "safetensors", "safetensors.torch", "yaml",
]
for _mod in _EVAFRILL_MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
```

### torch CPU-only 설치

`_top_p_filtering` 테스트는 실제 텐서 연산이 필요. `torch` CPU-only 버전을 `requirements-dev.txt`에는 포함하지 않고, `.venv`에 별도 설치. torch가 없으면 해당 테스트는 `pytest.skip()`으로 건너뜀.

### time.sleep mock

`runner.py`와 `judge.py`의 재시도 로직이 `time.sleep()`을 호출하므로, 테스트 속도를 위해 fixtures로 mock 처리. 통합 테스트에서도 개별 테스트마다 필요한 mock을 `@patch` 데코레이터로 적용.
