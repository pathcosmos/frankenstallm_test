#!/usr/bin/env python3
"""
FRANKENSTALLM vs Korean LLM Benchmark
11개 모델 비교 테스트 (frankenstallm 6종 + 비교 모델 5종)
"""

import json
import time
import requests
import sys
from datetime import datetime
from pathlib import Path

OLLAMA_API = "http://localhost:11434/api/generate"

MODELS = [
    # frankenstallm variants
    "frankenstallm-3b-Q4_K_M",
    "frankenstallm-3b-Q8_0",
    "frankenstallm-3b-f16",
    "frankenstallm-3b-v2-Q4_K_M",
    "frankenstallm-3b-v2-Q8_0",
    "frankenstallm-3b-v2-f16",
    # comparison models
    "qwen2.5:3b",
    "gemma3:4b",
    "phi4-mini",
    "exaone3.5:2.4b",
    "llama3.2:3b",
]

# 한국어 테스트 프롬프트 - 다양한 능력 평가
PROMPTS = [
    {
        "id": "intro",
        "category": "기본대화",
        "prompt": "안녕하세요, 자기소개를 해주세요.",
    },
    {
        "id": "knowledge",
        "category": "지식",
        "prompt": "대한민국의 수도와 인구, 주요 산업에 대해 설명해주세요.",
    },
    {
        "id": "reasoning",
        "category": "추론",
        "prompt": "철수는 영희보다 키가 크고, 영희는 민수보다 키가 큽니다. 세 사람 중 가장 키가 작은 사람은 누구인가요? 이유를 설명해주세요.",
    },
    {
        "id": "creative",
        "category": "창작",
        "prompt": "봄비가 내리는 서울의 풍경을 묘사하는 짧은 시를 써주세요.",
    },
    {
        "id": "summary",
        "category": "요약",
        "prompt": "인공지능이 의료 분야에서 활용되는 사례를 3가지 간단히 요약해주세요.",
    },
    {
        "id": "translation",
        "category": "번역",
        "prompt": "다음 문장을 영어로 번역해주세요: '오늘 날씨가 좋아서 공원에서 산책을 했습니다.'",
    },
    {
        "id": "code",
        "category": "코드",
        "prompt": "Python으로 피보나치 수열의 처음 10개 숫자를 출력하는 코드를 작성해주세요.",
    },
    {
        "id": "math",
        "category": "수학",
        "prompt": "어떤 수에 3을 곱하고 7을 더하면 28이 됩니다. 그 수는 무엇인가요? 풀이 과정을 보여주세요.",
    },
]


def generate(model: str, prompt: str, timeout: int = 120) -> dict:
    """Ollama API로 텍스트 생성"""
    try:
        start = time.time()
        resp = requests.post(
            OLLAMA_API,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 512, "num_ctx": 4096},
            },
            timeout=timeout,
        )
        wall_time = time.time() - start
        data = resp.json()

        return {
            "response": data.get("response", ""),
            "eval_count": data.get("eval_count", 0),
            "eval_duration_s": data.get("eval_duration", 0) / 1e9,
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_duration_s": data.get("prompt_eval_duration", 0) / 1e9,
            "total_duration_s": data.get("total_duration", 0) / 1e9,
            "wall_time_s": wall_time,
            "tokens_per_sec": (
                data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9)
                if data.get("eval_duration", 0) > 0
                else 0
            ),
            "error": None,
        }
    except Exception as e:
        return {
            "response": "",
            "eval_count": 0,
            "eval_duration_s": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration_s": 0,
            "total_duration_s": 0,
            "wall_time_s": 0,
            "tokens_per_sec": 0,
            "error": str(e),
        }


def run_benchmark():
    """전체 벤치마크 실행"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "models": MODELS,
        "prompts": PROMPTS,
        "results": [],
    }

    total = len(MODELS) * len(PROMPTS)
    current = 0

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  Model: {model}")
        print(f"{'='*60}")

        # warm up - 모델 로딩
        print(f"  [warming up...]", end="", flush=True)
        generate(model, "hello", timeout=180)
        print(" done")

        for prompt_info in PROMPTS:
            current += 1
            pid = prompt_info["id"]
            cat = prompt_info["category"]
            prompt = prompt_info["prompt"]

            print(f"  [{current}/{total}] {cat} ({pid})...", end="", flush=True)
            result = generate(model, prompt)

            entry = {
                "model": model,
                "prompt_id": pid,
                "category": cat,
                "prompt": prompt,
                **result,
            }
            results["results"].append(entry)

            tok_s = result["tokens_per_sec"]
            n_tok = result["eval_count"]
            err = result["error"]
            if err:
                print(f" ERROR: {err}")
            else:
                print(f" {n_tok} tokens, {tok_s:.1f} tok/s")

    # 결과 저장
    out_dir = Path("/home/lanco/cursor/temp_git/frankenstallm_test/results")
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / f"benchmark_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {json_path}")

    # 요약 리포트 생성
    print_summary(results, out_dir, timestamp)

    return results


def print_summary(results: dict, out_dir: Path, timestamp: str):
    """요약 리포트 출력 및 저장"""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  FRANKENSTALLM 벤치마크 요약 리포트 — {timestamp}")
    lines.append(f"{'='*80}\n")

    # 모델별 평균 성능
    model_stats = {}
    for r in results["results"]:
        m = r["model"]
        if m not in model_stats:
            model_stats[m] = {
                "tokens": [],
                "tok_s": [],
                "times": [],
                "responses": [],
            }
        if not r["error"]:
            model_stats[m]["tokens"].append(r["eval_count"])
            model_stats[m]["tok_s"].append(r["tokens_per_sec"])
            model_stats[m]["times"].append(r["total_duration_s"])
            model_stats[m]["responses"].append(len(r["response"]))

    lines.append(f"{'모델':<35} {'평균tok/s':>10} {'평균토큰':>10} {'평균시간(s)':>12} {'평균응답길이':>12}")
    lines.append("-" * 80)

    for m in MODELS:
        s = model_stats.get(m, {})
        if s.get("tok_s"):
            avg_toks = sum(s["tok_s"]) / len(s["tok_s"])
            avg_tokens = sum(s["tokens"]) / len(s["tokens"])
            avg_time = sum(s["times"]) / len(s["times"])
            avg_resp = sum(s["responses"]) / len(s["responses"])
            lines.append(f"{m:<35} {avg_toks:>10.1f} {avg_tokens:>10.0f} {avg_time:>12.2f} {avg_resp:>12.0f}")
        else:
            lines.append(f"{m:<35} {'ERROR':>10}")

    # 카테고리별 응답 비교
    lines.append(f"\n{'='*80}")
    lines.append("  카테고리별 상세 응답 (처음 150자)")
    lines.append(f"{'='*80}")

    for p in PROMPTS:
        lines.append(f"\n--- [{p['category']}] {p['prompt'][:50]}... ---")
        for r in results["results"]:
            if r["prompt_id"] == p["id"]:
                resp_preview = r["response"].replace("\n", " ")[:150]
                lines.append(f"  {r['model']:<35}: {resp_preview}")

    report = "\n".join(lines)
    print(report)

    report_path = out_dir / f"benchmark_summary_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    run_benchmark()
