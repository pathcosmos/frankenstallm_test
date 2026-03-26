"""
LLM-as-Judge — Ollama API를 통한 gemma3:12b 기반 응답 채점

Track 2 (Ko-Bench), Track 3 (한국어 심화), Track 7 (쌍대비교) 에서 사용
"""

import json
import re
import time

import requests

from . import config


def _call_judge(prompt: str, timeout: int | None = None) -> str:
    """
    Ollama /api/generate로 judge 모델 호출.
    stream=false로 전체 응답을 한 번에 수신.
    """
    if timeout is None:
        timeout = config.JUDGE_TIMEOUT

    resp = requests.post(
        config.OLLAMA_API_GENERATE,
        json={
            "model": config.JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": config.JUDGE_SAMPLING,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _extract_json(text: str) -> dict:
    """응답에서 JSON 추출 — 마크다운 블록, 전후 텍스트 처리"""
    # 마크다운 코드 블록 안의 JSON
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # 중괄호 범위 추출
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def score_response(
    prompt: str,
    response: str,
    category: str,
    criteria: str = "",
    max_retries: int = 3,
) -> dict:
    """
    단일 응답 채점 (1-10)

    Returns:
        {"score": int, "reasoning": str, "error": str|None}
    """
    judge_prompt = f"""당신은 한국어 AI 모델의 응답을 채점하는 전문 평가자입니다.
주어진 프롬프트에 대한 AI의 응답을 1-10점으로 평가하세요.

평가 기준:
- 정확성: 사실 관계가 맞는가
- 관련성: 질문에 적절히 답변했는가
- 완성도: 충분히 자세하고 완전한가
- 한국어 품질: 자연스럽고 정확한 한국어인가
- 유용성: 실제로 도움이 되는 답변인가
{"" if not criteria else f"""
추가 평가 기준 ({category}):
{criteria}
"""}
[프롬프트]
{prompt}

[AI 응답]
{response}

위 응답을 1-10점으로 평가하세요.
반드시 아래 JSON 형식으로만 응답하세요:
{{"score": <1-10>, "reasoning": "<평가 이유 1-2문장>"}}"""

    text = ""
    for attempt in range(max_retries):
        try:
            text = _call_judge(judge_prompt)
            result = _extract_json(text)
            return {
                "score": int(result["score"]),
                "reasoning": result.get("reasoning", ""),
                "error": None,
            }
        except json.JSONDecodeError:
            # JSON 파싱 실패 — 숫자만이라도 추출 시도
            nums = re.findall(r"\b(\d{1,2})\b", text)
            for n in nums:
                score = int(n)
                if 1 <= score <= 10:
                    return {"score": score, "reasoning": text[:200], "error": None}
        except requests.Timeout:
            print(f"    ⏳ judge 타임아웃 — 재시도 {attempt + 1}/{max_retries}")
        except Exception as e:
            if attempt == max_retries - 1:
                return {"score": 0, "reasoning": "", "error": str(e)}
            time.sleep(3)

    return {"score": 0, "reasoning": "", "error": "채점 실패 (최대 재시도 초과)"}


def score_pairwise(
    prompt: str,
    response_a: str,
    response_b: str,
    max_retries: int = 3,
) -> dict:
    """
    쌍대비교 채점 — 두 응답 중 더 나은 것을 선택

    Returns:
        {"winner": "A"|"B"|"TIE", "reasoning": str, "error": str|None}
    """
    judge_prompt = f"""당신은 한국어 AI 모델의 응답을 비교 평가하는 전문 평가자입니다.
동일한 프롬프트에 대한 두 AI의 응답을 비교하여 더 나은 응답을 선택하세요.

평가 기준: 정확성, 완성도, 한국어 품질, 유용성

[프롬프트]
{prompt}

[응답 A]
{response_a}

[응답 B]
{response_b}

어떤 응답이 더 나은지 평가하세요.
반드시 아래 JSON 형식으로만 응답하세요:
{{"winner": "A" 또는 "B" 또는 "tie", "reasoning": "<비교 이유 1-2문장>"}}"""

    for attempt in range(max_retries):
        try:
            text = _call_judge(judge_prompt)
            result = _extract_json(text)
            winner = result["winner"].upper()
            if winner not in ("A", "B", "TIE"):
                winner = "TIE"
            return {
                "winner": winner,
                "reasoning": result.get("reasoning", ""),
                "error": None,
            }
        except requests.Timeout:
            print(f"    ⏳ judge 타임아웃 — 재시도 {attempt + 1}/{max_retries}")
        except Exception as e:
            if attempt == max_retries - 1:
                return {"winner": "TIE", "reasoning": "", "error": str(e)}
            time.sleep(3)

    return {"winner": "TIE", "reasoning": "", "error": "채점 실패"}


def score_with_criteria(
    prompt: str,
    response: str,
    criteria: dict[str, str],
    max_retries: int = 3,
) -> dict:
    """
    다면적 채점 — 여러 기준별 개별 점수

    Args:
        criteria: {"기준명": "설명", ...}

    Returns:
        {"scores": {"기준명": int, ...}, "reasoning": str, "error": str|None}
    """
    criteria_text = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    score_format = ", ".join(f'"{k}": <1-10>' for k in criteria)

    judge_prompt = f"""당신은 한국어 AI 모델 응답 평가자입니다.
아래 기준 각각에 대해 1-10점으로 평가하세요.

평가 기준:
{criteria_text}

[프롬프트]
{prompt}

[AI 응답]
{response}

각 기준별로 평가하세요.
반드시 아래 JSON 형식으로만 응답하세요:
{{"scores": {{{score_format}}}, "reasoning": "<종합 평가 1-2문장>"}}"""

    for attempt in range(max_retries):
        try:
            text = _call_judge(judge_prompt)
            result = _extract_json(text)
            scores = {k: int(v) for k, v in result["scores"].items()}
            return {
                "scores": scores,
                "reasoning": result.get("reasoning", ""),
                "error": None,
            }
        except requests.Timeout:
            print(f"    ⏳ judge 타임아웃 — 재시도 {attempt + 1}/{max_retries}")
        except Exception as e:
            if attempt == max_retries - 1:
                return {"scores": {}, "reasoning": "", "error": str(e)}
            time.sleep(3)

    return {"scores": {}, "reasoning": "", "error": "채점 실패"}
