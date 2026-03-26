"""eval_framework/judge.py 단위 테스트"""

import json
import pytest
from unittest.mock import patch, MagicMock

import requests

from eval_framework.judge import (
    _call_judge,
    _extract_json,
    score_response,
    score_pairwise,
    score_with_criteria,
)


# ═══════════════════════════════════════════════════════════════════════════════
# _call_judge 테스트 (3 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCallJudge:
    """_call_judge: Ollama API 호출"""

    @patch("eval_framework.judge.requests.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "  답변  "}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        result = _call_judge("test prompt")
        assert result == "답변"

    @patch("eval_framework.judge.requests.post")
    def test_custom_timeout(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        _call_judge("prompt", timeout=60)
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 60

    @patch("eval_framework.judge.requests.post")
    def test_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_post.return_value = mock_resp
        with pytest.raises(requests.HTTPError):
            _call_judge("prompt")


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_json 테스트 (12 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractJson:
    """_extract_json: 다양한 형태의 텍스트에서 JSON 추출"""

    def test_plain_json(self):
        text = '{"score": 8, "reasoning": "good"}'
        result = _extract_json(text)
        assert result["score"] == 8
        assert result["reasoning"] == "good"

    def test_markdown_json_block(self):
        text = '```json\n{"score": 7, "reasoning": "ok"}\n```'
        result = _extract_json(text)
        assert result["score"] == 7

    def test_markdown_no_lang_tag(self):
        text = '```\n{"score": 6, "reasoning": "fair"}\n```'
        result = _extract_json(text)
        assert result["score"] == 6

    def test_with_preamble(self):
        text = 'Here is my evaluation: {"score": 9, "reasoning": "excellent"}'
        result = _extract_json(text)
        assert result["score"] == 9

    def test_with_trailing_text(self):
        text = '{"score": 5, "reasoning": "poor"} I hope this helps!'
        result = _extract_json(text)
        assert result["score"] == 5

    def test_nested_braces(self):
        text = '{"scores": {"정확성": 8, "유용성": 7}, "reasoning": "good"}'
        result = _extract_json(text)
        assert result["scores"]["정확성"] == 8
        assert result["scores"]["유용성"] == 7

    def test_unicode_korean(self):
        text = '{"score": 8, "reasoning": "정확한 답변입니다"}'
        result = _extract_json(text)
        assert result["reasoning"] == "정확한 답변입니다"

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("no json here at all")

    def test_empty_string_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _extract_json("")

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json('{"score": }')

    def test_multiple_json_objects_outermost(self):
        """rfind('}')로 가장 바깥 중괄호를 잡으므로 전체가 추출됨"""
        text = '{"a": 1} some text {"b": 2}'
        # find('{') → 0, rfind('}') → end → '{"a": 1} some text {"b": 2}' 전체
        # json.loads 실패 가능 — 이 동작 확인
        with pytest.raises(json.JSONDecodeError):
            _extract_json(text)

    def test_whitespace_in_markdown_block(self):
        text = '```json\n  \n  {"score": 10, "reasoning": "perfect"}\n  \n```'
        result = _extract_json(text)
        assert result["score"] == 10


# ═══════════════════════════════════════════════════════════════════════════════
# score_response 테스트 (6 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreResponse:
    """score_response: 단일 응답 채점 (1-10)"""

    @patch("eval_framework.judge._call_judge")
    def test_valid_json_response(self, mock_call):
        mock_call.return_value = '{"score": 8, "reasoning": "정확합니다"}'
        result = score_response("질문", "답변", "general")
        assert result["score"] == 8
        assert result["reasoning"] == "정확합니다"
        assert result["error"] is None

    @patch("eval_framework.judge._call_judge")
    def test_fallback_to_number_extraction(self, mock_call):
        """JSON 파싱 실패 시 숫자 fallback"""
        mock_call.return_value = "I think the score is 7 out of 10."
        result = score_response("질문", "답변", "general")
        assert result["score"] == 7
        assert result["error"] is None

    @patch("eval_framework.judge._call_judge")
    def test_timeout_then_success(self, mock_call):
        """첫 호출 Timeout, 두번째 성공"""
        mock_call.side_effect = [
            requests.Timeout("timeout"),
            '{"score": 6, "reasoning": "ok"}',
        ]
        result = score_response("질문", "답변", "general")
        assert result["score"] == 6
        assert result["error"] is None

    @patch("eval_framework.judge._call_judge")
    @patch("eval_framework.judge.time.sleep")
    def test_all_retries_fail(self, mock_sleep, mock_call):
        """전체 재시도 실패 → error 반환"""
        mock_call.side_effect = Exception("서버 장애")
        result = score_response("질문", "답변", "general", max_retries=3)
        assert result["score"] == 0
        assert result["error"] is not None
        assert "서버 장애" in result["error"]

    @patch("eval_framework.judge._call_judge")
    @patch("eval_framework.judge.time.sleep")
    def test_connection_error_then_success(self, mock_sleep, mock_call):
        """ConnectionError 후 성공"""
        mock_call.side_effect = [
            requests.ConnectionError("refused"),
            '{"score": 5, "reasoning": "average"}',
        ]
        result = score_response("질문", "답변", "general")
        assert result["score"] == 5

    @patch("eval_framework.judge._call_judge")
    def test_bug_fix_text_initialized(self, mock_call):
        """버그 수정 검증: _call_judge가 JSONDecodeError를 발생시켜도 text가 초기화됨"""
        mock_call.side_effect = [
            json.JSONDecodeError("err", "", 0),
            '{"score": 4, "reasoning": "retry worked"}',
        ]
        result = score_response("질문", "답변", "general")
        assert result["score"] == 4
        assert result["error"] is None

    @patch("eval_framework.judge._call_judge")
    def test_json_fallback_no_valid_number(self, mock_call):
        """JSON 파싱 실패 + 유효 숫자도 없으면 재시도 소진 후 error"""
        mock_call.return_value = "no valid json and no numbers"
        result = score_response("질문", "답변", "general", max_retries=2)
        assert result["score"] == 0
        assert result["error"] == "채점 실패 (최대 재시도 초과)"


# ═══════════════════════════════════════════════════════════════════════════════
# score_pairwise 테스트 (4 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScorePairwise:
    """score_pairwise: 쌍대비교 채점"""

    @patch("eval_framework.judge._call_judge")
    def test_winner_a(self, mock_call):
        mock_call.return_value = '{"winner": "A", "reasoning": "A가 더 낫다"}'
        result = score_pairwise("질문", "응답A", "응답B")
        assert result["winner"] == "A"
        assert result["error"] is None

    @patch("eval_framework.judge._call_judge")
    def test_winner_b(self, mock_call):
        mock_call.return_value = '{"winner": "B", "reasoning": "B가 낫다"}'
        result = score_pairwise("질문", "응답A", "응답B")
        assert result["winner"] == "B"

    @patch("eval_framework.judge._call_judge")
    def test_tie(self, mock_call):
        mock_call.return_value = '{"winner": "tie", "reasoning": "동점"}'
        result = score_pairwise("질문", "응답A", "응답B")
        assert result["winner"] == "TIE"

    @patch("eval_framework.judge._call_judge")
    def test_invalid_winner_defaults_tie(self, mock_call):
        mock_call.return_value = '{"winner": "C", "reasoning": "invalid"}'
        result = score_pairwise("질문", "응답A", "응답B")
        assert result["winner"] == "TIE"

    @patch("eval_framework.judge._call_judge")
    def test_timeout_then_success(self, mock_call):
        mock_call.side_effect = [
            requests.Timeout("timeout"),
            '{"winner": "B", "reasoning": "B가 낫다"}',
        ]
        result = score_pairwise("질문", "응답A", "응답B")
        assert result["winner"] == "B"

    @patch("eval_framework.judge._call_judge")
    @patch("eval_framework.judge.time.sleep")
    def test_all_retries_fail(self, mock_sleep, mock_call):
        mock_call.side_effect = Exception("서버 장애")
        result = score_pairwise("질문", "응답A", "응답B", max_retries=3)
        assert result["winner"] == "TIE"
        assert result["error"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# score_with_criteria 테스트 (3 케이스)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreWithCriteria:
    """score_with_criteria: 다면적 채점"""

    @patch("eval_framework.judge._call_judge")
    def test_valid_multi_criteria(self, mock_call):
        mock_call.return_value = (
            '{"scores": {"정확성": 8, "유용성": 7, "창의성": 9}, '
            '"reasoning": "전반적으로 우수"}'
        )
        criteria = {"정확성": "사실 관계", "유용성": "도움 여부", "창의성": "독창성"}
        result = score_with_criteria("질문", "답변", criteria)
        assert result["scores"]["정확성"] == 8
        assert result["scores"]["유용성"] == 7
        assert result["scores"]["창의성"] == 9
        assert result["error"] is None

    @patch("eval_framework.judge._call_judge")
    def test_timeout_retry(self, mock_call):
        mock_call.side_effect = [
            requests.Timeout("timeout"),
            '{"scores": {"a": 5}, "reasoning": "ok"}',
        ]
        result = score_with_criteria("질문", "답변", {"a": "desc"})
        assert result["scores"]["a"] == 5

    @patch("eval_framework.judge._call_judge")
    @patch("eval_framework.judge.time.sleep")
    def test_all_fail(self, mock_sleep, mock_call):
        mock_call.side_effect = Exception("fail")
        result = score_with_criteria("질문", "답변", {"a": "desc"}, max_retries=3)
        assert result["scores"] == {}
        assert result["error"] is not None
