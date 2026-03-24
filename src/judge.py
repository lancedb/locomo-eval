from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from src.schema import JudgedResult, QaResult

load_dotenv()

JUDGE_SYSTEM_PROMPT = "You are an expert grader that determines whether answers match a gold answer."


def grade_results(
    results: list[QaResult],
    *,
    model: str,
    base_url: str | None = None,
    token: str | None = None,
    concurrency: int = 1,
) -> list[JudgedResult]:
    _, resolved_model = parse_model_ref(model)
    api_key = token or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Judge API key is required via --judge-token or OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url=base_url)

    def _grade_at(idx: int, result: QaResult) -> tuple[int, JudgedResult]:
        if result.error or not result.response:
            return idx, JudgedResult(
                benchmark_id=result.benchmark_id,
                sample_id=result.sample_id,
                category=result.category,
                result="WRONG",
                reasoning=result.error or "empty_response",
                question=result.question,
                answer=result.answer,
                response=result.response,
                error=result.error,
            )
        try:
            return idx, _grade_one(client, resolved_model, result)
        except Exception as exc:
            return idx, _judge_error_result(result, _format_judge_error(exc))

    if concurrency <= 1:
        judged: list[JudgedResult] = []
        sticky_judge_error: str | None = None
        for result in results:
            if result.error or not result.response:
                judged.append(
                    JudgedResult(
                        benchmark_id=result.benchmark_id,
                        sample_id=result.sample_id,
                        category=result.category,
                        result="WRONG",
                        reasoning=result.error or "empty_response",
                        question=result.question,
                        answer=result.answer,
                        response=result.response,
                        error=result.error,
                    )
                )
                continue
            if sticky_judge_error:
                judged.append(_judge_error_result(result, sticky_judge_error))
                continue
            try:
                judged.append(_grade_one(client, resolved_model, result))
            except Exception as exc:
                sticky_judge_error = _format_judge_error(exc)
                judged.append(_judge_error_result(result, sticky_judge_error))
        return judged

    ordered: list[JudgedResult | None] = [None] * len(results)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_grade_at, i, r): i for i, r in enumerate(results)}
        for future in as_completed(futures):
            idx, judged_result = future.result()
            ordered[idx] = judged_result
    return ordered  # type: ignore[return-value]


def parse_model_ref(model: str) -> tuple[str | None, str]:
    if "/" not in model:
        return None, model
    provider, model_name = model.split("/", 1)
    provider = provider.strip() or None
    model_name = model_name.strip()
    if not model_name:
        raise ValueError(f"Invalid model reference: {model}")
    return provider, model_name


def _grade_one(client: OpenAI, model: str, result: QaResult) -> JudgedResult:
    prompt = f"""
Your task is to label an answer to a question as CORRECT or WRONG.

You will be given:
1. a question
2. a gold answer
3. a generated answer

Be generous in grading.
If the generated answer clearly refers to the same fact, count it as CORRECT.
For dates and times, count format differences or relative wording as CORRECT if they refer to the same date or time period.

Question: {result.question}
Gold answer: {result.answer}
Generated answer: {result.response}

Respond with JSON only:
{{"is_correct":"CORRECT or WRONG","reasoning":"one short sentence"}}
""".strip()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    data = _extract_json(content)
    is_correct = str(data.get("is_correct", "WRONG")).strip().upper() == "CORRECT"
    reasoning = str(data.get("reasoning", "")).strip() or "judge_returned_no_reasoning"
    return JudgedResult(
        benchmark_id=result.benchmark_id,
        sample_id=result.sample_id,
        category=result.category,
        result="CORRECT" if is_correct else "WRONG",
        reasoning=reasoning,
        question=result.question,
        answer=result.answer,
        response=result.response,
        error=result.error,
    )


def _extract_json(content: str) -> dict[str, object]:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _judge_error_result(result: QaResult, reason: str) -> JudgedResult:
    return JudgedResult(
        benchmark_id=result.benchmark_id,
        sample_id=result.sample_id,
        category=result.category,
        result="WRONG",
        reasoning=reason,
        question=result.question,
        answer=result.answer,
        response=result.response,
        error=result.error,
    )


def _format_judge_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"judge_api_error: {message}"
