import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer


DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
JSON_CODE_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(value: Any) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a prediction or reference value."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    candidates = [value.strip()]
    code_block = JSON_CODE_BLOCK.search(value)
    if code_block:
        candidates.insert(0, code_block.group(1))

    start, end = value.find("{"), value.rfind("}")
    if start != -1 and end > start:
        candidates.append(value[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def compute_bleu(prediction: str, reference: str) -> float:
    """Compute sentence-level BLEU using the original smoothing configuration."""
    prediction_tokens = word_tokenize(prediction, preserve_line=True)
    reference_tokens = word_tokenize(reference, preserve_line=True)
    if not prediction_tokens or not reference_tokens:
        return 0.0
    return float(
        sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            smoothing_function=SmoothingFunction().method7,
            auto_reweigh=True,
        )
    )


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load non-empty JSON objects from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            records.append(record)
    return records


def evaluate(records: List[Dict[str, Any]], model: SentenceTransformer) -> Dict[str, float]:
    """Compute location, rule, BLEU, and SentenceBERT scores."""
    if not records:
        return {
            "Location Accuracy": 0.0,
            "Rule Accuracy": 0.0,
            "BLEU": 0.0,
            "SentenceBERT": 0.0,
        }

    location_total = 0.0
    rule_total = 0.0
    bleu_total = 0.0
    prediction_comments: List[str] = []
    reference_comments: List[str] = []

    for record in records:
        prediction = extract_json(record.get("predict"))
        reference = extract_json(record.get("label"))
        if prediction is None or reference is None:
            continue

        predicted_location = prediction.get("line_number")
        reference_location = reference.get("line_number")
        location_total += float(
            predicted_location is not None
            and reference_location is not None
            and str(predicted_location) == str(reference_location)
        )

        predicted_rule = prediction.get("rule")
        reference_rule = reference.get("rule")
        rule_total += float(
            predicted_rule is not None
            and reference_rule is not None
            and str(predicted_rule) == str(reference_rule)
        )

        predicted_comment = prediction.get("comment")
        reference_comment = reference.get("comment")
        if isinstance(predicted_comment, str) and isinstance(reference_comment, str):
            bleu_total += compute_bleu(predicted_comment, reference_comment)
            prediction_comments.append(predicted_comment)
            reference_comments.append(reference_comment)

    sentencebert_total = 0.0
    if prediction_comments:
        prediction_embeddings = model.encode(
            prediction_comments,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        reference_embeddings = model.encode(
            reference_comments,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sentencebert_total = float(
            (prediction_embeddings * reference_embeddings).sum(dim=1).sum().item()
        )

    sample_count = len(records)
    return {
        "Location Accuracy": location_total / sample_count,
        "Rule Accuracy": rule_total / sample_count,
        "BLEU": bleu_total / sample_count,
        "SentenceBERT": sentencebert_total / sample_count,
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions in a JSONL file.")
    parser.add_argument("jsonl_path", type=Path, help="Path to the prediction JSONL file.")
    args = parser.parse_args()

    try:
        records = load_jsonl(args.jsonl_path)
        model_name_or_path = os.environ.get(
            "SENTENCE_TRANSFORMER_MODEL", DEFAULT_SENTENCE_TRANSFORMER
        )
        model = SentenceTransformer(model_name_or_path)
        print_metrics(evaluate(records, model))
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
