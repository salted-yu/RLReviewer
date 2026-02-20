import json
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

format_rate = 0.1
location_rule_rate = 0.3
comment_rate = 0.6
epsilon = 1e-8

MODEL_PATH = "path/to/all-MiniLM-L6-v2"
sbert_model = SentenceTransformer(MODEL_PATH)

def extract_json(predict_str: str):
    try:
        match = re.search(r'```json\s*({[\s\S]*?})\s*```', predict_str)
        if match:
            s = match.group(1)
            data = json.loads(s)
        else:
            data = json.loads(predict_str)
        return data
    except (json.JSONDecodeError, AttributeError):
        return None

def format_score(predict_data) -> float:
    return 1.0 if isinstance(predict_data, dict) and set(predict_data.keys()) == {"line_number", "rule", "comment"} else 0.0

def location_match(predict_data, ground_truth_data) -> float:
    pl = predict_data.get("line_number")
    gl = ground_truth_data.get("line_number")
    return 1.0 if pl is not None and gl is not None and str(pl) == str(gl) else 0.0

def rule_match(predict_data, ground_truth_data) -> float:
    pr = predict_data.get("rule")
    gr = ground_truth_data.get("rule")
    return 1.0 if pr is not None and gr is not None and pr == gr else 0.0

def comment_metrics(predict_data, ground_truth_data, alpha=0.0, beta=1.0, epsilon=1e-8):
    pred = predict_data.get("comment")
    gt = ground_truth_data.get("comment")
    if pred is None or gt is None or not isinstance(pred, str):
        return {"bleu": 0.0, "semantic_sim": 0.0, "comment_score": 0.0}

    pred_tokens = word_tokenize(pred)
    gt_tokens = word_tokenize(gt)
    smooth = SmoothingFunction().method7
    bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smooth, auto_reweigh=True)

    embeddings = sbert_model.encode([pred, gt], convert_to_tensor=True, normalize_embeddings=True)
    semantic_sim = util.cos_sim(embeddings[0], embeddings[1]).item()

    reward = alpha * bleu + beta * semantic_sim

    return {"bleu": bleu, "semantic_sim": semantic_sim, "comment_score": reward}

def compute_score(predict_str: str, ground_truth_str: str) -> dict:
    score_json = {
        "score": 0.0,
        "format_score": 0.0,
        "location_rule_score": 0.0,
        "location_ok": 0.0,
        "rule_ok": 0.0,
        "comment_score": 0.0,
        "bleu": 0.0,
        "semantic_sim": 0.0,
    }

    predict_json = extract_json(predict_str)
    ground_truth = extract_json(ground_truth_str)
    if predict_json is None or ground_truth is None:
        return score_json

    fmt = format_score(predict_json)

    if fmt == 0.0:
        return score_json
    score_json["format_score"] = fmt * format_rate

    loc_ok = location_match(predict_json, ground_truth)
    rule_ok = rule_match(predict_json, ground_truth)
    score_json["location_ok"] = loc_ok
    score_json["rule_ok"] = rule_ok

    if loc_ok + rule_ok == 2:
        loc_rule_score = location_rule_rate
    elif loc_ok + rule_ok == 1:
        loc_rule_score = 0.1
    else:
        loc_rule_score = 0.0
    score_json["location_rule_score"] = loc_rule_score

    # comment
    ## balanced bleu and bertscore (ours)
    # alpha = 0.5
    # beta = 0.5
    ## Baseline 1: only BLEU
    # alpha = 1.0
    # beta = 0.0
    ## Baseline 2: Only SentenceBert
    alpha = 0.0
    beta = 1.0
    cm = comment_metrics(predict_json, ground_truth, alpha=alpha, beta=beta)
    score_json["bleu"] = cm["bleu"]
    score_json["semantic_sim"] = cm["semantic_sim"]
    score_json["comment_score"] = cm["comment_score"] * comment_rate

    score_json["score"] = score_json["format_score"] + score_json["location_rule_score"] + score_json["comment_score"]

    return score_json

def bleu_cosine_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    score_json = compute_score(solution_str, ground_truth)
    return score_json