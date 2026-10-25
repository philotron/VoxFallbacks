from typing import Dict, Any
from functools import cache
import ast
import json
import logging
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import httpx
from ollama import ChatResponse, Client
from openai import OpenAI
from dotenv import load_dotenv

from utils import (
    read_data,
    read_data_skill_specific,
    eval_at_threshold,
    pick_threshold_precision,
    summarize_latency,
    _LABEL_MAP_SKILL_SPECIFIC,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

@cache
def _get_openai_client() -> OpenAI:
    return OpenAI()


ollama_client = Client(timeout=90.0)
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
RASA_API_URL = "http://localhost:5005/webhooks/rest/webhook"
RASA_INTENT_DESCRIPTIONS: dict[str, str] = {
    "about_me": "Fragen zur Person / Identitat des Nutzers",
    "active_listener": "Aktiver Zuhorer / Smalltalk",
    "blood_pressure_log": "Blutdruckmessung protokollieren",
    "body_weight_log": "Korpergewicht protokollieren",
    "calling": "Jemanden anrufen",
    "change_volume": "Lautstarke andern (lauter / leiser)",
    "daily_wellbeing": "Tagliches Wohlbefinden erfassen",
    "day_planer": "Tagesplanung",
    "emergency": "Notfall auslosen",
    "general_reminder": "Allgemeine Erinnerung stellen",
    "hydro": "Trinkprotokoll",
    "knowledge_search": "Wissensfrage beantworten",
    "medication_reminder": "Medikamentenerinnerung stellen",
    "messaging": "Nachricht senden",
    "mindfulness": "Achtsamkeitsubung durchfuhren",
    "mobilisation": "Mobilisationsubung durchfuhren",
    "news_search": "Aktuelle Nachrichten abfragen",
    "notebook": "Notiz aufnehmen",
    "repeat": "Etwas wiederholen",
    "sleep_log": "Schlaf protokollieren",
    "standing_test": "Stehversuch durchfuhren",
    "step_counter": "Schrittzahler abfragen",
    "suicide_risk": "Suizidrisiko-Einschatzung",
    "tell_time": "Uhrzeit nennen",
    "time_reminder": "Zeitbasierte Erinnerung / Wecker stellen",
    "vital_measure": "Vitalzeichen messen (Blutdruck / Sauerstoff)",
    "weather": "Wetter abfragen",
    "word_riddle": "Wortratsel (alte Schatzchen)",
    "writing": "Witz oder kreativen Text erzahlen",
}


def _progress(iterable, desc: str):
    """Use plain tqdm output to avoid widget rendering issues in VS Code notebooks."""
    return tqdm(iterable, desc=desc, unit="sample", leave=True, dynamic_ncols=False, ascii=True)


def _timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def _logsumexp(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def rasa_ranked_intent_names(
    message: str,
    sender: str = "evaluation_script",
    url: str = RASA_API_URL,
) -> list[str]:
    """Query Rasa and return ranked intent names parsed from the webhook response."""
    payload = {
        "sender": sender,
        "message": message,
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected Rasa response format: expected a list.")

    for entry in data:
        text = entry.get("text")
        if not isinstance(text, str):
            continue
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            continue

        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return [item["name"] for item in parsed if isinstance(item.get("name"), str)]

    raise ValueError("Could not parse ranked intents from Rasa response.")


def build_rasa_prompt(base_prompt: str, ranked_intents: list[str]) -> str:
    """Inject Rasa intents into the prompt, using a placeholder if present."""
    if ranked_intents:
        intent_lines = [
            f"- {intent} : {RASA_INTENT_DESCRIPTIONS.get(intent, 'Keine Beschreibung verfugbar')}"
            for intent in ranked_intents
        ]
        intents_text = "\n".join(intent_lines)
    else:
        intents_text = "none"

    if "{RASA_INTENTS}" in base_prompt:
        return base_prompt.replace("{RASA_INTENTS}", intents_text)
    return (
        f"{base_prompt}\n\n"
        "RASA top intent candidates (highest confidence first):\n"
        f"{intents_text}\n"
        "Use these candidates as additional context when selecting the final label."
    )


def llm_score_logprobs(prompt: str, model: str, text: str, top_logprobs: int = 20) -> float:
    """
    Returns p_intended in [0,1] using token logprobs for '1' vs '0'
    at the first generated position.
    """

    if model.startswith("gpt-"):
        response = _get_openai_client().chat.completions.create(
            model=model,
            temperature=0.0,
            top_p=1.0,
            logprobs=True,
            top_logprobs=top_logprobs,
            max_tokens=1,
            messages=[
                {
                    'role': 'system',
                    'content': prompt
                },
                {
                    'role': 'user',
                    'content': f'Klassifiziere folgende Nutzeräußerung: {text}',
                },
            ]
        )
        choice = response.choices[0]
        generated_token = (choice.message.content or "").strip()

        token_logprobs = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []
        if not token_logprobs:
            return generated_token, 1.0 if generated_token == "1" else 0.0

        first = token_logprobs[0]
        cand = {}
        for t in first.top_logprobs or []:
            tok = (t.token or "").strip()
            if tok:
                cand[tok] = float(t.logprob)

        lp1 = cand.get("1", None)
        lp0 = cand.get("0", None)

        if lp1 is None and generated_token == "1":
            lp1 = float(first.logprob)
        if lp0 is None and generated_token == "0":
            lp0 = float(first.logprob)

        if lp1 is None or lp0 is None:
            return generated_token, 1.0 if generated_token == "1" else 0.0

        denom = _logsumexp(lp1, lp0)
        return generated_token, float(math.exp(lp1 - denom))

    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Klassifiziere folgende Nutzeräußerung: {text}"},
            ],
            "stream": False,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "num_predict": 1,
                "stop": ["\n", " "],  # keep output to one token
            },
        }

        response: ChatResponse = ollama_client.chat(**payload)


        # logprobs is an array of token-level info for generated tokens
        lp = response.get("logprobs", None)
        if not lp:
            # fallback: if we don't have logprobs, use the generated token
            token = response["message"]["content"].strip()
            return 1.0 if token == "1" else 0.0

        first = lp[0]
        generated_token = response["message"]["content"].strip()

        # Build a map from candidate token -> logprob.
        # Tokens may include leading spaces depending on tokenizer; normalize by strip().
        cand = {}
        for t in first.get("top_logprobs", []) or []:
            tok = (t.get("token") or "").strip()
            if tok:
                cand[tok] = float(t.get("logprob"))

        # Get logprobs for '1' and '0' if present
        lp1 = cand.get("1", None)
        lp0 = cand.get("0", None)

        # If one is missing, try to use the generated token's own logprob if it matches
        if lp1 is None and generated_token == "1":
            lp1 = float(first.get("logprob"))
        if lp0 is None and generated_token == "0":
            lp0 = float(first.get("logprob"))

        # If still missing (rare), fallback to hard decision
        if lp1 is None or lp0 is None:
            return 1.0 if generated_token == "1" else 0.0

        # Convert two logprobs to normalized probability
        denom = _logsumexp(lp1, lp0)
        p_intended = math.exp(lp1 - denom)
        return generated_token, float(p_intended)


def run_llm_evaluation_binary(
    data_path: str,
    text_col: str,
    label_col: str,
    split_col: str,
    id_col: str,
    prompt: str,
    model_name: str,
):
    
    # Step 1: Load data
    logger.info("[1/5] Loading data from %s", data_path)
    df = read_data(data_path, text_col, label_col, split_col, id_col)
    logger.info("Loaded %d samples (label distribution: %s)", len(df), df[label_col].value_counts().to_dict())
    
    # Step 5: Calculate test scores
    #logger.info("[5/5] Scoring test split at threshold %.2f", thr)
    predictions_by_id = {}
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        x = row[text_col]
        uid = row[id_col]
        label = row[label_col]
        prediction, latency_seconds = _timed_call(llm_score_logprobs, prompt, model_name, x)
        predictions_by_id[uid] = {
                "score": float(prediction[1]),
                "prediction": int(prediction[0]),
                "label": int(label),  # Included so you can calculate metrics against ground truth later
                "latency": latency_seconds,
        }
        
    return predictions_by_id
    
    

def llm_predict_class(
    prompt: str,
    model: str,
    text: str,
    valid_labels: set[str] | None = None,
    return_invalid_label: bool = False,
) -> str | tuple[str, str | None]:
    """Call the LLM and return a valid label, falling back to unclear.

    When return_invalid_label is True, also return the invalid first token
    emitted by the model, if any.
    """
    valid = valid_labels if valid_labels is not None else set(_LABEL_MAP_SKILL_SPECIFIC.values())

    if model.startswith("gpt-"):
        response = _get_openai_client().chat.completions.create(
            model=model,
            temperature=0.0,
            top_p=1.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Klassifiziere folgende Nutzeräußerung: {text}"},
            ],
        )
        raw = response.choices[0].message.content.strip()
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Klassifiziere folgende Nutzeräußerung: {text}"},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "top_p": 1,
            },
        }
        try:
            response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=90.0)
            response.raise_for_status()
            raw = response.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger.warning("Ollama call timed out for input: %s", text)
            raw = ""
            predicted = "TIMEOUT"
            invalid_label = "TIMEOUT"
            normalized_prediction = "unclear"
            if return_invalid_label:
                return normalized_prediction, invalid_label
            return normalized_prediction, -1

    predicted = raw.split()[0] if raw else ""
    invalid_label = None if predicted in valid else predicted
    if invalid_label is not None:
        logger.warning(
            "Predicted label '%s' not in valid labels %s for input: %s",
            predicted, valid, text,
        )
    normalized_prediction = predicted if predicted in valid else "unclear"
    if return_invalid_label:
        return normalized_prediction, invalid_label
    return normalized_prediction, -1


def run_llm_evaluation_skills(
    data_path: str,
    text_col: str,
    label_col: str,
    split_col: str,
    id_col: str,
    prompt: str,
    model_name: str,
    prompt_variant: str = "default",
) -> Dict[Any, Dict[str, Any]]:
    """
    Run skill-specific LLM evaluation, grouped by cross-validation splits,
    returning item-level predictions mapped by ID.
    """
    logger.info("[1/2] Loading skill-specific data from %s", data_path)
    df = read_data_skill_specific(data_path, text_col, label_col, split_col, id_col)
    
    skill_labels = sorted(df[label_col].unique().tolist())
    valid_labels = set(skill_labels)

    unique_splits = df[split_col].unique()
    predictions_by_id = {}
    use_rasa_prompt_variant = (prompt_variant or "").strip().lower() == "rasa"

    if use_rasa_prompt_variant:
        logger.info("RASA prompt variant enabled: querying Rasa intents before each LLM call")

    logger.info("[2/2] Scoring splits with %s", model_name)
    
    for split_val in unique_splits:
        print(f"\n========== Evaluating split: {split_val} ==========")
        test_df = df[df[split_col] == split_val]
        
        X_test = test_df[text_col].tolist()
        y_test = test_df[label_col].tolist()
        test_ids = test_df[id_col].tolist()

        for uid, x, true_label in zip(test_ids, _progress(X_test, f"Predicting ({split_val})"), y_test):
            sample_start = time.perf_counter()
            prompt_for_sample = prompt
            rasa_latency = None
            
            if use_rasa_prompt_variant:
                rasa_start = time.perf_counter()
                try:
                    ranked_intents = rasa_ranked_intent_names(x)
                    prompt_for_sample = build_rasa_prompt(prompt, ranked_intents)
                except (requests.RequestException, ValueError) as e:
                    logger.warning("Rasa intent lookup failed for sample '%s': %s", x, e)
                    prompt_for_sample = build_rasa_prompt(prompt, [])
                finally:
                    rasa_latency = time.perf_counter() - rasa_start

            prediction_result, llm_latency = _timed_call(
                llm_predict_class,
                prompt_for_sample,
                model_name,
                x,
                valid_labels=valid_labels,
                return_invalid_label=True,
            )
            predicted, invalid_label = prediction_result
            total_latency = time.perf_counter() - sample_start

            predictions_by_id[uid] = {
                "prediction": predicted,
                "label": true_label,
                "invalid_prediction": invalid_label,
                "latency_seconds": total_latency,
                "llm_latency_seconds": llm_latency,
                "rasa_latency_seconds": rasa_latency,
            }

    return predictions_by_id
