# NLP2 Test-Time Scaling for Multimodal Reasoning
#
# Minimal template to:
# 1. Load and inspect the EXAMS-V and ImageCLEF 2026 data.
# 2. Run zero-shot and chain-of-thought baselines with Qwen2.5-VL-7B.
# 3. Implement self-consistency (majority voting over N sampled chains).
# 4. Evaluate your outputs (accuracy overall and stratified).
#
# Runs on Google Colab (T4/A100) or locally with a GPU.
# Qwen2.5-VL-7B needs ~16 GB VRAM; use 4-bit quantization on smaller GPUs.
#
# Official evaluation scripts and baselines:
#   https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning

# Uncomment the line below when running on Google Colab:
# !pip install transformers accelerate bitsandbytes qwen-vl-utils datasets -q

import json
import os
import re
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

# --- EXAMS-V (MCQ) -----------------------------------------------------------
# ~20k+ multilingual MCQ questions with images. The question and answer options
# are embedded IN the image - the VLM must read them visually.
# Columns: image, sample_id, answer_key, type, grade, subject, language,
#          chemical_structure, table, figure, graph

examsv = load_dataset("MBZUAI/EXAMS-V")
print(examsv)

examsv_val = examsv["validation"]
print(f"\nEXAMS-V validation: {examsv_val.num_rows} samples")
print(f"Columns: {examsv_val.column_names}")

# Inspect a sample
sample = examsv_val[0]
print(f"\n--- EXAMS-V sample ---")
print(f"  sample_id : {sample['sample_id']}")
print(f"  subject   : {sample['subject']}")
print(f"  language  : {sample['language']}")
print(f"  grade     : {sample['grade']}")
print(f"  type      : {sample['type']}")
print(f"  answer_key: {sample['answer_key']}")
print(f"  image size: {sample['image'].size}")
# Uncomment to display the image inline (Jupyter/Colab):
# display(sample["image"].resize((400, 400)))

# --- OpenQA (Textual) --------------------------------------------------------
# 625 open-ended questions with text-based question content + images.
# Columns: question_id, answer, type, subject, language, image

openqa_textual = load_dataset("SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual")
print(f"\nOpenQA-Textual: {openqa_textual}")

openqa_dev = openqa_textual["dev"]
oqa_sample = openqa_dev[0]
print(f"\n--- OpenQA-Textual sample ---")
print(f"  question_id: {oqa_sample['question_id']}")
print(f"  answer     : {oqa_sample['answer']}")
print(f"  subject    : {oqa_sample['subject']}")
print(f"  language   : {oqa_sample['language']}")
print(f"  image size : {oqa_sample['image'].size}")

# --- Dataset statistics -------------------------------------------------------

print("\n=== EXAMS-V validation set statistics ===")
val_df = examsv_val.to_pandas()
val_df_no_img = val_df.drop(columns=["image"], errors="ignore")
print(f"Total: {len(val_df_no_img)}")
print(f"\nBy language:\n{val_df_no_img['language'].value_counts().to_string()}")
print(f"\nBy subject:\n{val_df_no_img['subject'].value_counts().head(10).to_string()}")
print(f"\nVisual content: {val_df_no_img[['figure', 'graph', 'table', 'chemical_structure']].sum().to_string()}")

# --- Subsetting for development -----------------------------------------------
# IMPORTANT: For development, work on a small subset (200-500 questions).
# Only scale to the full validation set for final results.

DEV_SUBSET_SIZE = 200

val_indices = list(range(examsv_val.num_rows))
random.shuffle(val_indices)
dev_indices = val_indices[:DEV_SUBSET_SIZE]

print(f"\nDevelopment subset: {len(dev_indices)} questions (random sample)")

# You can also filter by language for focused development:
# english_indices = [i for i in range(examsv_val.num_rows) if examsv_val[i]["language"] == "English"]

from transformers import AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Choose your model
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# Lighter alternative for faster iteration:
# MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# --- Option A: 4-bit quantization (recommended for T4 / limited VRAM) --------
USE_4BIT = True

if USE_4BIT:
    from transformers import Qwen2_5_VLForConditionalGeneration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
else:
    # --- Option B: Full precision (needs ~16 GB VRAM) -------------------------
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

processor = AutoProcessor.from_pretrained(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME} (4-bit={USE_4BIT})")


# --- Answer normalization (handles Cyrillic / numeric answer keys) ------------

CYRILLIC_TO_LATIN = {
    "А": "A", "Б": "B", "В": "C", "Г": "D", "Д": "E",
    "а": "A", "б": "B", "в": "C", "г": "D", "д": "E",
}
CANONICAL_MCQ = {"A", "B", "C", "D", "E"}


def normalize_answer_key(raw: str) -> str | None:
    """Normalize answer keys from various formats to A-E."""
    x = str(raw).strip()
    if x.upper() in CANONICAL_MCQ:
        return x.upper()
    if x in CYRILLIC_TO_LATIN:
        return CYRILLIC_TO_LATIN[x]
    if x.isdigit() and 1 <= int(x) <= 5:
        return chr(ord("A") + int(x) - 1)
    return None


def extract_answer(text: str, choices: set[str] = CANONICAL_MCQ) -> str | None:
    """Extract a single answer letter from model output.

    Tries multiple parsing strategies:
    1. "The answer is X" pattern
    2. Standalone letter (entire response is a single letter)
    3. Last standalone letter A-E in the text
    """
    if not text:
        return None
    text_upper = text.upper().strip()

    # Strategy 1: explicit "the answer is X" or "answer: X"
    m = re.search(r"(?:the answer is|answer:?)\s*([A-E])\b", text_upper)
    if m and m.group(1) in choices:
        return m.group(1)

    # Strategy 2: a single letter on its own (the entire response)
    if text_upper.rstrip(".") in choices:
        return text_upper.rstrip(".")

    # Strategy 3: last standalone letter A-E in the text
    matches = re.findall(r"\b([A-E])\b", text_upper)
    if matches:
        return matches[-1]

    return None


# --- Core generation function -------------------------------------------------

def generate_answer(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> str:
    """Generate a text response from the VLM given an image and prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    })

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Trim the input tokens from the output
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


# --- Alternative: vLLM inference (recommended for experiments) ----------------
# The PyTorch inference above processes one image at a time. For real experiments
# (self-consistency, scaling curves, full validation), use vLLM which is much faster.
#
# Start the server in a separate terminal:
#   pip install vllm
#   vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
#       --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.9 \
#       --limit-mm-per-prompt image=1
#
# Note: --max-model-len must be large enough for image tokens + text. Qwen2.5-VL
# can expand a single image to 10k-16k tokens. If you get a BadRequestError about
# decoder prompt length, increase this value or resize input images.
#
# Then uncomment below. This redefines generate_answer to call vLLM instead of
# PyTorch, so all downstream code (CoT, self-consistency, etc.) works without
# changes. You can skip model loading entirely - model and processor are accepted
# but ignored. Throughput tuning depends on your GPU - consult https://docs.vllm.ai/

# import base64, io
# from openai import OpenAI
#
# VLLM_BASE_URL = "http://localhost:8000/v1"
# VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
# vllm_client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)
#
#
# def image_to_data_uri(img):
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     b64 = base64.b64encode(buf.getvalue()).decode()
#     return f"data:image/png;base64,{b64}"
#
#
# def generate_answer(model, processor, image, prompt, system_prompt="",
#                     temperature=0.0, max_new_tokens=512):
#     """Generate a response via the vLLM OpenAI-compatible API.
#
#     Drop-in replacement for the PyTorch generate_answer above.
#     model and processor are accepted for compatibility but not used.
#     """
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": image_to_data_uri(image)}},
#             {"type": "text", "text": prompt},
#         ],
#     })
#     resp = vllm_client.chat.completions.create(
#         model=VLLM_MODEL,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_new_tokens,
#     )
#     return resp.choices[0].message.content or ""


# --- Zero-shot prompts --------------------------------------------------------

MCQ_ZERO_SHOT_PROMPT = (
    "Look at the image and answer the following multiple-choice question. "
    "The question and answer options are shown in the image. "
    "Choose the correct answer from the options.\n\n"
    "Answer with only the letter of the correct option (A, B, C, D, or E)."
)

OPENQA_ZERO_SHOT_PROMPT = (
    "Look at the image and answer the question shown. "
    "Give a short, direct answer."
)


# --- Run zero-shot on 10 MCQ examples ----------------------------------------

print("\n=== Zero-shot baseline (MCQ) ===")
N_DEMO = 10
demo_indices = dev_indices[:N_DEMO]

for i, idx in enumerate(demo_indices):
    item = examsv_val[idx]
    gold = normalize_answer_key(item["answer_key"])

    output = generate_answer(model, processor, item["image"], MCQ_ZERO_SHOT_PROMPT)
    pred = extract_answer(output)

    correct = "✓" if pred == gold else "✗"
    print(f"  [{i+1:2d}] gold={gold}  pred={pred}  {correct}  (raw: {output[:60]})")

MCQ_COT_PROMPT = (
    "Look at the image and answer the following multiple-choice question. "
    "The question and answer options are shown in the image.\n\n"
    "Think step by step before answering. First describe what you see in the "
    "image, then reason through each option. "
    "End your response with: The answer is <letter>."
)


def generate_with_cot(
    model,
    processor,
    image: Image.Image,
    prompt: str = MCQ_COT_PROMPT,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
) -> tuple[str | None, str]:
    """Generate a chain-of-thought response and extract the final answer.

    Returns:
        (extracted_answer, full_reasoning_text)
    """
    reasoning = generate_answer(
        model, processor, image, prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    answer = extract_answer(reasoning)
    return answer, reasoning


# --- Run CoT on same 10 examples ---------------------------------------------

print("\n=== Chain-of-thought baseline (MCQ) ===")
for i, idx in enumerate(demo_indices):
    item = examsv_val[idx]
    gold = normalize_answer_key(item["answer_key"])

    pred, reasoning = generate_with_cot(model, processor, item["image"])

    correct = "✓" if pred == gold else "✗"
    # Show first line of reasoning + extracted answer
    first_line = reasoning.split("\n")[0][:80]
    print(f"  [{i+1:2d}] gold={gold}  pred={pred}  {correct}  ({first_line}...)")

def self_consistency(
    model,
    processor,
    image: Image.Image,
    prompt: str = MCQ_COT_PROMPT,
    n: int = 8,
    temperature: float = 0.7,
) -> tuple[str | None, dict]:
    """Sample N reasoning chains and return the majority-vote answer.

    Args:
        n: Number of reasoning chains to sample.
        temperature: Sampling temperature (must be > 0).

    Returns:
        (majority_answer, {"votes": Counter, "chains": list[str]})
    """
    votes = []
    chains = []
    for _ in range(n):
        answer, reasoning = generate_with_cot(
            model, processor, image, prompt,
            temperature=temperature,
        )
        chains.append(reasoning)
        if answer is not None:
            votes.append(answer)

    vote_counts = Counter(votes)
    majority = vote_counts.most_common(1)[0][0] if vote_counts else None
    return majority, {"votes": vote_counts, "chains": chains}


# --- Run self-consistency on a small subset -----------------------------------

print("\n=== Self-consistency (n=4, MCQ) ===")
SC_N = 4  # Use 4 for demo; increase to 8-16 for real experiments
for i, idx in enumerate(demo_indices[:5]):
    item = examsv_val[idx]
    gold = normalize_answer_key(item["answer_key"])

    pred, info = self_consistency(model, processor, item["image"], n=SC_N)

    correct = "✓" if pred == gold else "✗"
    print(f"  [{i+1}] gold={gold}  pred={pred}  {correct}  votes={dict(info['votes'])}")


# --- TODO: Implement your own TTS strategies ----------------------------------

def tree_of_thought(model, processor, image, prompt, breadth=3, depth=2):
    """TODO: Implement Tree-of-Thought search strategy.

    Generate multiple reasoning paths with step-level branching and evaluation.
    At each step, generate `breadth` continuations, evaluate them, and keep
    the best ones for the next depth level.

    Hint: You can use the model itself to evaluate partial reasoning chains
    (e.g., "Is this reasoning step correct so far? Rate 1-10.").
    """
    raise NotImplementedError


def verify_chain(model, processor, image, question_prompt, chain):
    """TODO: Implement a verification/scoring strategy.

    Score a reasoning chain by checking if its claims are grounded in the image.
    Options to explore:
    - Self-verification: ask the model "Is this reasoning correct?"
    - Claim extraction + checking: extract factual claims, verify each against image
    - VisualPRM: use OpenGVLab/VisualPRM-8B as a process reward model
    - LLM-as-judge: use a second model to evaluate the chain
    """
    raise NotImplementedError


def best_of_n_with_verification(
    model, processor, image, prompt, n=8, temperature=0.7,
):
    """TODO: Implement Best-of-N with your verification strategy.

    Generate N chains, score each with verify_chain, return the highest-scored.
    This separates generation (high temperature) from selection (verification).
    """
    raise NotImplementedError


def your_own_strategy(model, processor, image, prompt, **kwargs):
    """TODO: Implement your own TTS strategy here."""
    raise NotImplementedError


def evaluate(predictions: list[dict], references: list[dict]) -> dict:
    """Compute accuracy overall and stratified by metadata fields.

    Args:
        predictions: list of {"question_id": ..., "predicted_answer": ..., ...}
        references:  list of {"question_id": ..., "answer": ..., "subject": ...,
                     "language": ..., ...}

    Returns:
        dict with overall accuracy and per-subject/language breakdowns.
    """
    pred_map = {str(p["question_id"]): str(p["predicted_answer"]).strip().upper()
                for p in predictions}
    ref_map = {}
    ref_meta = {}
    for r in references:
        qid = str(r["question_id"])
        ref_map[qid] = str(r["answer"]).strip().upper()
        ref_meta[qid] = r

    common = set(pred_map) & set(ref_map)
    correct = sum(1 for qid in common if pred_map[qid] == ref_map[qid])
    total = len(common)

    results = {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}

    # Stratified accuracy
    for field in ["subject", "language"]:
        buckets = {}
        for qid in common:
            val = ref_meta[qid].get(field, "unknown")
            if val not in buckets:
                buckets[val] = {"correct": 0, "total": 0}
            buckets[val]["total"] += 1
            if pred_map[qid] == ref_map[qid]:
                buckets[val]["correct"] += 1
        results[f"by_{field}"] = {
            k: {**v, "accuracy": v["correct"] / v["total"]}
            for k, v in sorted(buckets.items(), key=lambda x: -x[1]["total"])
        }

    return results


def save_predictions(predictions: list[dict], output_path: str):
    """Save predictions in the ImageCLEF 2026 competition JSON format.

    Expected format: [{"id": "...", "prediction": "...", "language": "..."}, ...]
    """
    # Convert from our internal format to competition format
    competition_format = []
    for p in predictions:
        competition_format.append({
            "id": p.get("question_id") or p.get("id"),
            "prediction": p.get("predicted_answer") or p.get("prediction"),
            "language": p.get("language", "English"),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(competition_format, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(competition_format)} predictions to {output_path}")

    # TODO: Adapt for competition submission on AI4Media-Bench platform.
    # Upload the JSON file to the appropriate competition track:
    #   MCQ:    https://ai4media-bench.aimultimedialab.ro/competitions/16/
    #   OpenQA: https://ai4media-bench.aimultimedialab.ro/competitions/18/


def compute_scaling_curve(
    model,
    processor,
    dataset,
    indices: list[int],
    n_values: list[int] = [1, 2, 4, 8, 16],
    prompt: str = MCQ_COT_PROMPT,
    temperature: float = 0.7,
) -> dict[int, float]:
    """TODO: Implement compute-scaling curves (accuracy vs. number of VLM calls).

    For each N in n_values, run your TTS strategy with N reasoning chains
    and measure accuracy. This produces the key plot showing how accuracy
    scales with compute budget.

    Args:
        dataset: HuggingFace dataset split.
        indices: indices into the dataset to evaluate.
        n_values: list of N values to sweep.

    Returns:
        dict mapping N -> accuracy.
    """
    raise NotImplementedError


# --- For official evaluation, use the ImageCLEF repo scripts: ----------------
# python ImageCLEF-MultimodalReasoning/2026/src/evaluation/evaluate_mcq.py \
#     --pred_file outputs/predictions.json --gold_file gold.json --print_score True
