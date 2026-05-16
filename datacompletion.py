import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =========================================================
# PERFORMANCE SETTINGS
# =========================================================

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================================================
# IMPORTS
# =========================================================

import json
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# GPU CHECK
# =========================================================

assert torch.cuda.is_available(), "❌ CUDA NOT AVAILABLE"

gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)

print(f"✅ GPU: {gpu_name}")
print(f"✅ Compute Capability: {major}.{minor}")

# =========================================================
# ATTENTION + DTYPE SELECTION
# =========================================================

if major >= 8:
    ATTN_IMPL = "flash_attention_2"
    DTYPE = torch.bfloat16
else:
    ATTN_IMPL = "sdpa"
    DTYPE = torch.float16

print(f"✅ Attention Backend: {ATTN_IMPL}")
print(f"✅ Torch DType: {DTYPE}")

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "Qwen/Qwen3-8B"

INPUT_FILE = BASE_DIR / "dataset_train.jsonl"
OUTPUT_FILE = BASE_DIR / "clean_science_dataset.jsonl"

MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 180

REPETITION_PENALTY = 1.1

MIN_OUTPUT_LENGTH = 40
MAX_OUTPUT_LENGTH = 600

# =========================================================
# PLACEHOLDERS
# =========================================================

PLACEHOLDERS = {
    "",
    "OK",
    "Summary.",
    "Explanation.",
    "Summary of the text.",
    "Explanation of the text."
}

# =========================================================
# LOAD TOKENIZER
# =========================================================

print("\n🚀 Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# =========================================================
# LOAD MODEL
# =========================================================

print("🚀 Loading model on GPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation=ATTN_IMPL
)

model.eval()

print("✅ Model loaded successfully")

# =========================================================
# CLEANING FUNCTIONS
# =========================================================

THINKING_CLOSE = ("</" + "think>", "</think>")
THINKING_OPEN = ("<" + "think>", "<think>")


def has_thinking_tags(text: str) -> bool:
    lower = text.lower()
    return any(tag.lower() in lower for tag in THINKING_OPEN + THINKING_CLOSE)


def remove_thinking(text):

    text = text.strip()

    for close in THINKING_CLOSE:
        if close.lower() in text.lower():
            text = re.split(re.escape(close), text, flags=re.IGNORECASE)[-1]
            break
    else:
        for open_tag in THINKING_OPEN:
            match = re.search(re.escape(open_tag), text, re.IGNORECASE)
            if match:
                text = text[: match.start()]
                break

    patterns = [
        r"<" + r"think>.*?</" + r"think>",
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"```.*?```",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

    return text.strip()

# =========================================================

def clean_output(text):

    text = remove_thinking(text)

    bad_markers = [
        "Human:",
        "Assistant:",
        "<|im_end|>",
        "<|endoftext|>",
        "<|assistant|>"
    ]

    for marker in bad_markers:
        if marker in text:
            text = text.split(marker)[0]

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================================================

def normalize_output(text):

    text = text.strip()

    if len(text) > MAX_OUTPUT_LENGTH:
        cut = text[:MAX_OUTPUT_LENGTH].rfind(" ")
        text = (text[:cut] if cut > MIN_OUTPUT_LENGTH else text[:MAX_OUTPUT_LENGTH]).strip()

    if text and text[-1] not in ".!?":
        text += "."

    return text

# =========================================================

def is_mcq(text):

    text_lower = text.lower()

    patterns = [
        r"\ba\)",
        r"\bb\)",
        r"\bc\)",
        r"\bd\)",
        r"\boption\b",
        r"\bmultiple choice\b",
        r"\bchoose the correct\b"
    ]

    matches = 0

    for p in patterns:
        if re.search(p, text_lower):
            matches += 1

    return matches >= 3

# =========================================================

def is_valid_output(text):

    text = text.strip()

    if len(text) < MIN_OUTPUT_LENGTH:
        return False

    if len(text) > MAX_OUTPUT_LENGTH:
        return False

    if has_thinking_tags(text):
        return False

    if is_mcq(text):
        return False

    if not text.endswith((".", "!", "?")):
        return False

    return True

# =========================================================

def needs_generation(text):

    text = text.strip()

    if text in PLACEHOLDERS:
        return True

    if len(text) < 30:
        return True

    if has_thinking_tags(text):
        return True

    return False

# =========================================================
# PROMPT BUILDER
# =========================================================

def build_messages(subject, instruction, input_text):

    system_prompt = """
You are creating a professional instruction-tuning dataset.

STRICT RULES:
- Answer ONLY the instruction
- Use clear educational language
- Keep answers concise and informative
- Write 2-4 sentences only
- Never explain your reasoning
- Never use chain-of-thought or reasoning tags
- Never generate MCQs
- Never use bullet points
- Never repeat the input text
- Return only the final answer
"""

    user_prompt = f"""
Subject: {subject}

Instruction:
{instruction}

Text:
{input_text}
"""

    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

# =========================================================
# GENERATION
# =========================================================

def generate_answer(subject, instruction, input_text):

    messages = build_messages(
        subject,
        instruction,
        input_text
    )

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        print("⚠️  tokenizer lacks enable_thinking; upgrade transformers for Qwen3")
        template_kwargs.pop("enable_thinking")
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(model.device)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]

    decoded = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    cleaned = normalize_output(clean_output(decoded))

    if not cleaned or has_thinking_tags(cleaned):
        return ""

    return cleaned

# =========================================================
# PROCESS DATASET
# =========================================================

def count_lines(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def process_dataset():

    total = 0
    generated = 0
    kept = 0
    failed = 0
    skipped = 0

    if not INPUT_FILE.exists():
        print(f"❌ Missing input file: {INPUT_FILE}")
        return

    resume_from = count_lines(OUTPUT_FILE)
    input_total = count_lines(INPUT_FILE)

    print("\n" + "=" * 70)
    print(f"📚 INPUT  : {INPUT_FILE.name} ({input_total} rows)")
    print(f"📁 OUTPUT : {OUTPUT_FILE.name}")
    if resume_from:
        print(f"⏩ RESUME : skipping first {resume_from} rows already in output")
    print("=" * 70)

    with INPUT_FILE.open("r", encoding="utf-8") as fin, \
         OUTPUT_FILE.open("a", encoding="utf-8") as fout:

        for line in fin:

            total += 1

            if total <= resume_from:
                skipped += 1
                continue

            try:
                item = json.loads(line)

            except Exception as e:
                failed += 1
                print(f"❌ JSON ERROR [{total}] -> {e}")
                continue

            subject = (
                item.get("subject", "General")
                .strip()
                .capitalize()
            )

            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output_text = item.get("output", "").strip()

            print(f"\n➡️ Item {total} [{subject}]")

            try:

                if needs_generation(output_text):

                    print("🔄 Generating clean output...")

                    generated_output = generate_answer(
                        subject,
                        instruction,
                        input_text
                    )

                    if not generated_output:
                        failed += 1
                        print("❌ Empty or thinking output — not saved (will retry on next run)")
                        continue

                    item["output"] = generated_output
                    generated += 1

                    print("✨ Generated:")
                    print(generated_output[:200])

                else:

                    cleaned = clean_output(output_text)

                    if is_valid_output(cleaned):

                        item["output"] = cleaned

                        kept += 1

                        print("✓ Existing output kept")

                    else:

                        print("🔄 Existing output invalid -> regenerating")

                        generated_output = generate_answer(
                            subject,
                            instruction,
                            input_text
                        )

                        if not generated_output:
                            failed += 1
                            print("❌ Empty or thinking output — not saved (will retry on next run)")
                            continue

                        item["output"] = generated_output
                        generated += 1

                item["subject"] = subject

                fout.write(
                    json.dumps(
                        item,
                        ensure_ascii=False
                    ) + "\n"
                )
                fout.flush()

            except Exception as e:

                failed += 1

                print(f"❌ FAILED [{total}] -> {e}")

    # =====================================================
    # FINAL STATS
    # =====================================================

    print("\n" + "=" * 70)
    print("🎉 DATASET GENERATION COMPLETE")
    print("=" * 70)

    print(f"📁 Output File : {OUTPUT_FILE}")
    print(f"📊 Total Items : {total}")
    print(f"✨ Generated   : {generated}")
    print(f"✓ Kept        : {kept}")
    print(f"❌ Failed      : {failed}")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    process_dataset()