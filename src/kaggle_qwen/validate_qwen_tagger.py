import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Qwen tagger model on held-out JSONL samples.")
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--tags-path", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen25_7b_validation"))

    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all rows")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_bnb_config(load_in_4bit: bool, use_bf16: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None

    compute_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def extract_json_object(text: str) -> dict:
    if not text:
        raise ValueError("Empty output")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def normalize_tags(tags: Sequence[str], allowed_set: set[str]) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    outside: List[str] = []
    seen = set()

    for tag in tags:
        t = str(tag).strip()
        if not t:
            continue
        if t in allowed_set:
            if t not in seen:
                kept.append(t)
                seen.add(t)
        else:
            outside.append(t)

    return kept, outside


def to_model_inputs(encoded, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Normalize chat-template outputs across transformers versions.
    It may return a tensor (older behavior) or a BatchEncoding/dict.
    """
    if isinstance(encoded, torch.Tensor):
        return {"input_ids": encoded.to(device)}

    if hasattr(encoded, "keys"):
        out: Dict[str, torch.Tensor] = {}
        for k in encoded.keys():
            v = encoded[k]
            if torch.is_tensor(v):
                out[k] = v.to(device)
        if "input_ids" not in out:
            raise ValueError("tokenizer output missing input_ids")
        return out

    if isinstance(encoded, dict):
        out = {k: v.to(device) for k, v in encoded.items() if torch.is_tensor(v)}
        if "input_ids" not in out:
            raise ValueError("tokenizer output missing input_ids")
        return out

    raise TypeError(f"Unsupported tokenizer output type: {type(encoded)}")


def micro_prf(pred_sets: Sequence[set[str]], gold_sets: Sequence[set[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for pred, gold in zip(pred_sets, gold_sets):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def macro_prf(pred_sets: Sequence[set[str]], gold_sets: Sequence[set[str]], labels: Sequence[str]) -> Dict[str, float]:
    per_label = []

    for label in labels:
        tp = fp = fn = 0
        for pred, gold in zip(pred_sets, gold_sets):
            pred_has = label in pred
            gold_has = label in gold
            if pred_has and gold_has:
                tp += 1
            elif pred_has and not gold_has:
                fp += 1
            elif (not pred_has) and gold_has:
                fn += 1

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label.append((p, r, f))

    if not per_label:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = sum(x[0] for x in per_label) / len(per_label)
    recall = sum(x[1] for x in per_label) / len(per_label)
    f1 = sum(x[2] for x in per_label) / len(per_label)
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match_rate(pred_sets: Sequence[set[str]], gold_sets: Sequence[set[str]]) -> float:
    if not pred_sets:
        return 0.0
    matches = sum(1 for pred, gold in zip(pred_sets, gold_sets) if pred == gold)
    return matches / len(pred_sets)


def main() -> None:
    args = parse_args()

    if not args.val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {args.val_file}")
    if not args.tags_path.exists():
        raise FileNotFoundError(f"Tags file not found: {args.tags_path}")
    if args.adapter_path is not None and not args.adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    all_tags = load_json(args.tags_path)
    if not isinstance(all_tags, list) or not all(isinstance(t, str) for t in all_tags):
        raise ValueError("Tags file must be a JSON array of strings")
    allowed_set = set(all_tags)

    rows = load_jsonl(args.val_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(load_in_4bit=args.load_in_4bit, use_bf16=args.bf16)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter_path))

    model.eval()

    predictions: List[dict] = []
    pred_sets: List[set[str]] = []
    gold_sets: List[set[str]] = []
    parse_success = 0
    raw_outside_count = 0
    raw_pred_tag_count = 0

    for row in rows:
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        model_inputs = to_model_inputs(encoded, model.device)

        do_sample = args.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["top_p"] = args.top_p
            gen_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            output_ids = model.generate(**model_inputs, **gen_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        parsed_ok = True
        parse_error = ""
        raw_tags: List[str] = []

        try:
            parsed = extract_json_object(raw_text)
            raw_tags = parsed.get("final_tags", []) or []
            if not isinstance(raw_tags, list):
                raw_tags = []
        except Exception as exc:
            parsed_ok = False
            parse_error = str(exc)

        pred_tags, outside_tags = normalize_tags(raw_tags, allowed_set)
        raw_outside_count += len(outside_tags)
        raw_pred_tag_count += len([str(t).strip() for t in raw_tags if str(t).strip()])

        gold_tags = row.get("gold_tags", []) or []
        gold_tags = [str(t).strip() for t in gold_tags if str(t).strip() in allowed_set]

        if parsed_ok:
            parse_success += 1

        pred_set = set(pred_tags)
        gold_set = set(gold_tags)
        pred_sets.append(pred_set)
        gold_sets.append(gold_set)

        predictions.append(
            {
                "id": row.get("id", ""),
                "book_id": row.get("book_id", ""),
                "gold_tags": gold_tags,
                "pred_tags": pred_tags,
                "outside_tags_raw": outside_tags,
                "parse_success": parsed_ok,
                "parse_error": parse_error,
                "raw_response": raw_text,
            }
        )

    micro = micro_prf(pred_sets, gold_sets)
    macro = macro_prf(pred_sets, gold_sets, all_tags)
    em = exact_match_rate(pred_sets, gold_sets)

    parse_success_rate = parse_success / len(rows) if rows else 0.0
    outside_rate = raw_outside_count / raw_pred_tag_count if raw_pred_tag_count else 0.0

    report = {
        "total_samples": len(rows),
        "parse_success": parse_success,
        "parse_success_rate": parse_success_rate,
        "raw_pred_tag_count": raw_pred_tag_count,
        "raw_outside_taxonomy_tag_count": raw_outside_count,
        "raw_outside_taxonomy_rate": outside_rate,
        "exact_match_rate": em,
        "micro": micro,
        "macro": macro,
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path) if args.adapter_path else "",
        "val_file": str(args.val_file),
        "tags_path": str(args.tags_path),
    }

    pred_path = args.output_dir / "predictions.jsonl"
    report_path = args.output_dir / "validation_report.json"

    write_jsonl(pred_path, predictions)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[done] validation finished")
    print(f"  predictions: {pred_path}")
    print(f"  report:      {report_path}")
    print(f"  micro_f1:    {micro['f1']:.4f}")
    print(f"  macro_f1:    {macro['f1']:.4f}")
    print(f"  exact_match: {em:.4f}")


if __name__ == "__main__":
    main()
