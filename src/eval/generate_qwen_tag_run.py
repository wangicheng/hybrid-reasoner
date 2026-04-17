import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-16") as f:
            return json.load(f)


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


def build_system_prompt() -> str:
    return (
        "你是繁體中文小說標籤提取器。"
        "你只能根據給定需求文字與候選標籤做判斷，不得使用外部知識。"
        "輸出必須是 JSON，欄位為 thinking 與 final_tags。"
    )


def build_user_prompt(query_text: str, allowed_tags: Sequence[str]) -> str:
    tag_block = "、".join(allowed_tags)
    return (
        "請根據以下找書需求選出最符合的標籤。\n"
        "硬性規則：\n"
        "1. 只能從候選標籤中挑選。\n"
        "2. 若證據不足可少標，不要亂標。\n"
        "3. 請輸出 JSON，格式必須正確。\n\n"
        f"候選標籤：{tag_block}\n\n"
        "請輸出 JSON：\n"
        "{\n"
        "  \"thinking\": \"...\",\n"
        "  \"final_tags\": [\"標籤1\", \"標籤2\"]\n"
        "}\n\n"
        f"需求：\n{query_text}"
    )


def to_model_inputs(encoded, device: torch.device) -> Dict[str, torch.Tensor]:
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


def build_comparison(base_report: dict, tuned_report: dict) -> dict:
    keys = [
        "parse_success_rate",
        "raw_outside_taxonomy_rate",
        "required_exact_cover_rate",
        "blocked_clean_rate",
        "required_micro_f1",
        "required_macro_f1",
    ]

    metrics: Dict[str, Dict[str, float]] = {}
    for key in keys:
        base_value = float(base_report.get(key, 0.0))
        tuned_value = float(tuned_report.get(key, 0.0))
        metrics[key] = {
            "base": base_value,
            "tuned": tuned_value,
            "delta_tuned_minus_base": tuned_value - base_value,
        }

    return {
        "total_queries": int(tuned_report.get("total_queries", 0)),
        "base_model": tuned_report.get("base_model", ""),
        "adapter_path": tuned_report.get("adapter_path", ""),
        "metrics": metrics,
    }


class QwenTagRunGenerator:
    """
    多實驗執行器（Qwen 標籤抽取版）
    負責對輸入的 Query 跑 base/tuned 兩組模型並輸出具體性能。
    """

    def __init__(
        self,
        all_tags: Sequence[str],
        base_model: str,
        max_new_tokens: int = 220,
        temperature: float = 0.0,
        top_p: float = 1.0,
        load_in_4bit: bool = True,
        bf16: bool = False,
        max_samples: int = 0,
    ):
        self.all_tags = list(all_tags)
        self.allowed_set = set(all_tags)
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_in_4bit = load_in_4bit
        self.bf16 = bf16
        self.max_samples = max_samples

    def _load_model_and_tokenizer(self, adapter_path: Optional[Path]):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = build_bnb_config(load_in_4bit=self.load_in_4bit, use_bf16=self.bf16)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )

        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path))

        model.eval()
        return tokenizer, model

    def _run_single_query(self, tokenizer, model, q_item: Dict[str, object]) -> Dict[str, object]:
        q_id = str(q_item.get("id", ""))
        query = str(q_item.get("query", "")).strip()
        golden_rules = q_item.get("golden_rules") or {}

        required_gold, _ = normalize_tags(golden_rules.get("required_tags") or [], self.allowed_set)
        blocked_gold, _ = normalize_tags(golden_rules.get("blocked_tags") or [], self.allowed_set)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(query_text=query, allowed_tags=self.all_tags)},
        ]

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        model_inputs = to_model_inputs(encoded, model.device)

        do_sample = self.temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["top_p"] = self.top_p
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = model.generate(**model_inputs, **gen_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        parse_success = True
        parse_error = ""
        raw_tags: List[str] = []

        try:
            parsed = extract_json_object(raw_text)
            raw_tags = parsed.get("final_tags", []) or []
            if not isinstance(raw_tags, list):
                raw_tags = []
        except Exception as exc:
            parse_success = False
            parse_error = str(exc)

        pred_tags, outside_tags = normalize_tags(raw_tags, self.allowed_set)

        pred_set = set(pred_tags)
        req_set = set(required_gold)
        blocked_set = set(blocked_gold)

        return {
            "query_id": q_id,
            "query": query,
            "required_tags": required_gold,
            "blocked_tags": blocked_gold,
            "pred_tags": pred_tags,
            "outside_tags_raw": outside_tags,
            "parse_success": parse_success,
            "parse_error": parse_error,
            "required_exact_cover": bool(req_set and req_set.issubset(pred_set)),
            "blocked_clean": bool(not blocked_set or pred_set.isdisjoint(blocked_set)),
            "raw_response": raw_text,
        }

    def _build_report(self, run_data: Sequence[Dict[str, object]], adapter_path: Optional[Path]) -> Dict[str, object]:
        total_queries = len(run_data)
        parse_success = sum(1 for row in run_data if row.get("parse_success"))
        parse_success_rate = parse_success / total_queries if total_queries else 0.0

        pred_sets: List[set[str]] = []
        required_sets: List[set[str]] = []
        required_query_count = 0
        required_cover_hits = 0

        blocked_query_count = 0
        blocked_clean_hits = 0

        raw_outside_count = 0
        raw_pred_tag_count = 0

        for row in run_data:
            pred_tags = row.get("pred_tags") or []
            required_tags = row.get("required_tags") or []
            blocked_tags = row.get("blocked_tags") or []
            outside = row.get("outside_tags_raw") or []

            pred_set = set(pred_tags)
            required_set = set(required_tags)
            blocked_set = set(blocked_tags)

            if required_set:
                required_query_count += 1
                required_cover_hits += int(required_set.issubset(pred_set))
                pred_sets.append(pred_set)
                required_sets.append(required_set)

            if blocked_set:
                blocked_query_count += 1
                blocked_clean_hits += int(pred_set.isdisjoint(blocked_set))

            raw_outside_count += len(outside)
            raw_pred_tag_count += len([str(t).strip() for t in pred_tags if str(t).strip()])

        required_micro = micro_prf(pred_sets, required_sets) if pred_sets else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        required_macro = macro_prf(pred_sets, required_sets, self.all_tags) if pred_sets else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        required_em = exact_match_rate(pred_sets, required_sets) if pred_sets else 0.0

        outside_rate = raw_outside_count / raw_pred_tag_count if raw_pred_tag_count else 0.0

        return {
            "total_queries": total_queries,
            "parse_success": parse_success,
            "parse_success_rate": parse_success_rate,
            "required_query_count": required_query_count,
            "required_exact_cover_hits": required_cover_hits,
            "required_exact_cover_rate": required_cover_hits / required_query_count if required_query_count else 0.0,
            "required_micro_precision": float(required_micro.get("precision", 0.0)),
            "required_micro_recall": float(required_micro.get("recall", 0.0)),
            "required_micro_f1": float(required_micro.get("f1", 0.0)),
            "required_macro_precision": float(required_macro.get("precision", 0.0)),
            "required_macro_recall": float(required_macro.get("recall", 0.0)),
            "required_macro_f1": float(required_macro.get("f1", 0.0)),
            "required_exact_match_rate": required_em,
            "blocked_query_count": blocked_query_count,
            "blocked_clean_hits": blocked_clean_hits,
            "blocked_clean_rate": blocked_clean_hits / blocked_query_count if blocked_query_count else 0.0,
            "raw_pred_tag_count": raw_pred_tag_count,
            "raw_outside_taxonomy_tag_count": raw_outside_count,
            "raw_outside_taxonomy_rate": outside_rate,
            "base_model": self.base_model,
            "adapter_path": str(adapter_path) if adapter_path else "",
        }

    def generate_run(
        self,
        queries_config: List[Dict[str, object]],
        run_name: str,
        output_dir: Path,
        adapter_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        print(f"\n🚀 [Batch] Starting Experiment: {run_name}")
        if adapter_path is None:
            print("   Using model: BASE")
        else:
            print(f"   Using model: TUNED ({adapter_path})")

        output_dir.mkdir(parents=True, exist_ok=True)
        run_path = output_dir / f"{run_name}.json"
        report_path = output_dir / f"{run_name}_report.json"

        queries = list(queries_config)
        if self.max_samples > 0:
            queries = queries[: self.max_samples]

        tokenizer, model = self._load_model_and_tokenizer(adapter_path=adapter_path)

        run_data: List[Dict[str, object]] = []
        for idx, q_item in enumerate(queries, start=1):
            qid = str(q_item.get("id", f"q{idx}"))
            print(f"   - Processing query: {qid}")
            try:
                row = self._run_single_query(tokenizer=tokenizer, model=model, q_item=q_item)
                run_data.append(row)
            except Exception as query_err:
                print(f"     ⚠️ Error processing query {qid}: {query_err}")
                run_data.append(
                    {
                        "query_id": qid,
                        "query": str(q_item.get("query", "")),
                        "required_tags": [],
                        "blocked_tags": [],
                        "pred_tags": [],
                        "outside_tags_raw": [],
                        "parse_success": False,
                        "parse_error": str(query_err),
                        "required_exact_cover": False,
                        "blocked_clean": True,
                        "raw_response": "",
                        "error": str(query_err),
                    }
                )

        report = self._build_report(run_data=run_data, adapter_path=adapter_path)

        with run_path.open("w", encoding="utf-8") as f:
            json.dump(run_data, f, ensure_ascii=False, indent=2)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✅ [{run_name}] Run complete! Saved to {run_path}")
        print(f"✅ [{run_name}] Report saved to {report_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "run_path": str(run_path),
            "report_path": str(report_path),
            "report": report,
        }


def resolve_queries_path(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data/experiments/queries.json",
        repo_root / "queries.json",
        repo_root / "data/raw_repo_data/experiments/queries.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    checked = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"queries.json not found. Checked:\n  - {checked}")


def resolve_adapter_path(repo_root: Path, adapter_root: Path) -> Optional[Path]:
    full = adapter_root if adapter_root.is_absolute() else (repo_root / adapter_root)
    full = full.resolve()
    if not full.exists():
        return None

    required = ["adapter_config.json", "adapter_model.safetensors"]
    if all((full / x).exists() for x in required):
        return full

    checkpoints = []
    for child in full.iterdir():
        if not child.is_dir():
            continue
        if all((child / x).exists() for x in required):
            m = re.match(r"checkpoint-(\d+)$", child.name)
            step = int(m.group(1)) if m else -1
            checkpoints.append((step, child.stat().st_mtime, child))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return checkpoints[0][2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen tag extraction benchmark using existing experiment template style")
    parser.add_argument("--queries-path", type=Path, default=None)
    parser.add_argument("--tags-path", type=Path, default=Path("data/all_tags.json"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-root", type=Path, default=Path("data/models/qwen25_7b_tag_lora"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/qwen_tag_runs"))

    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--skip-tuned", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    queries_path = args.queries_path
    if queries_path is None:
        queries_path = resolve_queries_path(repo_root)
    elif not queries_path.is_absolute():
        queries_path = (repo_root / queries_path).resolve()

    tags_path = args.tags_path if args.tags_path.is_absolute() else (repo_root / args.tags_path).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir).resolve()

    if not queries_path.exists():
        raise FileNotFoundError(f"queries file not found: {queries_path}")
    if not tags_path.exists():
        raise FileNotFoundError(f"tags file not found: {tags_path}")

    queries = load_json(queries_path)
    if not isinstance(queries, list):
        raise ValueError("queries file must be a JSON array")

    all_tags = load_json(tags_path)
    if not isinstance(all_tags, list) or not all(isinstance(t, str) for t in all_tags):
        raise ValueError("tags file must be a JSON array of strings")

    generator = QwenTagRunGenerator(
        all_tags=all_tags,
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=not args.disable_4bit,
        bf16=args.bf16,
        max_samples=args.max_samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    base_result = generator.generate_run(
        queries_config=queries,
        run_name="qwen_base",
        output_dir=output_dir,
        adapter_path=None,
    )

    if args.skip_tuned:
        print("\n[done] base run completed (skip tuned)")
        print(f"  base_report: {base_result['report_path']}")
        return

    adapter_path = resolve_adapter_path(repo_root=repo_root, adapter_root=args.adapter_root)
    if adapter_path is None:
        raise FileNotFoundError(
            f"No LoRA adapter files found under: {(repo_root / args.adapter_root).resolve()}"
        )

    print(f"\n[info] Using tuned adapter: {adapter_path}")

    tuned_result = generator.generate_run(
        queries_config=queries,
        run_name="qwen_tuned",
        output_dir=output_dir,
        adapter_path=adapter_path,
    )

    comparison = build_comparison(
        base_report=base_result["report"],
        tuned_report=tuned_result["report"],
    )
    comparison_path = output_dir / "qwen_base_vs_tuned_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("\n🎉 Qwen tag benchmark finished!")
    print(f"  base_report: {base_result['report_path']}")
    print(f"  tuned_report: {tuned_result['report_path']}")
    print(f"  comparison: {comparison_path}")


if __name__ == "__main__":
    main()
