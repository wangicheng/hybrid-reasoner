import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-7B-Instruct with QLoRA for tag extraction.")

    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen25_7b_tag_lora"))

    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    return parser.parse_args()


def build_bnb_config(use_4bit: bool, use_bf16: bool) -> BitsAndBytesConfig | None:
    if not use_4bit:
        return None

    compute_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def gpu_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(checker):
        return bool(checker())
    return False


def build_text(example: dict, tokenizer: AutoTokenizer) -> dict:
    messages = example.get("messages")
    if isinstance(messages, list):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}
        except Exception:
            pass

    # Fallback for compatibility with manually created rows.
    system_prompt = str(example.get("system_prompt", "")).strip()
    user_prompt = str(example.get("user_prompt", "")).strip()
    assistant = str(example.get("assistant", "")).strip()
    text = (
        "<|im_start|>system\n"
        f"{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant}\n"
        "<|im_end|>"
    )
    return {"text": text}


def tokenize_for_causal_lm(example: dict, tokenizer: AutoTokenizer, max_seq_length: int) -> dict:
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    labels = []
    for token_id, mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
        labels.append(token_id if mask == 1 else -100)
    tokenized["labels"] = labels
    return tokenized


def main() -> None:
    args = parse_args()

    if not args.train_file.exists():
        raise FileNotFoundError(f"Train file not found: {args.train_file}")
    if not args.val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {args.val_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(use_4bit=args.load_in_4bit, use_bf16=args.bf16)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.val_file),
        },
    )

    def to_chat_text(example: dict) -> dict:
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_dataset = dataset["train"].map(to_chat_text, remove_columns=dataset["train"].column_names)
    val_dataset = dataset["validation"].map(to_chat_text, remove_columns=dataset["validation"].column_names)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    use_bf16 = bool(args.bf16 and gpu_supports_bf16())
    use_fp16 = bool(not use_bf16)

    if args.bf16 and not use_bf16:
        print("[warn] bf16 requested but not supported on current GPU; falling back to fp16.")

    training_args_kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": False,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": "none",
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": "paged_adamw_8bit",
        "remove_unused_columns": False,
    }

    # Transformers renamed this argument in newer versions.
    training_args_sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_args_sig:
        training_args_kwargs["evaluation_strategy"] = "steps"
    else:
        training_args_kwargs["eval_strategy"] = "steps"

    model = prepare_model_for_kbit_training(model) if args.load_in_4bit else model
    model = get_peft_model(model, peft_config)

    train_dataset = train_dataset.map(
        lambda x: tokenize_for_causal_lm(x, tokenizer, args.max_seq_length),
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_for_causal_lm(x, tokenizer, args.max_seq_length),
        remove_columns=val_dataset.column_names,
    )

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=default_data_collator,
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics_path = args.output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    config_path = args.output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    print("[done] training finished")
    print(f"  output_dir: {args.output_dir}")
    print(f"  metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
