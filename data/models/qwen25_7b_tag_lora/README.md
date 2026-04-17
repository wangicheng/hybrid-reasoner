# Qwen LoRA Adapter Drop Folder

Put your downloaded LoRA adapter files in this folder before running the test script.

Required files:

- `adapter_config.json`
- `adapter_model.safetensors`

Optional files (recommended):

- `tokenizer_config.json`
- `special_tokens_map.json`
- `tokenizer.json`

Run test command from repository root:

```bash
python scripts/run_qwen_base_tuned_test.py
```

If you want to run base model only when adapter files are not ready yet:

```bash
python scripts/run_qwen_base_tuned_test.py --allow-missing-adapter
```
