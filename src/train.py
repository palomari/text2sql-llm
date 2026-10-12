"""Fine-tuning QLoRA para Text-to-SQL em português.

Exemplo:
    python src/train.py --name run1 --r 8 --lr 2e-4 --epochs 3
"""
import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from common import MODEL_ID, ROOT, build_messages, supports_bf16


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_text(tokenizer, rows):
    texts = [
        tokenizer.apply_chat_template(
            build_messages(r["question"], r["sql"]), tokenize=False
        )
        for r in rows
    ]
    return Dataset.from_dict({"text": texts})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="nome da execução (ex.: run1)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--r", type=int, default=16, help="rank do LoRA")
    ap.add_argument("--alpha", type=int, default=None, help="padrão: 2*r")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--no_4bit", action="store_true", help="carrega em fp16 (sem quantização)")
    ap.add_argument("--out_dir", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "GPU não encontrada! Verifique o driver/CUDA."
    print("GPU:", torch.cuda.get_device_name(0))
    use_bf16 = supports_bf16()
    print("Precisão:", "bf16" if use_bf16 else "fp16", "| compute capability:",
          torch.cuda.get_device_capability(0))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map={"": 0},
    )

    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha or 2 * args.r,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    train_ds = to_text(tokenizer, load_jsonl(ROOT / "data" / "train.jsonl"))
    val_ds = to_text(tokenizer, load_jsonl(ROOT / "data" / "val.jsonl"))

    out = Path(args.out_dir) / args.name

    # warmup em passos (em vez de warmup_ratio, que foi removido nas versões novas)
    steps_per_epoch = max(1, len(train_ds) // (args.batch * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    cfg_kwargs = dict(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total_steps)),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        max_length=args.max_len,
        dataset_text_field="text",
        report_to="none",
        seed=42,
    )
    # Compatibilidade entre versões do trl/transformers: renomeia/descarta argumentos
    # que a versão instalada não conhece (e avisa, para não ficar silencioso).
    valid = {f.name for f in dataclasses.fields(SFTConfig)}
    if "max_length" not in valid and "max_seq_length" in valid:
        cfg_kwargs["max_seq_length"] = cfg_kwargs.pop("max_length")
    dropped = [k for k in cfg_kwargs if k not in valid]
    if dropped:
        print(f"AVISO: argumentos ignorados nesta versão do trl: {dropped}")
    cfg = SFTConfig(**{k: v for k, v in cfg_kwargs.items() if k in valid})

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    minutes = (time.time() - t0) / 60
    eval_metrics = trainer.evaluate()

    final = out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    row = {
        "run": args.name,
        "lora_r": args.r,
        "lora_alpha": args.alpha or 2 * args.r,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_efetivo": args.batch * args.grad_accum,
        "train_loss": round(result.training_loss, 4),
        "val_loss": round(eval_metrics["eval_loss"], 4),
        "tempo_min": round(minutes, 1),
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    print(row)

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    csv_path = res_dir / "experimentos.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"Adaptador salvo em {final}")


if __name__ == "__main__":
    main()