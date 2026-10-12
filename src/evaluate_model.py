"""Avalia o modelo no conjunto de teste (accuracy por execução).

Baseline (sem fine-tuning):  python src/evaluate_model.py --name baseline
Com adaptador LoRA:          python src/evaluate_model.py --name run1 --adapter outputs/run1/final
"""
import argparse
import csv
import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig

from common import (MODEL_ID, ROOT, build_messages, clean_sql, load_causal_lm,
                    normalize_rows, run_sql, supports_bf16)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_model(model_id, adapter=None, use_4bit=True):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if supports_bf16() else torch.float16
    quant = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        if use_4bit else None
    )
    model = load_causal_lm(model_id, quant, dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def generate_sql(tok, model, questions, batch_size=16, max_new_tokens=200,
                 temperature=0.0, top_p=0.9):
    outs = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        prompts = [
            tok.apply_chat_template(build_messages(q), tokenize=False, add_generation_prompt=True)
            for q in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        gen = model.generate(**enc, **kwargs)
        for g in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(clean_sql(tok.decode(g, skip_special_tokens=True)))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--split", default="test")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    rows = load_jsonl(ROOT / "data" / f"{args.split}.jsonl")
    train_sqls = {r["sql"] for r in load_jsonl(ROOT / "data" / "train.jsonl")}

    tok, model = load_model(args.model, args.adapter, not args.no_4bit)
    preds = generate_sql(tok, model, [r["question"] for r in rows],
                         temperature=args.temperature, top_p=args.top_p,
                         max_new_tokens=args.max_new_tokens)

    n = len(rows)
    valid = exact = correct = 0
    new_total = new_correct = 0
    details = []
    for r, p in zip(rows, preds):
        ok_g, gold = run_sql(r["sql"])
        ok_p, pred = run_sql(p)
        valid += ok_p
        exact += p.strip().lower() == r["sql"].strip().lower()
        hit = bool(ok_g and ok_p and normalize_rows(gold[1]) == normalize_rows(pred[1]))
        correct += hit
        unseen = r["sql"] not in train_sqls
        if unseen:
            new_total += 1
            new_correct += hit
        details.append({"question": r["question"], "gold_sql": r["sql"], "pred_sql": p,
                        "sql_valida": ok_p, "correto": hit, "sql_nao_vista_no_treino": unseen})

    metrics = {
        "run": args.name,
        "n": n,
        "exec_accuracy": round(correct / n, 4),
        "sql_valida": round(valid / n, 4),
        "exact_match": round(exact / n, 4),
        "exec_acc_sql_inedita": round(new_correct / new_total, 4) if new_total else None,
        "n_sql_inedita": new_total,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    tag = f"{args.name}_T{args.temperature}"
    with open(res / f"pred_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    csv_path = res / "avaliacao.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics))
        if new:
            w.writeheader()
        w.writerow(metrics)


if __name__ == "__main__":
    main()