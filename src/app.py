"""Frontend Gradio: o usuário escreve a pergunta, o modelo gera o SQL e o resultado é exibido.

Uso:  python src/app.py --adapter outputs/run1/final
      (no Colab: python src/app.py --adapter ... --share)
"""
import argparse

import gradio as gr
import pandas as pd

from common import MODEL_ID, SCHEMA, clean_sql, run_sql
from evaluate_model import generate_sql, load_model

EXEMPLOS = [
    "Quantos clientes são do estado SP?",
    "Quais os 5 produtos mais caros?",
    "Qual o faturamento total dos pedidos entregues?",
    "Quais os 5 clientes que mais gastaram?",
    "Quantos pedidos existem por status?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="pasta do adaptador LoRA (vazio = modelo base)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--share", action="store_true", help="gera link público (útil no Colab)")
    args = ap.parse_args()

    tok, model = load_model(args.model, args.adapter, not args.no_4bit)

    def responder(pergunta, temperature, top_p, max_tokens):
        if not pergunta.strip():
            return "", pd.DataFrame(), "Digite uma pergunta."
        sql = generate_sql(tok, model, [pergunta], batch_size=1, max_new_tokens=int(max_tokens),
                           temperature=float(temperature), top_p=float(top_p))[0]
        ok, res = run_sql(clean_sql(sql))
        if not ok:
            return sql, pd.DataFrame(), f"Erro ao executar a SQL: {res}"
        cols, rows = res
        return sql, pd.DataFrame(rows, columns=cols), f"{len(rows)} linha(s) retornada(s)."

    with gr.Blocks(title="Text-to-SQL em português") as demo:
        gr.Markdown("# Text-to-SQL em português\nPergunte em linguagem natural sobre a loja fictícia; "
                    "o modelo (Qwen2.5 + LoRA) gera a consulta SQL e ela é executada no banco.")
        with gr.Row():
            with gr.Column(scale=2):
                pergunta = gr.Textbox(label="Pergunta", lines=3, placeholder="Ex.: Quais os 5 produtos mais caros?")
                with gr.Accordion("Parâmetros de geração", open=False):
                    temperature = gr.Slider(0.0, 1.5, 0.0, step=0.1, label="Temperature (0 = determinístico)")
                    top_p = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-p")
                    max_tokens = gr.Slider(32, 400, 200, step=8, label="Máximo de tokens")
                btn = gr.Button("Gerar SQL", variant="primary")
                gr.Examples(EXEMPLOS, inputs=pergunta)
            with gr.Column(scale=3):
                sql_out = gr.Code(label="SQL gerada", language="sql")
                status = gr.Markdown()
                tabela = gr.Dataframe(label="Resultado da consulta")
        with gr.Accordion("Esquema do banco", open=False):
            gr.Code(SCHEMA, language="sql")
        btn.click(responder, [pergunta, temperature, top_p, max_tokens], [sql_out, tabela, status])
        pergunta.submit(responder, [pergunta, temperature, top_p, max_tokens], [sql_out, tabela, status])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
