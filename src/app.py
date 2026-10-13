
"""
Frontend Gradio — Text-to-SQL em português.

Uso:
    python src/app.py --adapter outputs/run1/final
    python src/app.py --adapter outputs/run1/final --share
"""

import argparse
import html

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


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #6254ed;
    --primary-light: #edeaff;
    --text: #171944;
    --muted: #777da7;
    --border: #e6e4ff;
    --surface: rgba(255,255,255,.78);
}

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    max-width: 1440px !important;
    margin: auto !important;
    background:
        radial-gradient(ellipse at 50% 0%, #ffffff 0%, transparent 60%),
        linear-gradient(135deg, #f5f4ff, #e9eaff) !important;
}

.gradio-container,
.gradio-container * {
    box-sizing: border-box;
}

#app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 22px;
    margin-bottom: 10px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
    color: var(--text);
    font-size: 19px;
    font-weight: 700;
}

.brand-icon {
    width: 43px;
    height: 43px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #5143ed;
    background: #e8e4ff;
    border-radius: 13px;
    font-size: 23px;
}

#landing {
    padding: 24px 12px 30px;
}

.welcome {
    text-align: center;
    margin: 4px auto 40px;
    max-width: 740px;
}

.welcome .greeting {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 14px;
}

.welcome h1 {
    color: var(--text);
    font-size: clamp(25px, 3vw, 34px);
    line-height: 1.25;
    font-weight: 750;
    letter-spacing: -.8px;
    margin: 0 0 12px;
}

.welcome p {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.8;
    margin: 0;
}

#suggestions {
    max-width: 1080px;
    margin: 0 auto 48px;
}

#suggestions button {
    text-align: left !important;
    min-height: 100px;
    height: 100%;
    padding: 17px 15px !important;
    white-space: normal !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    color: #3d456f !important;
    box-shadow: 0 5px 20px rgba(91, 79, 190, .05);
    transition: .2s ease;
}

#suggestions button:hover {
    border-color: #a9a0ff !important;
    background: white !important;
    transform: translateY(-2px);
}

#suggestions button span {
    white-space: normal !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}

#question-bar {
    max-width: 650px;
    margin: 0 auto;
    padding: 9px 12px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: rgba(255,255,255,.9);
    box-shadow: 0 8px 28px rgba(82, 73, 180, .09);
}

#question-bar textarea {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    resize: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    color: var(--text) !important;
}

#question-bar textarea:focus {
    box-shadow: none !important;
}

#send-btn {
    min-width: 45px !important;
    height: 43px !important;
    border-radius: 15px !important;
    background: #ded9ff !important;
    color: #5143ed !important;
    border: none !important;
    font-size: 22px !important;
}

#send-btn:hover {
    background: #c9c1ff !important;
}

#answer-panel {
    max-width: 820px;
    margin: 0 auto;
    padding: 14px 16px 18px;
    background: rgba(255,255,255,.76);
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(74, 67, 165, .06);
}

#question-bubble {
    display: flex;
    justify-content: flex-end;
    margin: 0 auto 18px;
    max-width: 820px;
}

.question-bubble-inner {
    max-width: 80%;
    padding: 14px 18px;
    border-radius: 15px;
    background: #e3dfff;
    color: #242052;
    font-size: 14px;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.section-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 700;
    margin: 5px 0 10px;
}

#answer-panel .block,
#answer-panel .form {
    border-color: var(--border) !important;
}

#answer-panel .cm-editor,
#answer-panel .code_editor {
    border-radius: 11px !important;
}

#answer-panel .wrap,
#answer-panel table {
    border-radius: 10px;
}

#answer-panel .accordion {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,.65) !important;
    margin-top: 14px;
}

#status {
    margin-top: 12px;
    padding: 9px 12px;
    border-radius: 9px;
    background: #f0faf6;
    color: #269773;
    font-size: 12px;
}

#new-question {
    margin-top: 18px;
    border-radius: 12px !important;
}

footer {
    text-align: center;
    color: #8589aa;
    font-size: 11px;
    padding: 25px 0 12px;
}

@media (max-width: 700px) {
    #landing {
        padding: 12px 4px;
    }

    .welcome {
        margin-bottom: 25px;
    }

    #suggestions {
        margin-bottom: 25px;
    }

    #answer-panel {
        padding: 12px 9px;
    }

    .brand {
        font-size: 16px;
    }
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        default=None,
        help="Pasta do adaptador LoRA (vazio = modelo base)",
    )
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument(
        "--share",
        action="store_true",
        help="Gera link público (útil no Colab)",
    )
    args = ap.parse_args()

    # Carrega o modelo uma única vez.
    tok, model = load_model(
        args.model,
        args.adapter,
        not args.no_4bit,
    )

    def responder(pergunta, temperature, top_p, max_tokens):
        pergunta = pergunta or ""

        if not pergunta.strip():
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        pergunta_html = (
            '<div class="question-bubble-inner">'
            + html.escape(pergunta)
            + "</div>"
        )

        try:
            sql = generate_sql(
                tok,
                model,
                [pergunta],
                batch_size=1,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )[0]

            sql = clean_sql(sql)
            ok, resultado = run_sql(sql)

            if not ok:
                return (
                    pergunta_html,
                    sql,
                    pd.DataFrame(),
                    f"Erro ao executar a SQL: {resultado}",
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )

            cols, rows = resultado
            tabela_df = pd.DataFrame(rows, columns=cols)

            return (
                pergunta_html,
                sql,
                tabela_df,
                f"✓ {len(rows)} linha(s) retornada(s).",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        except Exception as exc:
            return (
                pergunta_html,
                "",
                pd.DataFrame(),
                f"Erro ao gerar ou executar a consulta: {exc}",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

    def nova_pergunta():
        return (
            "",
            "",
            pd.DataFrame(),
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    with gr.Blocks(
        title="Text-to-SQL em português",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
            radius_size="lg",
        ),
        css=CSS,
    ) as demo:

        # Cabeçalho
        gr.HTML("""
        <header id="app-header">
            <div class="brand">
                <div class="brand-icon">▤</div>
                <span>Text-to-SQL em português</span>
            </div>
            <div style="
                width:40px;height:40px;border-radius:50%;
                border:1px solid #e9e6ff;background:#fff;
                display:flex;align-items:center;justify-content:center;
                color:#6254ed;font-size:20px;">
                ☼
            </div>
        </header>
        """)

        # Tela inicial
        landing = gr.Column(visible=True, elem_id="landing")

        with landing:
            gr.HTML("""
            <div class="welcome">
                <div class="greeting">Olá! 👋🏻</div>
                <h1>O que você gostaria de saber?</h1>
                <p>
                    Pergunte em linguagem natural sobre a loja fictícia;<br>
                    o modelo Qwen2.5 + LoRA gera a consulta SQL e ela é
                    executada no banco.
                </p>
            </div>
            """)

            with gr.Row(elem_id="suggestions"):
                botoes = [
                    gr.Button("♙\n\nQuantos clientes são do estado SP?"),
                    gr.Button("✉\n\nQuais os 5 produtos mais caros?"),
                    gr.Button("▥\n\nQual o faturamento total dos pedidos entregues?"),
                    gr.Button("♧\n\nQuais os 5 clientes que mais gastaram?"),
                    gr.Button("☷\n\nQuantos pedidos existem por status?"),
                ]

            with gr.Row(elem_id="question-bar"):
                pergunta = gr.Textbox(
                    label="",
                    placeholder="Faça sua pergunta aqui...",
                    lines=1,
                    max_lines=4,
                    show_label=False,
                    scale=9,
                    container=False,
                )
                btn = gr.Button(
                    "➜",
                    variant="primary",
                    scale=1,
                    elem_id="send-btn",
                )

            with gr.Accordion("⚙ Parâmetros de geração", open=False):
                temperature = gr.Slider(
                    0.0, 1.5, value=0.0, step=0.1,
                    label="Temperature (0 = determinístico)",
                )
                top_p = gr.Slider(
                    0.1, 1.0, value=0.9, step=0.05,
                    label="Top-p",
                )
                max_tokens = gr.Slider(
                    32, 400, value=200, step=8,
                    label="Máximo de tokens",
                )

        # Tela da resposta
        answer = gr.Column(visible=False)

        with answer:
            question_html = gr.HTML(elem_id="question-bubble")

            with gr.Column(elem_id="answer-panel"):
                gr.HTML('<div class="section-title">〈/〉 SQL gerada</div>')

                sql_out = gr.Code(
                    label="",
                    language="sql",
                    interactive=False,
                )

                # O esquema aparece junto com a resposta,
                # em vez de ocupar espaço na tela inicial.
                with gr.Accordion("▤  Esquema do banco", open=True):
                    gr.Code(
                        value=SCHEMA,
                        language="sql",
                        label="",
                        interactive=False,
                    )

                gr.HTML('<div class="section-title">▦ Resultado da consulta</div>')

                tabela = gr.Dataframe(
                    label="",
                    interactive=False,
                    wrap=True,
                )

                status = gr.Markdown(elem_id="status")

            with gr.Row():
                nova = gr.Button(
                    "＋ Nova pergunta",
                    variant="secondary",
                    elem_id="new-question",
                )

            # Os parâmetros também ficam acessíveis após a resposta.
            with gr.Accordion("⚙ Parâmetros de geração", open=False):
                # Os mesmos componentes da tela inicial são reutilizados
                # no callback; não é necessário duplicá-los aqui.
                gr.Markdown(
                    "Os parâmetros podem ser ajustados na tela inicial "
                    "antes de enviar uma nova pergunta."
                )

        saidas = [
            question_html,
            sql_out,
            tabela,
            status,
            landing,
            answer,
            landing,
            answer,
        ]

        btn.click(
            responder,
            inputs=[pergunta, temperature, top_p, max_tokens],
            outputs=saidas,
        )

        pergunta.submit(
            responder,
            inputs=[pergunta, temperature, top_p, max_tokens],
            outputs=saidas,
        )

        for botao, exemplo in zip(botoes, EXEMPLOS):
            botao.click(
                fn=lambda q=exemplo: q,
                outputs=pergunta,
            )

        nova.click(
            nova_pergunta,
            outputs=[
                pergunta,
                sql_out,
                tabela,
                status,
                landing,
                answer,
                landing,
                answer,
            ],
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
