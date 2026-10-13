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
    width: 100% !important;
    max-width: 100% !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    background:
        radial-gradient(ellipse at 50% 0%, #ffffff 0%, transparent 60%),
        linear-gradient(135deg, #f5f4ff, #e9eaff) !important;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.gradio-container,
.gradio-container * {
    box-sizing: border-box;
}

#app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 32px;
    width: 100%;
    border-bottom: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(8px);
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
    color: var(--text);
    font-size: 20px;
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

.header-links {
    display: flex;
    align-items: center;
    gap: 15px;
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
}

#main-content {
    flex: 1;
    overflow-y: auto;
    padding: 24px 20px;
    max-width: 1000px;
    margin: 0 auto;
    width: 100%;
}

.welcome {
    text-align: center;
    margin: 10px auto 30px;
    max-width: 740px;
}

.welcome .greeting {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 10px;
}

.welcome h1 {
    color: var(--text);
    font-size: clamp(24px, 3vw, 32px);
    line-height: 1.25;
    font-weight: 750;
    margin: 0 0 10px;
}

.welcome p {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
    margin: 0;
}

#suggestions {
    width: 100%;
    margin: 0 auto 30px;
}

#suggestions button {
    text-align: left !important;
    min-height: 90px;
    height: 100%;
    padding: 15px !important;
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

#chat-history {
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin-bottom: 20px;
}

.turn-block {
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 8px 30px rgba(74, 67, 165, .05);
}

#question-bubble {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 15px;
}

.question-bubble-inner {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 15px;
    background: #e3dfff;
    color: #242052;
    font-size: 14px;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.section-title {
    color: var(--text);
    font-size: 13px;
    font-weight: 700;
    margin: 10px 0 6px;
}

#question-container {
    padding: 15px 20px;
    background: rgba(255, 255, 255, 0.85);
    border-top: 1px solid var(--border);
    backdrop-filter: blur(8px);
    width: 100%;
}

#question-bar {
    max-width: 800px;
    margin: 0 auto;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: white;
    box-shadow: 0 8px 28px rgba(82, 73, 180, .08);
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

#send-btn {
    min-width: 42px !important;
    height: 40px !important;
    border-radius: 14px !important;
    background: #ded9ff !important;
    color: #5143ed !important;
    border: none !important;
    font-size: 18px !important;
}

#send-btn:hover {
    background: #c9c1ff !important;
}

#status {
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background: #f0faf6;
    color: #269773;
    font-size: 12px;
}

footer {
    display: none !important;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    tok, model = load_model(args.model, args.adapter, not args.no_4bit)

    def responder(pergunta, temperature, top_p, max_tokens, historico_atual):
        pergunta = (pergunta or "").strip()
        if not pergunta:
            return historico_atual, "", gr.update(value="➜")

        # Mostra spinner de carregamento no botão
        yield historico_atual, "", gr.update(value="⏳")

        pergunta_html = (
            f'<div id="question-bubble"><div class="question-bubble-inner">'
            f"{html.escape(pergunta)}"
            f"</div></div>"
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
                conteudo_resposta = (
                    f"{pergunta_html}"
                    f'<div class="turn-block">'
                    f'<div class="section-title">〈/〉 SQL gerada</div>'
                    f"<pre><code>{html.escape(sql)}</code></pre>"
                    f'<div style="color: red; font-size: 13px; margin-top: 10px;">Erro ao executar a SQL: {resultado}</div>'
                    f"</div>"
                )
            else:
                cols, rows = resultado
                tabela_df = pd.DataFrame(rows, columns=cols)
                # Formata tabela HTML simples para o histórico do chat contínuo
                tabela_html = tabela_df.to_html(classes="dataframe", index=False, border=0)

                conteudo_resposta = (
                    f"{pergunta_html}"
                    f'<div class="turn-block">'
                    f'<div class="section-title">〈/〉 SQL gerada</div>'
                    f"<pre style='background:#f4f3ff; padding:10px; border-radius:8px;'><code>{html.escape(sql)}</code></pre>"
                    f'<div class="section-title">▤ Esquema do banco</div>'
                    f"<pre style='background:#f4f3ff; padding:10px; border-radius:8px; font-size:11px;'><code>{html.escape(SCHEMA)}</code></pre>"
                    f'<div class="section-title">▦ Resultado da consulta ({len(rows)} linha(s))</div>'
                    f"<div style='overflow-x:auto; margin-top:5px;'>{tabela_html}</div>"
                    f'<div id="status">✓ {len(rows)} linha(s) retornada(s).</div>'
                    f"</div>"
                )

            novo_historico = (historico_atual or "") + "<br>" + conteudo_resposta
            yield novo_historico, "", gr.update(value="➜")

        except Exception as exc:
            erro_html = (
                f"{pergunta_html}"
                f'<div class="turn-block"><div style="color: red;">Erro: {exc}</div></div>'
            )
            novo_historico = (historico_atual or "") + "<br>" + erro_html
            yield novo_historico, "", gr.update(value="➜")

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

        # Cabeçalho com Parâmetros de Geração e links no topo
        with gr.Row(elem_id="app-header"):
            gr.HTML("""
            <div class="brand">
                <div class="brand-icon">▤</div>
                <span>Text-to-SQL em português</span>
            </div>
            """)
            with gr.Row(elem_id="", scale=0, custom_css="gap:15px; align-items:center;"):
                gr.HTML('<span style="cursor:pointer; color:var(--muted);">Usar via API 🚀</span>')
                gr.HTML('<span style="color:var(--border);">·</span>')
                gr.HTML('<span style="cursor:pointer; color:var(--muted);">Construído com Gradio 🔶</span>')
                gr.HTML('<span style="color:var(--border);">·</span>')
                
        with gr.Accordion("⚙ Parâmetros de Geração & Configurações", open=False):
            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="Temperature (0 = determinístico)")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                max_tokens = gr.Slider(32, 400, value=200, step=8, label="Máximo de tokens")

        # Conteúdo Principal estilo Chat
        with gr.Column(elem_id="main-content"):
            landing_intro = gr.HTML("""
            <div class="welcome">
                <div class="greeting">Olá! 👋🏻</div>
                <h1>O que você gostaria de saber?</h1>
                <p>
                    Pergunte em linguagem natural sobre a loja fictícia;<br>
                    o modelo Qwen2.5 + LoRA gera a consulta SQL e ela é executada no banco.
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

            chat_history = gr.HTML(value="", elem_id="chat-history")

        # Barra de perguntas fixa na parte inferior (estilo chat contínuo)
        with gr.Column(elem_id="question-container"):
            with gr.Row(elem_id="question-bar"):
                pergunta = gr.Textbox(
                    label="",
                    placeholder="Faça sua nova pergunta aqui...",
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

        # Eventos
        inputs_list = [pergunta, temperature, top_p, max_tokens, chat_history]
        outputs_list = [chat_history, pergunta, btn]

        btn.click(responder, inputs=inputs_list, outputs=outputs_list)
        pergunta.submit(responder, inputs=inputs_list, outputs=outputs_list)

        for botao, exemplo in zip(botoes, EXEMPLOS):
            botao.click(fn=lambda q=exemplo: q, outputs=pergunta)

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()