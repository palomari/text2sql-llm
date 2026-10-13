"""
Frontend Gradio — Text-to-SQL em português.

Uso:
    python src/app.py --adapter outputs/run1/final
    python src/app.py --adapter outputs/run1/final --share     # link público (Colab)
    python src/app.py --demo                                    # só a interface, sem modelo/GPU
"""

import argparse
import html
import inspect
import time

import gradio as gr
import pandas as pd

from common import MODEL_ID, SCHEMA, clean_sql, run_sql

TITULO = "Text-to-SQL em português"
ICONE_ENVIAR = "➜"
LIMITE_LINHAS = 200  # mesmo limite usado em run_sql

SUGESTOES = [
    ("♙", "Quantos clientes são do estado SP?"),
    ("✉", "Quais os 5 produtos mais caros?"),
    ("▥", "Qual o faturamento total dos pedidos entregues?"),
    ("♧", "Quais os 5 clientes que mais gastaram?"),
    ("☷", "Quantos pedidos existem por status?"),
]

HEADER_HTML = """
<div class="header-inner">
    <div class="brand">
        <div class="brand-icon">▤</div>
        <span>Text-to-SQL em português</span>
    </div>
    <div class="header-links">
        <span>Qwen2.5 + LoRA</span>
        <span>·</span>
        <span>Construído com Gradio</span>
    </div>
</div>
"""

INTRO_HTML = """
<div class="welcome">
    <div class="greeting">Olá! 👋🏻</div>
    <h1>O que você gostaria de saber?</h1>
    <p>
        Pergunte em linguagem natural sobre a loja fictícia;<br>
        o modelo Qwen2.5 + LoRA gera a consulta SQL e ela é executada no banco.
    </p>
</div>
"""

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
}

.gradio-container,
.gradio-container * {
    box-sizing: border-box;
}

#app-header {
    width: 100%;
    border-bottom: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(8px);
}

#app-header .header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    padding: 18px 32px;
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
    gap: 12px;
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
}

#main-content {
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
    font-weight: 700;
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
    white-space: pre-line !important;
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
    margin-bottom: 20px;
}

.turn {
    margin-bottom: 22px;
}

.turn-block {
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 8px 30px rgba(74, 67, 165, .05);
    color: var(--text);
}

.question-bubble {
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
    margin: 12px 0 6px;
}

.turn-block pre {
    background: #f4f3ff;
    padding: 10px 12px;
    border-radius: 8px;
    margin: 0;
    overflow-x: auto;
    white-space: pre-wrap;
    color: #2b2a5c;
}

.turn-block details summary {
    cursor: pointer;
    color: var(--text);
    font-size: 13px;
    font-weight: 700;
    margin-top: 12px;
}

.turn-block details pre {
    font-size: 11px;
    margin-top: 6px;
}

.tabela-scroll {
    overflow-x: auto;
    margin-top: 5px;
}

table.resultado {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}

table.resultado th,
table.resultado td {
    text-align: left;
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
}

table.resultado th {
    background: #f4f3ff;
    color: var(--text);
    font-weight: 600;
}

.status {
    margin-top: 12px;
    padding: 8px 12px;
    border-radius: 8px;
    background: #f0faf6;
    color: #269773;
    font-size: 12px;
}

.status.erro {
    background: #fff1f1;
    color: #c0392b;
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

footer {
    display: none !important;
}
"""


# --------------------------------------------------------------------------- HTML
def html_pergunta(pergunta):
    return (
        '<div class="question-bubble"><div class="question-bubble-inner">'
        f"{html.escape(pergunta)}"
        "</div></div>"
    )


def html_esquema():
    return (
        "<details><summary>▤ Esquema do banco</summary>"
        f"<pre><code>{html.escape(SCHEMA)}</code></pre></details>"
    )


def html_turno(pergunta, sql, resultado=None, erro=None):
    """Monta o bloco HTML de uma pergunta + resposta. Tudo que vem do usuário ou do
    modelo passa por html.escape, para a página nunca quebrar nem executar HTML alheio."""
    partes = [
        '<div class="turn">',
        html_pergunta(pergunta),
        '<div class="turn-block">',
        '<div class="section-title">〈/〉 SQL gerada</div>',
        f"<pre><code>{html.escape(sql or '(vazia)')}</code></pre>",
    ]
    if erro is not None:
        partes.append(
            f'<div class="status erro">Erro ao executar a SQL: {html.escape(str(erro))}</div>'
        )
        partes.append(html_esquema())
    else:
        cols, rows = resultado
        n = len(rows)
        partes.append(f'<div class="section-title">▦ Resultado da consulta ({n} linha(s))</div>')
        if n == 0:
            partes.append('<div class="status">A consulta executou, mas não retornou linhas.</div>')
        else:
            df = pd.DataFrame(rows, columns=cols)
            tabela = df.to_html(
                classes="resultado",
                index=False,
                border=0,
                float_format=lambda x: f"{x:.2f}",
            )
            partes.append(f'<div class="tabela-scroll">{tabela}</div>')
            aviso = f" (mostrando só as primeiras {LIMITE_LINHAS})" if n >= LIMITE_LINHAS else ""
            partes.append(f'<div class="status">✓ {n} linha(s) retornada(s){aviso}.</div>')
        partes.append(html_esquema())
    partes.append("</div></div>")
    return "".join(partes)


def html_falha(pergunta, exc):
    return (
        '<div class="turn">'
        + html_pergunta(pergunta)
        + '<div class="turn-block"><div class="status erro">'
        + f"Erro ao gerar a resposta: {html.escape(str(exc))}"
        + "</div></div></div>"
    )


# --------------------------------------------------------------------------- interface
def _aceita(func, nome):
    try:
        return nome in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def construir_interface(gerar):
    """Monta a interface. `gerar(pergunta, temperature, top_p, max_tokens) -> sql`.

    Devolve (demo, launch_kwargs). `theme` e `css` ficam no Blocks() nas versões
    do Gradio que os aceitam ali e no launch() nas que os moveram para lá."""

    def responder(pergunta, temperature, top_p, max_tokens, historico_atual):
        historico_atual = historico_atual or ""
        pergunta = (pergunta or "").strip()
        if not pergunta:
            yield historico_atual, "", gr.update(value=ICONE_ENVIAR, interactive=True)
            return

        # bloqueia o botão enquanto o modelo gera (evita enviar duas vezes)
        yield historico_atual, "", gr.update(value="⏳", interactive=False)

        try:
            sql = clean_sql(gerar(pergunta, temperature, top_p, max_tokens))
            ok, res = run_sql(sql)
            if ok:
                bloco = html_turno(pergunta, sql, resultado=res)
            else:
                bloco = html_turno(pergunta, sql, erro=res)
        except Exception as exc:  # noqa: BLE001
            bloco = html_falha(pergunta, exc)

        yield historico_atual + bloco, "", gr.update(value=ICONE_ENVIAR, interactive=True)

    tema = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        radius_size="lg",
    )
    blocks_kwargs = {"title": TITULO}
    launch_kwargs = {}
    for nome, valor in (("theme", tema), ("css", CSS)):
        if _aceita(gr.Blocks.__init__, nome):
            blocks_kwargs[nome] = valor
        elif _aceita(gr.Blocks.launch, nome):
            launch_kwargs[nome] = valor

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(HEADER_HTML, elem_id="app-header")

        with gr.Accordion("⚙ Parâmetros de geração", open=False):
            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.1,
                                        label="Temperature (0 = determinístico)")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                max_tokens = gr.Slider(32, 400, value=200, step=8, label="Máximo de tokens")

        with gr.Column(elem_id="main-content"):
            gr.HTML(INTRO_HTML)
            with gr.Row(elem_id="suggestions"):
                botoes = [gr.Button(f"{icone}\n\n{texto}") for icone, texto in SUGESTOES]
            historico = gr.HTML(value="", elem_id="chat-history")

        with gr.Column(elem_id="question-container"):
            with gr.Row(elem_id="question-bar"):
                pergunta = gr.Textbox(
                    label="Pergunta",
                    placeholder="Faça sua pergunta aqui...",
                    lines=1,
                    max_lines=4,
                    show_label=False,
                    container=False,
                    scale=9,
                )
                btn = gr.Button(ICONE_ENVIAR, variant="primary", scale=1, elem_id="send-btn")

        entradas = [pergunta, temperature, top_p, max_tokens, historico]
        saidas = [historico, pergunta, btn]

        btn.click(responder, inputs=entradas, outputs=saidas)
        pergunta.submit(responder, inputs=entradas, outputs=saidas)
        # cada sugestão envia a própria pergunta direto, sem passar pela caixa de texto
        for botao, (_, texto) in zip(botoes, SUGESTOES):
            botao.click(
                responder,
                inputs=[gr.State(texto), temperature, top_p, max_tokens, historico],
                outputs=saidas,
            )

    return demo, launch_kwargs


# --------------------------------------------------------------------------- modo demo
DEMO_SQL = {
    "Quantos clientes são do estado SP?":
        "SELECT COUNT(*) FROM clientes WHERE estado = 'SP'",
    "Quais os 5 produtos mais caros?":
        "SELECT nome, preco FROM produtos ORDER BY preco DESC LIMIT 5",
    "Qual o faturamento total dos pedidos entregues?":
        "SELECT SUM(valor_total) FROM pedidos WHERE status = 'entregue'",
    "Quais os 5 clientes que mais gastaram?":
        "SELECT c.nome, SUM(p.valor_total) AS total FROM clientes c "
        "JOIN pedidos p ON p.cliente_id = c.id GROUP BY c.id ORDER BY total DESC LIMIT 5",
    "Quantos pedidos existem por status?":
        "SELECT status, COUNT(*) FROM pedidos GROUP BY status",
}


def gerador_demo():
    """Gerador de mentira para testar a interface sem modelo e sem GPU."""

    def gerar(pergunta, temperature, top_p, max_tokens):
        time.sleep(0.5)
        if "erro" in pergunta.lower():  # para ver como a tela mostra um erro de SQL
            return "SELECT * FROM tabela_que_nao_existe"
        return DEMO_SQL.get(pergunta.strip(), "SELECT nome, cidade, estado FROM clientes LIMIT 5")

    return gerar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="pasta do adaptador LoRA (vazio = modelo base)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--share", action="store_true", help="gera link público (útil no Colab)")
    ap.add_argument("--demo", action="store_true", help="só a interface, sem carregar o modelo")
    args = ap.parse_args()

    estado = {}

    def gerar(pergunta, temperature, top_p, max_tokens):
        return estado["fn"](pergunta, temperature, top_p, max_tokens)

    # A interface é montada ANTES de carregar o modelo: se houver erro nela, aparece em
    # segundos, e não depois de minutos de carregamento.
    demo, launch_kwargs = construir_interface(gerar)

    if args.demo:
        estado["fn"] = gerador_demo()
    else:
        from evaluate_model import generate_sql, load_model  # importa torch só aqui

        tok, model = load_model(args.model, args.adapter, not args.no_4bit)

        def _gerar_real(pergunta, temperature, top_p, max_tokens):
            return generate_sql(
                tok, model, [pergunta],
                batch_size=1,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )[0]

        estado["fn"] = _gerar_real

    demo.launch(share=args.share, **launch_kwargs)


if __name__ == "__main__":
    main()