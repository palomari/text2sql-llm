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
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import pandas as pd

from common import MODEL_ID, SCHEMA, clean_sql, run_sql

TITULO = "Text-to-SQL em português"
ICONE_ENVIAR = "→"
ICONE_AGUARDE = "⏳"
LIMITE_LINHAS = 200  # mesmo limite usado em run_sql

SUGESTOES = [
    ("user", "Quantos clientes são do estado SP?"),
    ("mail", "Quais os 5 produtos mais caros?"),
    ("bar", "Qual o faturamento total dos pedidos entregues?"),
    ("users", "Quais os 5 clientes que mais gastaram?"),
    ("list", "Quantos pedidos existem por status?"),
]

# Ícones de traço (estilo Lucide), desenhados em 24x24.
ICONES = {
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "mail": '<rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>',
    "bar": '<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/>',
    "list": '<line x1="21" x2="3" y1="6" y2="6"/><line x1="15" x2="3" y1="12" y2="12"/><line x1="17" x2="3" y1="18" y2="18"/>',
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "table": '<path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
}


def svg(nome, tam=18, classe=""):
    return (
        f'<svg class="{classe}" xmlns="http://www.w3.org/2000/svg" width="{tam}" height="{tam}" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONES[nome]}</svg>'
    )


def uri_svg(nome, cor="#000000"):
    desenho = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
        f"stroke='{cor}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        f"{ICONES[nome].replace(chr(34), chr(39))}</svg>"
    )
    return "data:image/svg+xml," + quote(desenho, safe="")


TEMA_ONCLICK = (
    "event.stopPropagation();document.body.classList.toggle('dark');"
    "try{localStorage.setItem('tema',document.body.classList.contains('dark')?'escuro':'claro')}catch(e){}"
)
COPIAR_ONCLICK = (
    "event.stopPropagation();var b=this,t=b.closest('.code-box').querySelector('code').innerText;"
    "navigator.clipboard.writeText(t).then(function(){b.classList.add('ok');"
    "setTimeout(function(){b.classList.remove('ok')},1200)})"
)

JS_CARGA = """
() => {
  if (window.__t2sqlPronto) return;
  window.__t2sqlPronto = true;
  try {
    document.body.classList.toggle('dark', localStorage.getItem('tema') === 'escuro');
  } catch (e) {
    document.body.classList.remove('dark');
  }
  document.addEventListener('click', (ev) => {
    const alvo = ev.target.closest('[data-acao]');
    if (!alvo) return;
    if (alvo.dataset.acao === 'tema') {
      const escuro = document.body.classList.toggle('dark');
      try { localStorage.setItem('tema', escuro ? 'escuro' : 'claro'); } catch (e) {}
    } else if (alvo.dataset.acao === 'copiar') {
      const codigo = alvo.closest('.code-box').querySelector('code');
      navigator.clipboard.writeText(codigo.innerText).then(() => {
        alvo.classList.add('ok');
        setTimeout(() => alvo.classList.remove('ok'), 1200);
      }).catch(() => {});
    } else if (alvo.dataset.acao === 'topo') {
      window.scrollTo({top: 0, behavior: 'smooth'});
    }
  });
}
"""

JS_ROLAR = """
() => {
  const msgs = document.querySelectorAll('#chat-history .msg.user');
  if (msgs.length) msgs[msgs.length - 1].scrollIntoView({behavior: 'smooth', block: 'start'});
}
"""


SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "JOIN", "LEFT",
    "RIGHT", "INNER", "OUTER", "CROSS", "ON", "AS", "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE",
    "BETWEEN", "DISTINCT", "ASC", "DESC", "UNION", "ALL", "CASE", "WHEN", "THEN", "ELSE", "END",
    "WITH", "EXISTS", "CREATE", "TABLE", "PRIMARY", "KEY", "INTEGER", "TEXT", "REAL", "DATE",
}
SQL_FUNCS = {
    "COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "STRFTIME", "DATE", "LENGTH", "UPPER", "LOWER",
    "COALESCE", "IFNULL", "CAST", "ABS",
}
_TOKEN = re.compile(
    r"('(?:[^']|'')*')|(--[^\n]*)|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z_0-9]*)|(\s+)|(.)", re.S
)
_LITERAL = re.compile(r"('(?:[^']|'')*')")
_CLAUSULAS = re.compile(
    r"\s+(?=(?:FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|"
    r"(?:LEFT\s+|RIGHT\s+|INNER\s+|CROSS\s+)?JOIN)\b)",
    re.I,
)


def destacar_sql(texto):
    saida = []
    for m in _TOKEN.finditer(texto):
        texto_str, comentario, numero, palavra, espaco, outro = m.groups()
        if texto_str:
            saida.append(f'<span class="s-str">{html.escape(texto_str)}</span>')
        elif comentario:
            saida.append(f'<span class="s-com">{html.escape(comentario)}</span>')
        elif numero:
            saida.append(f'<span class="s-num">{numero}</span>')
        elif palavra:
            maiuscula = palavra.upper()
            proximo = texto[m.end():].lstrip()[:1]
            if maiuscula in SQL_FUNCS and proximo == "(":
                saida.append(f'<span class="s-fn">{html.escape(palavra)}</span>')
            elif maiuscula in SQL_KEYWORDS:
                saida.append(f'<span class="s-kw">{html.escape(palavra)}</span>')
            else:
                saida.append(html.escape(palavra))
        else:
            saida.append(html.escape(espaco or outro))
    return "".join(saida)


def formatar_sql(sql):
    partes = _LITERAL.split(sql.strip())
    for i in range(0, len(partes), 2):
        partes[i] = _CLAUSULAS.sub("\n", re.sub(r"\s+", " ", partes[i]))
    return "".join(partes).strip()


def formatar_esquema(esquema):
    saida = []
    for linha in esquema.splitlines():
        m = re.match(r"\s*CREATE TABLE (\w+) \((.*)\);\s*$", linha)
        if m:
            colunas = [c.strip() for c in m.group(2).split(",")]
            saida.append(f"CREATE TABLE {m.group(1)} (")
            saida.append(",\n".join(f"    {c}" for c in colunas))
            saida.append(");\n")
        else:
            saida.append(linha)
    return "\n".join(saida).strip()


def html_header():
    return (
        '<div class="header-inner">'
        '<div class="brand" data-acao="topo" title="Voltar ao início" style="cursor: pointer;">'
        f'<div class="brand-icon">{svg("database", 22)}</div>'
        f"<span>{TITULO}</span></div>"
        '<button type="button" class="theme-toggle" data-acao="tema" '
        'title="Alternar tema claro/escuro" aria-label="Alternar tema claro/escuro" '
        f'onclick="{TEMA_ONCLICK}">{svg("sun", 19, "i-sun")}{svg("moon", 19, "i-moon")}</button>'
        "</div>"
    )


INTRO_HTML = """
<div class="welcome">
    <div class="greeting">Bem-vindo(a)</div>
    <h1>O que você gostaria de saber?</h1>
    <p>Pergunte em linguagem natural sobre a loja fictícia;<br>
    o modelo (Qwen2.5 + LoRA) gera a consulta SQL e ela é executada no banco.</p>
</div>
"""


def html_usuario(pergunta):
    return (
        '<div class="msg user"><div class="bubble">' + html.escape(pergunta) + "</div>"
        f'<div class="av av-user">{svg("user", 19)}</div></div>'
    )


def html_bot(conteudo):
    return (
        f'<div class="msg bot"><div class="av av-bot">{svg("sparkles", 18)}</div>'
        f'<div class="answer">{conteudo}</div></div>'
    )


def html_pensando():
    return html_bot(
        '<div class="typing"><span></span><span></span><span></span>'
        "<em>Gerando a consulta SQL…</em></div>"
    )


def caixa_codigo(texto, variante, numerar=False):
    gutter = ""
    if numerar:
        numeros = "\n".join(str(i) for i in range(1, texto.count("\n") + 2))
        gutter = f'<div class="ln">{numeros}</div>'
    return (
        f'<div class="code-box cb-{variante}">'
        '<button type="button" class="copy-btn" data-acao="copiar" title="Copiar" '
        f'aria-label="Copiar" onclick="{COPIAR_ONCLICK}">{svg("copy", 15)}</button>'
        f'<div class="code-body">{gutter}<pre><code>{destacar_sql(texto)}</code></pre></div></div>'
    )


def titulo(icone, texto):
    return f'<div class="title">{svg(icone, 16)}<span>{texto}</span></div>'


def secao(icone, texto, corpo, aberta=None):
    if aberta is None:
        return f'<div class="sec">{titulo(icone, texto)}{corpo}</div>'
    atributo = " open" if aberta else ""
    return (
        f'<details class="sec"{atributo}><summary class="title">{svg(icone, 16)}'
        f"<span>{texto}</span></summary>{corpo}</details>"
    )


def status(texto, erro=False):
    icone = "alert" if erro else "check"
    classe = "status erro" if erro else "status"
    return f'<div class="{classe}">{svg(icone, 16)}<span>{html.escape(texto)}</span></div>'


def html_resposta(sql, resultado=None, erro=None, esquema_aberto=True):
    exibicao = (formatar_sql(sql) + ";") if sql else "(vazia)"
    if erro is not None:
        corpo = status(f"Erro ao executar a SQL: {erro}", erro=True)
    else:
        cols, rows = resultado
        n = len(rows)
        if n == 0:
            corpo = status("A consulta executou, mas não retornou linhas.")
        else:
            df = pd.DataFrame(rows, columns=cols)
            df.index = range(1, n + 1)
            tabela = df.to_html(
                classes="resultado",
                index=True,
                index_names=False,
                border=0,
                na_rep="NULL",
                float_format=lambda x: f"{x:.2f}",
            )
            aviso = f" (mostrando só as primeiras {LIMITE_LINHAS})" if n >= LIMITE_LINHAS else ""
            corpo = f'<div class="tabela">{tabela}</div>' + status(f"{n} linha(s) retornada(s){aviso}.")
    
    # ORDEM INVERTIDA: 1º Resultado da consulta, 2º SQL gerada, 3º Esquema do banco
    return html_bot(
        secao("table", "Resultado da consulta", corpo)
        + titulo("code", "SQL gerada")
        + caixa_codigo(exibicao, "night", numerar=True)
        + secao("database", "Esquema do banco", caixa_codigo(formatar_esquema(SCHEMA), "day"),
                aberta=esquema_aberto)
    )


def html_turno(pergunta, sql, resultado=None, erro=None, esquema_aberto=True):
    return html_usuario(pergunta) + html_resposta(sql, resultado, erro, esquema_aberto)


def html_falha(exc):
    return html_bot(status(f"Erro ao gerar a resposta: {exc}", erro=True))


CSS_BASE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --text: #1b1d4a;
    --text-soft: #4a4f7a;
    --muted: #8a8fb5;
    --border: #e7e5fb;
    --accent: #6a5af0;
    --accent-soft: #ece9ff;
    --card: linear-gradient(180deg, #f6f6fe 0%, #faf5fa 100%);
    --page: linear-gradient(135deg, #e6e9f9 0%, #ebeefb 55%, #efedfa 100%);
    --surface: #ffffff;
    --surface-2: #f8f7ff;
    --bubble: #dfdbfc;
    --bubble-text: #25235a;
    --code-night-bg: #1d2030;
    --code-night-text: #e6e8f5;
    --code-night-ln: #686d8f;
    --code-day-bg: #f7f6ff;
    --code-day-text: #2c2e5c;
    --ok-bg: #eefaf3;
    --ok-text: #23915f;
    --err-bg: #fff0f0;
    --err-text: #c0392b;
    --shadow: 0 10px 34px rgba(76, 66, 170, .08);
}

body.dark {
    --text: #ecebff;
    --text-soft: #c7c8ee;
    --muted: #9496c0;
    --border: #2f3052;
    --accent: #9a8cff;
    --accent-soft: #2b2952;
    --card: linear-gradient(180deg, #1b1c30 0%, #1e1b2f 100%);
    --page: linear-gradient(135deg, #0f101c 0%, #14152a 100%);
    --surface: #25263f;
    --surface-2: #202137;
    --bubble: #3b3573;
    --bubble-text: #f0eeff;
    --code-night-bg: #12131f;
    --code-night-text: #e6e8f5;
    --code-night-ln: #5d6285;
    --code-day-bg: #1f2038;
    --code-day-text: #dcdcff;
    --ok-bg: #16322a;
    --ok-text: #4cd29a;
    --err-bg: #3a1d22;
    --err-text: #ff8d8d;
    --shadow: 0 10px 34px rgba(0, 0, 0, .35);
}

body, gradio-app {
    background: var(--page) fixed !important;
    min-height: 100vh;
}

.gradio-container {
    --body-background-fill: transparent;
    --block-border-width: 0px;
    --block-shadow: none;
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    width: calc(100% - 48px) !important;
    max-width: 1180px !important;
    margin: 24px auto !important;
    padding: 0 !important;
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 24px !important;
    box-shadow: var(--shadow);
    color: var(--text);
}

.gradio-container button,
.gradio-container input,
.gradio-container textarea {
    font-family: inherit !important;
}

.gradio-container .contain,
.gradio-container .main {
    padding: 0 !important;
}

footer,
.built-with,
.show-api {
    display: none !important;
}

#app-header,
#chat-history,
#welcome-area .block,
#welcome-area .prose {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ---------- cabeçalho ---------- */
#app-header .header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 26px 44px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
    color: var(--text);
    font-size: 18px;
    font-weight: 600;
}

.brand-icon {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background: var(--accent-soft);
    color: var(--accent);
}

.brand-icon svg {
    display: block;
    margin: auto;
}

.theme-toggle {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--accent);
    box-shadow: 0 4px 14px rgba(76, 66, 170, .08);
    cursor: pointer;
    transition: transform .15s ease;
    padding: 0;
}

.theme-toggle svg {
    display: block;
    margin: auto;
}

.theme-toggle:hover {
    transform: scale(1.06);
}

.theme-toggle .i-moon {
    display: none;
}

body.dark .theme-toggle .i-sun {
    display: none;
}

body.dark .theme-toggle .i-moon {
    display: block;
    stroke: #ffffff !important;
}

/* ---------- boas-vindas ---------- */
.welcome {
    text-align: center;
    margin: 34px auto 34px;
    max-width: 700px;
    padding: 0 20px;
}

.welcome .greeting {
    color: var(--muted);
    font-size: 15px;
    margin-bottom: 12px;
}

.welcome h1 {
    color: var(--text) !important;
    font-size: clamp(24px, 3vw, 31px) !important;
    line-height: 1.25 !important;
    font-weight: 700 !important;
    margin: 0 0 12px !important;
    border: none !important;
}

.welcome p {
    color: var(--muted) !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    margin: 0 !important;
}

/* ---------- sugestões ---------- */
#suggestions {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 14px !important;
    max-width: 900px;
    width: calc(100% - 40px);
    margin: 0 auto 54px !important;
    border: none !important;
    background: transparent !important;
}

.sug {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: space-between !important;
    gap: 16px;
    min-height: 98px;
    min-width: 0 !important;
    padding: 16px !important;
    text-align: left !important;
    white-space: normal !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text-soft) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.45 !important;
    box-shadow: 0 6px 22px rgba(76, 66, 170, .06) !important;
    transition: transform .15s ease, border-color .15s ease !important;
}

.sug:hover {
    transform: translateY(-2px);
    border-color: var(--accent) !important;
}

.sug::before {
    content: "";
    flex: 0 0 auto;
    width: 20px;
    height: 20px;
    background-color: var(--text-soft);
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-position: center;
    mask-position: center;
    -webkit-mask-size: contain;
    mask-size: contain;
}

/* ---------- barra de pergunta ---------- */
#question-bar {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 12px !important;
    max-width: 720px;
    width: calc(100% - 40px);
    margin: 0 auto 18px !important;
    padding: 10px 12px 10px 16px !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    box-shadow: 0 10px 30px rgba(76, 66, 170, .09);
}

#question-bar::before {
    content: "";
    flex: 0 0 38px;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background-color: var(--accent-soft);
    background-repeat: no-repeat;
    background-position: center;
    background-size: 18px;
}

#question-bar > * {
    min-width: 0 !important;
}

#question-input {
    flex: 1 1 auto !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#question-input label,
#question-input .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#question-input textarea,
#question-input input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    resize: none !important;
    font-size: 15px !important;
    color: var(--text) !important;
}

#send-btn {
    flex: 0 0 auto !important;
    width: 64px !important;
    min-width: 64px !important;
    height: 40px !important;
    padding: 0 !important;
    border-radius: 20px !important;
    border: none !important;
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

#send-btn:hover {
    filter: brightness(.96);
}

#params {
    max-width: 720px;
    width: calc(100% - 40px);
    margin: 0 auto 14px !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}

/* ---------- conversa ---------- */
#chat-history {
    max-width: 800px;
    width: 100%;
    margin: 0 auto;
    padding: 0 28px !important;
}

.msg {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 22px 0;
}

.msg.user {
    justify-content: flex-end;
}

.bubble {
    max-width: 78%;
    padding: 13px 20px;
    border-radius: 18px;
    background: var(--bubble);
    color: var(--bubble-text);
    font-size: 15px;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.av {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
}

.av svg {
    display: block;
    margin: auto;
}

.av-user {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: var(--accent-soft);
    color: var(--accent);
}

.av-bot {
    width: 34px;
    height: 34px;
    border-radius: 11px;
    background: linear-gradient(135deg, #7664f4, #5a48e6);
    color: #ffffff;
    box-shadow: 0 6px 16px rgba(106, 90, 240, .3);
}

.answer {
    flex: 1 1 auto;
    min-width: 0;
    max-width: calc(100% - 50px);
    padding: 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow);
    color: var(--text);
}

.title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 10px;
    color: var(--text);
    font-size: 13.5px;
    font-weight: 600;
    list-style: none;
}

.title svg {
    display: inline-block;
    vertical-align: middle;
}

summary.title {
    cursor: pointer;
    margin: 0;
}

summary.title::-webkit-details-marker {
    display: none;
}

details[open] > summary.title {
    margin-bottom: 10px;
}

.sec {
    margin-top: 12px;
    padding: 13px 13px;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: var(--surface);
}

/* ---------- código ---------- */
.code-box {
    position: relative;
    border-radius: 11px;
    overflow: hidden;
}

.cb-night {
    background: var(--code-night-bg);
    color: var(--code-night-text);
}

.cb-day {
    background: var(--code-day-bg);
    color: var(--code-day-text);
    border: 1px solid var(--border);
}

.code-body {
    display: flex;
    gap: 16px;
    padding: 14px 48px 14px 16px;
    max-height: 300px;
    overflow: auto;
}

.ln,
.code-box pre,
.code-box code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace !important;
    font-size: 12.5px !important;
    line-height: 1.7 !important;
}

.ln {
    color: var(--code-night-ln);
    text-align: right;
    white-space: pre;
    user-select: none;
}

.code-box pre {
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    color: inherit !important;
    white-space: pre !important;
    overflow: visible !important;
}

.code-box code {
    background: transparent !important;
    color: inherit !important;
    padding: 0 !important;
}

.copy-btn {
    position: absolute;
    top: 10px;
    right: 10px;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    border-radius: 8px;
    background: transparent;
    color: #9aa0c4;
    cursor: pointer;
    padding: 0;
}

.copy-btn svg {
    display: block;
    margin: auto;
}

.copy-btn:hover {
    background: rgba(128, 128, 170, .18);
}

.copy-btn.ok {
    color: #3fc58f;
}

.cb-night .s-kw { color: #ffb06b; }
.cb-night .s-fn { color: #c3a6ff; }
.cb-night .s-str { color: #ffcf9a; }
.cb-night .s-num { color: #7fd6ff; }
.cb-night .s-com { color: #7c819f; }

/* CORES CORRIGIDAS PARA O TEMA CLARO */
.cb-day .s-kw { color: #3b28cc !important; font-weight: 600; }
.cb-day .s-fn { color: #6230b8 !important; }
.cb-day .s-str { color: #9c3c0a !important; }
.cb-day .s-num { color: #006699 !important; }
.cb-day .s-com { color: #5f638f !important; }

body.dark .cb-day .s-kw { color: #a99dff; }
body.dark .cb-day .s-fn { color: #c9a8ff; }
body.dark .cb-day .s-str { color: #ffbf8a; }
body.dark .cb-day .s-num { color: #7fd6ff; }

/* ---------- tabela de resultado ---------- */
.tabela {
    max-height: 340px;
    overflow: auto;
    border: 1px solid var(--border);
    border-radius: 10px;
}

table.resultado {
    width: 100% !important;
    margin: 0 !important;
    border-collapse: collapse !important;
    font-size: 13px !important;
    color: var(--text) !important;
}

table.resultado th,
table.resultado td {
    padding: 10px 14px !important;
    text-align: left !important;
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
    color: var(--text) !important;
}

table.resultado thead th {
    position: sticky;
    top: 0;
    background: var(--surface-2) !important;
    color: var(--text-soft) !important;
    font-weight: 500 !important;
}

table.resultado tbody th {
    width: 44px;
    color: var(--muted) !important;
    font-weight: 400 !important;
    border-right: 1px solid var(--border) !important;
}

table.resultado tbody tr:last-child th,
table.resultado tbody tr:last-child td {
    border-bottom: none !important;
}

/* ---------- status ---------- */
.status {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 12px;
    padding: 10px 14px;
    border-radius: 10px;
    background: var(--ok-bg);
    color: var(--ok-text);
    font-size: 12.5px;
}

.status svg {
    display: inline-block;
    vertical-align: middle;
}

.status.erro {
    background: var(--err-bg);
    color: var(--err-text);
}

.typing {
    display: flex;
    align-items: center;
    gap: 5px;
    color: var(--muted);
    font-size: 13px;
}

.typing em {
    margin-left: 6px;
    font-style: normal;
}

.typing span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulsar 1.2s infinite ease-in-out;
}

.typing span:nth-child(2) { animation-delay: .15s; }
.typing span:nth-child(3) { animation-delay: .3s; }

@keyframes pulsar {
    0%, 80%, 100% { opacity: .25; transform: scale(.8); }
    40% { opacity: 1; transform: scale(1); }
}

@media (max-width: 900px) {
    #suggestions { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    #app-header .header-inner { padding: 20px 22px; }
}
"""


def css_icones():
    regras = []
    for i, (icone, _) in enumerate(SUGESTOES, 1):
        uri = uri_svg(icone)
        regras.append(
            f'.sug-{i}::before {{ -webkit-mask-image: url("{uri}"); mask-image: url("{uri}"); }}'
        )
    regras.append(f'#question-bar::before {{ background-image: url("{uri_svg("sparkles", "#7a6bef")}"); }}')
    return "\n".join(regras)


CSS = CSS_BASE + css_icones()


def _aceita(func, nome):
    try:
        return nome in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _ligar(gatilho, fn, entradas, saidas):
    try:
        evento = gatilho(fn, inputs=entradas, outputs=saidas, show_progress="hidden")
    except TypeError:
        evento = gatilho(fn, inputs=entradas, outputs=saidas)
    try:
        evento.then(None, None, None, js=JS_ROLAR)
    except Exception:
        pass


def construir_interface(gerar):
    def responder(pergunta, temperature, top_p, max_tokens, historico_atual):
        historico_atual = historico_atual or ""
        pergunta = (pergunta or "").strip()
        if not pergunta:
            yield historico_atual, "", gr.update(value=ICONE_ENVIAR, interactive=True), gr.update()
            return

        esconder = gr.update(visible=False)
        usuario = html_usuario(pergunta)
        primeira = not historico_atual

        yield (historico_atual + usuario + html_pensando(), "",
               gr.update(value=ICONE_AGUARDE, interactive=False), esconder)

        try:
            sql = clean_sql(gerar(pergunta, temperature, top_p, max_tokens))
            ok, res = run_sql(sql)
            if ok:
                resposta = html_resposta(sql, resultado=res, esquema_aberto=primeira)
            else:
                resposta = html_resposta(sql, erro=res, esquema_aberto=primeira)
        except Exception as exc:
            resposta = html_falha(exc)

        yield (historico_atual + usuario + resposta, "",
               gr.update(value=ICONE_ENVIAR, interactive=True), esconder)

    tema = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        radius_size="lg",
    )
    blocks_kwargs = {"title": TITULO}
    launch_kwargs = {}
    for nome, valor in (("theme", tema), ("css", CSS), ("js", JS_CARGA)):
        if _aceita(gr.Blocks.__init__, nome):
            blocks_kwargs[nome] = valor
        elif _aceita(gr.Blocks.launch, nome):
            launch_kwargs[nome] = valor

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(html_header(), elem_id="app-header")

        with gr.Column(elem_id="welcome-area") as boas_vindas:
            gr.HTML(INTRO_HTML)
            with gr.Row(elem_id="suggestions"):
                botoes = [
                    gr.Button(texto, elem_classes=["sug", f"sug-{i}"])
                    for i, (_, texto) in enumerate(SUGESTOES, 1)
                ]

        historico = gr.HTML(value="", elem_id="chat-history")

        # PARÂMETROS ACIMA DA BARRA DE PESQUISA (ABRINDO PARA CIMA)
        with gr.Accordion("Parâmetros de geração", open=False, elem_id="params"):
            temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.1,
                                    label="Temperature (0 = determinístico)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            max_tokens = gr.Slider(32, 400, value=200, step=8, label="Máximo de tokens")

        with gr.Row(elem_id="question-bar"):
            pergunta = gr.Textbox(
                label="Pergunta",
                placeholder="Faça sua pergunta aqui...",
                lines=1,
                max_lines=4,
                show_label=False,
                container=False,
                elem_id="question-input",
            )
            btn = gr.Button(ICONE_ENVIAR, variant="primary", elem_id="send-btn")

        entradas = [pergunta, temperature, top_p, max_tokens, historico]
        saidas = [historico, pergunta, btn, boas_vindas]

        _ligar(btn.click, responder, entradas, saidas)
        _ligar(pergunta.submit, responder, entradas, saidas)
        for botao, (_, texto) in zip(botoes, SUGESTOES):
            _ligar(
                botao.click,
                responder,
                [gr.State(texto), temperature, top_p, max_tokens, historico],
                saidas,
            )

    return demo, launch_kwargs


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
    def gerar(pergunta, temperature, top_p, max_tokens):
        time.sleep(0.5)
        if "erro" in pergunta.lower():
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

    if not args.demo and args.adapter and not (Path(args.adapter) / "adapter_config.json").exists():
        raise SystemExit(
            f"\nAdaptador não encontrado em '{args.adapter}' (falta o arquivo adapter_config.json).\n"
            f"Pasta atual: {os.getcwd()}\n"
            "O adaptador é criado pelo treino (src/train.py --name run1 ...) e NÃO vai para o GitHub.\n"
            "Rode o treino nesta sessão, ou copie a pasta do adaptador do Drive para este caminho.\n"
            "Se estiver na pasta errada, rode antes:  %cd /content/text2sql-llm\n"
        )

    estado = {}

    def gerar(pergunta, temperature, top_p, max_tokens):
        return estado["fn"](pergunta, temperature, top_p, max_tokens)

    demo, launch_kwargs = construir_interface(gerar)

    if args.demo:
        estado["fn"] = gerador_demo()
    else:
        from evaluate_model import generate_sql, load_model

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