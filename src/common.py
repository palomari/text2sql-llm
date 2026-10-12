"""Funções compartilhadas: schema do banco, prompt e execução de SQL."""
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "loja.db"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SCHEMA = """CREATE TABLE clientes (id INTEGER PRIMARY KEY, nome TEXT, cidade TEXT, estado TEXT, data_cadastro DATE);
CREATE TABLE produtos (id INTEGER PRIMARY KEY, nome TEXT, categoria TEXT, preco REAL, estoque INTEGER);
CREATE TABLE pedidos (id INTEGER PRIMARY KEY, cliente_id INTEGER, data_pedido DATE, status TEXT, valor_total REAL);
CREATE TABLE itens_pedido (id INTEGER PRIMARY KEY, pedido_id INTEGER, produto_id INTEGER, quantidade INTEGER, preco_unitario REAL);
-- status: 'entregue', 'enviado', 'pendente', 'cancelado'
-- datas no formato 'AAAA-MM-DD'"""

SYSTEM_PROMPT = (
    "Você converte perguntas em português para consultas SQL (SQLite). "
    "Responda apenas com a consulta SQL.\n\nEsquema do banco:\n" + SCHEMA
)


def build_messages(question: str, answer: str | None = None):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def supports_bf16() -> bool:
    """bf16 nativo só existe em GPUs com compute capability >= 8.0 (A100, RTX 30xx+).

    Não usar torch.cuda.is_bf16_supported(): em versões novas do PyTorch ele devolve True
    na T4 (capability 7.5) por causa de emulação, e o treino quebra.
    """
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def load_causal_lm(model_id, quant, dtype):
    """Carrega o modelo na precisão pedida, em qualquer versão do transformers.

    Versões novas ignoram `torch_dtype` (o nome agora é `dtype`) e carregam no formato
    original (bf16), o que quebra o treino em fp16 na T4.
    """
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, dtype=dtype, device_map={"": 0})
    except TypeError:  # versões antigas
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, torch_dtype=dtype, device_map={"": 0})
    return model


def clean_sql(text: str) -> str:
    """Remove cercas de markdown e texto extra, mantendo só a primeira consulta."""
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1).strip()
    text = text.split(";")[0].strip()
    return text


def run_sql(sql: str, db_path: Path = DB_PATH):
    """Executa uma consulta SOMENTE LEITURA. Retorna (ok, linhas_ou_erro)."""
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        return False, "Apenas consultas SELECT são permitidas."
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(200)
            return True, (cols, rows)
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def normalize_rows(rows):
    """Compara resultados ignorando a ordem das linhas e arredondando floats."""
    def norm(v):
        return round(v, 2) if isinstance(v, float) else v
    return sorted(tuple(norm(v) for v in r) for r in rows)