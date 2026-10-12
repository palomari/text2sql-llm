"""Cria o banco SQLite fictício (loja.db) e o dataset pergunta -> SQL em português.

Uso:  python src/make_dataset.py
Saída: data/loja.db, data/train.jsonl, data/val.jsonl, data/test.jsonl
"""
import json
import random
import sqlite3
from datetime import date, timedelta

from common import DB_PATH, ROOT, SCHEMA, run_sql

random.seed(42)

ESTADOS = {
    "SP": ["São Paulo", "Campinas", "Santos", "Ribeirão Preto"],
    "RJ": ["Rio de Janeiro", "Niterói", "Petrópolis"],
    "MG": ["Belo Horizonte", "Uberlândia", "Juiz de Fora"],
    "RS": ["Porto Alegre", "Caxias do Sul", "Pelotas"],
    "PR": ["Curitiba", "Londrina", "Maringá"],
    "BA": ["Salvador", "Feira de Santana"],
    "PE": ["Recife", "Olinda"],
}
CATEGORIAS = {
    "Eletrônicos": ["Notebook", "Smartphone", "Fone de ouvido", "Monitor", "Teclado", "Mouse", "Tablet"],
    "Livros": ["Romance", "Ficção científica", "Biografia", "Livro de culinária", "Livro de programação"],
    "Casa": ["Cafeteira", "Liquidificador", "Ventilador", "Aspirador", "Panela elétrica"],
    "Esportes": ["Bola de futebol", "Tênis de corrida", "Bicicleta", "Halteres", "Tapete de yoga"],
    "Moda": ["Camiseta", "Calça jeans", "Jaqueta", "Vestido", "Boné"],
}
STATUS = ["entregue", "enviado", "pendente", "cancelado"]
NOMES = ["Ana", "Bruno", "Carla", "Diego", "Eduarda", "Felipe", "Gabriela", "Henrique", "Isabela", "João",
         "Karen", "Lucas", "Mariana", "Nicolas", "Olívia", "Pedro", "Rafaela", "Samuel", "Tatiana", "Vitor"]
SOBRENOMES = ["Silva", "Souza", "Oliveira", "Santos", "Lima", "Pereira", "Costa", "Rodrigues", "Almeida", "Nunes"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
         "outubro", "novembro", "dezembro"]


def build_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA.replace("-- status", "-- status"))  # comentários são ignorados pelo SQLite

    clientes = []
    for i in range(1, 201):
        uf = random.choice(list(ESTADOS))
        cidade = random.choice(ESTADOS[uf])
        nome = f"{random.choice(NOMES)} {random.choice(SOBRENOMES)}"
        d = date(2022, 1, 1) + timedelta(days=random.randint(0, 1000))
        clientes.append((i, nome, cidade, uf, d.isoformat()))
    con.executemany("INSERT INTO clientes VALUES (?,?,?,?,?)", clientes)

    produtos, pid = [], 1
    for cat, nomes in CATEGORIAS.items():
        for n in nomes:
            preco = round(random.uniform(20, 4000) if cat == "Eletrônicos" else random.uniform(15, 600), 2)
            produtos.append((pid, n, cat, preco, random.randint(0, 300)))
            pid += 1
    con.executemany("INSERT INTO produtos VALUES (?,?,?,?,?)", produtos)

    itens, pedidos, iid = [], [], 1
    for oid in range(1, 1501):
        cli = random.randint(1, 200)
        d = date(2023, 1, 1) + timedelta(days=random.randint(0, 900))
        st = random.choices(STATUS, weights=[60, 15, 15, 10])[0]
        total = 0.0
        for _ in range(random.randint(1, 4)):
            p = random.choice(produtos)
            q = random.randint(1, 5)
            itens.append((iid, oid, p[0], q, p[3]))
            total += q * p[3]
            iid += 1
        pedidos.append((oid, cli, d.isoformat(), st, round(total, 2)))
    con.executemany("INSERT INTO pedidos VALUES (?,?,?,?,?)", pedidos)
    con.executemany("INSERT INTO itens_pedido VALUES (?,?,?,?,?)", itens)
    con.commit()
    con.close()


# ---------------------------------------------------------------- templates
def templates():
    """Gera (pergunta, sql) amostrando parâmetros e paráfrases."""
    uf = random.choice(list(ESTADOS))
    cidade = random.choice([c for v in ESTADOS.values() for c in v])
    cat = random.choice(list(CATEGORIAS))
    st = random.choice(STATUS)
    ano = random.choice([2023, 2024, 2025])
    mes_i = random.randint(1, 12)
    mes = MESES[mes_i - 1]
    mm = f"{mes_i:02d}"
    n = random.choice([2, 3, 4, 5, 6, 7, 8, 10, 15])
    preco = random.choice([30, 50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000])
    qtd = random.choice([5, 10, 15, 20, 30, 50, 80, 100, 150])
    prod = random.choice([p for v in CATEGORIAS.values() for p in v])
    plural_st = {"entregue": "entregues", "enviado": "enviados", "pendente": "pendentes", "cancelado": "cancelados"}[st]

    T = [
        ([f"Quantos clientes existem?", "Qual o total de clientes cadastrados?", "Me diga o número de clientes."],
         "SELECT COUNT(*) FROM clientes"),
        ([f"Quantos clientes são do estado {uf}?", f"Quantos clientes moram em {uf}?", f"Qual o número de clientes de {uf}?"],
         f"SELECT COUNT(*) FROM clientes WHERE estado = '{uf}'"),
        ([f"Liste os nomes dos clientes de {cidade}.", f"Quais clientes moram em {cidade}?", f"Mostre o nome de todos os clientes da cidade de {cidade}."],
         f"SELECT nome FROM clientes WHERE cidade = '{cidade}'"),
        ([f"Quantos clientes se cadastraram em {ano}?", f"Qual o número de clientes cadastrados no ano de {ano}?"],
         f"SELECT COUNT(*) FROM clientes WHERE strftime('%Y', data_cadastro) = '{ano}'"),
        ([f"Quantos pedidos foram feitos?", "Qual o total de pedidos?", "Quantos pedidos existem no sistema?"],
         "SELECT COUNT(*) FROM pedidos"),
        ([f"Quantos pedidos estão {plural_st}?", f"Qual o número de pedidos com status {st}?", f"Conte os pedidos {plural_st}."],
         f"SELECT COUNT(*) FROM pedidos WHERE status = '{st}'"),
        ([f"Qual o faturamento total dos pedidos entregues?", "Some o valor de todos os pedidos entregues.", "Quanto foi vendido em pedidos entregues?"],
         "SELECT SUM(valor_total) FROM pedidos WHERE status = 'entregue'"),
        ([f"Qual o valor total vendido em {ano}?", f"Quanto foi faturado em {ano} (pedidos entregues)?"],
         f"SELECT SUM(valor_total) FROM pedidos WHERE status = 'entregue' AND strftime('%Y', data_pedido) = '{ano}'"),
        ([f"Quantos pedidos foram feitos em {mes} de {ano}?", f"Qual o número de pedidos de {mes}/{ano}?"],
         f"SELECT COUNT(*) FROM pedidos WHERE strftime('%Y-%m', data_pedido) = '{ano}-{mm}'"),
        ([f"Qual o ticket médio dos pedidos entregues?", "Qual o valor médio dos pedidos entregues?"],
         "SELECT AVG(valor_total) FROM pedidos WHERE status = 'entregue'"),
        ([f"Quais os {n} produtos mais caros?", f"Liste os {n} produtos com maior preço.", f"Mostre os {n} produtos de maior valor."],
         f"SELECT nome, preco FROM produtos ORDER BY preco DESC LIMIT {n}"),
        ([f"Quais os {n} produtos mais baratos?", f"Liste os {n} produtos com menor preço."],
         f"SELECT nome, preco FROM produtos ORDER BY preco ASC LIMIT {n}"),
        ([f"Quais produtos da categoria {cat} existem?", f"Liste os produtos de {cat}.", f"Mostre todos os produtos da categoria {cat}."],
         f"SELECT nome FROM produtos WHERE categoria = '{cat}'"),
        ([f"Quantos produtos há na categoria {cat}?", f"Qual o número de produtos de {cat}?"],
         f"SELECT COUNT(*) FROM produtos WHERE categoria = '{cat}'"),
        ([f"Quais produtos custam mais de {preco} reais?", f"Liste os produtos com preço acima de {preco}.", f"Mostre os produtos que custam mais que R$ {preco}."],
         f"SELECT nome, preco FROM produtos WHERE preco > {preco}"),
        ([f"Quais produtos têm menos de {qtd} unidades em estoque?", f"Liste os produtos com estoque abaixo de {qtd}."],
         f"SELECT nome, estoque FROM produtos WHERE estoque < {qtd}"),
        ([f"Qual o preço médio dos produtos da categoria {cat}?", f"Qual a média de preço de {cat}?"],
         f"SELECT AVG(preco) FROM produtos WHERE categoria = '{cat}'"),
        (["Qual o preço médio por categoria?", "Mostre a média de preço de cada categoria de produto."],
         "SELECT categoria, AVG(preco) FROM produtos GROUP BY categoria"),
        (["Quantos produtos existem por categoria?", "Conte os produtos agrupados por categoria."],
         "SELECT categoria, COUNT(*) FROM produtos GROUP BY categoria"),
        (["Quantos clientes existem por estado?", "Mostre o número de clientes em cada estado."],
         "SELECT estado, COUNT(*) FROM clientes GROUP BY estado"),
        (["Quantos pedidos existem por status?", "Conte os pedidos agrupados por status."],
         "SELECT status, COUNT(*) FROM pedidos GROUP BY status"),
        ([f"Quais os {n} clientes que mais gastaram?", f"Liste os {n} clientes com maior valor total em pedidos."],
         f"SELECT c.nome, SUM(p.valor_total) AS total FROM clientes c JOIN pedidos p ON p.cliente_id = c.id GROUP BY c.id ORDER BY total DESC LIMIT {n}"),
        ([f"Quais os {n} produtos mais vendidos em quantidade?", f"Liste os {n} produtos que mais saíram."],
         f"SELECT pr.nome, SUM(i.quantidade) AS total FROM itens_pedido i JOIN produtos pr ON pr.id = i.produto_id GROUP BY pr.id ORDER BY total DESC LIMIT {n}"),
        ([f"Qual o faturamento por categoria de produto?", "Some o valor vendido em cada categoria."],
         "SELECT pr.categoria, SUM(i.quantidade * i.preco_unitario) FROM itens_pedido i JOIN produtos pr ON pr.id = i.produto_id GROUP BY pr.categoria"),
        ([f"Quantos pedidos cada estado fez?", "Mostre o número de pedidos por estado do cliente."],
         "SELECT c.estado, COUNT(*) FROM pedidos p JOIN clientes c ON c.id = p.cliente_id GROUP BY c.estado"),
        ([f"Quantos pedidos foram feitos por clientes de {uf}?", f"Qual o total de pedidos de clientes do estado {uf}?"],
         f"SELECT COUNT(*) FROM pedidos p JOIN clientes c ON c.id = p.cliente_id WHERE c.estado = '{uf}'"),
        ([f"Quais clientes nunca fizeram pedidos?", "Liste os clientes sem nenhum pedido."],
         "SELECT nome FROM clientes WHERE id NOT IN (SELECT cliente_id FROM pedidos)"),
        ([f"Quantas unidades de {prod} foram vendidas?", f"Qual a quantidade total vendida de {prod}?"],
         f"SELECT SUM(i.quantidade) FROM itens_pedido i JOIN produtos pr ON pr.id = i.produto_id WHERE pr.nome = '{prod}'"),
        ([f"Qual o maior pedido já feito?", "Qual o valor do pedido mais caro?"],
         "SELECT MAX(valor_total) FROM pedidos"),
        ([f"Qual o faturamento mensal em {ano}?", f"Mostre o total vendido por mês em {ano}."],
         f"SELECT strftime('%m', data_pedido) AS mes, SUM(valor_total) FROM pedidos WHERE strftime('%Y', data_pedido) = '{ano}' GROUP BY mes ORDER BY mes"),
        ([f"Quais clientes de {uf} fizeram pedidos cancelados?", f"Liste os clientes de {uf} com algum pedido cancelado."],
         f"SELECT DISTINCT c.nome FROM clientes c JOIN pedidos p ON p.cliente_id = c.id WHERE c.estado = '{uf}' AND p.status = 'cancelado'"),
    ]
    idx = random.randrange(len(T))
    q, sql = T[idx]
    return stylize(random.choice(q)), sql, idx


PREFIXOS = ["", "", "", "Por favor, ", "Gostaria de saber: ", "Preciso saber: ", "Me ajude: ", "Consulta: ", "Pode me dizer? "]


def stylize(q: str) -> str:
    """Varia o estilo da pergunta (prefixos, minúsculas, sem pontuação) para ter mais diversidade."""
    q = random.choice(PREFIXOS) + q
    r = random.random()
    if r < 0.2:
        q = q.lower()
    elif r < 0.3:
        q = q.rstrip(".?")
    return q


def split_rows(rows):
    """Divide em treino/val/teste (80/10/10).

    - Modelos COM parâmetros: a divisão é por SQL (grupo). Uma mesma SQL nunca aparece em
      dois conjuntos, então o teste tem combinações de valores que o modelo não viu.
    - Modelos SEM parâmetros (SQL fixa): só as paráfrases são divididas, pois a SQL
      precisa aparecer no treino para o modelo poder aprendê-la.
    """
    by_tpl = {}
    for r in rows:
        by_tpl.setdefault(r["tpl"], []).append(r)
    splits = {"train": [], "val": [], "test": []}

    def assign(items):
        random.shuffle(items)
        n = len(items)
        a, b = int(n * 0.8), int(n * 0.9)
        splits["train"] += items[:a]
        splits["val"] += items[a:b]
        splits["test"] += items[b:]

    for items in by_tpl.values():
        groups = {}
        for r in items:
            groups.setdefault(r["sql"], []).append(r)
        if len(groups) == 1:  # SQL fixa
            assign(items)
        else:
            keys = list(groups)
            random.shuffle(keys)
            n = len(keys)
            a, b = max(1, int(n * 0.8)), max(2, int(n * 0.9))
            for k in keys[:a]:
                splits["train"] += groups[k]
            for k in keys[a:b]:
                splits["val"] += groups[k]
            for k in keys[b:]:
                splits["test"] += groups[k]
    for v in splits.values():
        random.shuffle(v)
    return splits


def main():
    build_db()
    seen, rows = set(), []
    attempts = 0
    while len(rows) < 1500 and attempts < 50000:
        attempts += 1
        q, sql, idx = templates()
        if q in seen:
            continue
        ok, res = run_sql(sql)
        if not ok:
            raise SystemExit(f"SQL inválida no gerador: {sql}\n{res}")
        seen.add(q)
        rows.append({"question": q, "sql": sql, "tpl": idx})
    splits = split_rows(rows)
    for name, data in splits.items():
        with open(ROOT / "data" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps({"question": r["question"], "sql": r["sql"]}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(data)} exemplos")
    tr = {r["sql"] for r in splits["train"]}
    inedita = sum(r["sql"] not in tr for r in splits["test"])
    print(f"SQL do teste inéditas no treino: {inedita}/{len(splits['test'])}")


if __name__ == "__main__":
    main()
