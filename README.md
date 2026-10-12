# Text-to-SQL em português com LLM local (QLoRA + GPU)

Fine-tuning de uma LLM pequena para converter perguntas em português em consultas SQL (SQLite) sobre o banco de uma loja fictícia. O projeto inclui dataset, treino com experimentos, avaliação por execução da SQL e um frontend em Gradio.

## 1. Objetivo e problema

Dado o esquema de um banco (clientes, produtos, pedidos, itens de pedido) e uma pergunta em português, o modelo deve devolver **uma consulta SQL** que, ao ser executada, retorna o resultado correto.

Exemplo:

| Pergunta | SQL gerada |
|---|---|
| Quais os 5 produtos mais caros? | `SELECT nome, preco FROM produtos ORDER BY preco DESC LIMIT 5` |

## 2. LLM escolhida

**Qwen2.5-1.5B-Instruct** (`Qwen/Qwen2.5-1.5B-Instruct`).

- Cabe com folga em 8 GB de VRAM usando quantização de 4 bits (QLoRA) e roda na T4 (16 GB) do Colab.
- Entende português razoavelmente bem e já conhece SQL.
- Licença permissiva (Apache 2.0).

> Preencha aqui qualquer troca de modelo que fizer.

## 3. Hardware e versões

> Preencha com os dados reais da execução (saída do `nvidia-smi`):

| Item | Valor |
|---|---|
| GPU | _ex.: NVIDIA Tesla T4, 16 GB_ |
| Driver / CUDA | _ex.: 550.x / 12.4_ |
| Python | _ex.: 3.10_ |
| PyTorch | _ex.: 2.x + cu12x_ |

Print da GPU funcionando no VSCode: `docs/imagens/gpu_vscode.png`

![GPU no VSCode](docs/imagens/gpu_vscode.png)

## 4. Estrutura do repositório

```
├── data/              # loja.db e train/val/test.jsonl (gerados por make_dataset.py)
├── src/
│   ├── common.py          # esquema, prompt e execução segura de SQL
│   ├── make_dataset.py    # cria o banco e o dataset
│   ├── train.py           # fine-tuning QLoRA
│   ├── evaluate_model.py  # avaliação (accuracy por execução)
│   └── app.py             # frontend Gradio
├── notebooks/text2sql_colab.ipynb   # fluxo completo no Colab
├── results/           # experimentos.csv, avaliacao.csv, previsões e gráficos
├── docs/imagens/      # prints
└── requirements.txt
```

## 5. Como executar

### Opção A: Google Colab (GPU T4)

Abra `notebooks/text2sql_colab.ipynb`, ative a GPU e execute as células em ordem. Para ver o notebook dentro do VSCode, use a extensão do Google Colab para VSCode.

### Opção B: máquina local com GPU NVIDIA

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)

# PyTorch com CUDA: use o comando de https://pytorch.org (exemplo para CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # deve imprimir True

python src/make_dataset.py
python src/evaluate_model.py --name baseline
python src/train.py --name run1 --r 8 --lr 2e-4 --epochs 3
python src/evaluate_model.py --name run1 --adapter outputs/run1/final
python src/app.py --adapter outputs/run1/final
```

No Windows, o `bitsandbytes` pode falhar; nesse caso use o WSL2 ou rode com `--no_4bit` (exige mais VRAM).

## 6. Dataset

- **Banco fictício** (`data/loja.db`, SQLite): 200 clientes, 27 produtos em 5 categorias, 1.500 pedidos e seus itens, gerados com semente fixa.
- **Pares pergunta → SQL:** 1.500 exemplos gerados a partir de 31 modelos de consulta (contagens, filtros, agregações, `GROUP BY`, `JOIN`, subconsultas, datas), com parâmetros e paráfrases variados (prefixos, minúsculas, sem pontuação).
- Toda SQL do dataset é executada na geração, então todas são válidas.
- **Divisão (1.142 treino / 125 validação / 233 teste):** nos modelos de consulta com parâmetros (estado, categoria, número N, preço etc.), a divisão é **por SQL**: a mesma SQL nunca aparece em dois conjuntos. Assim, o teste traz combinações de valores que o modelo não viu no treino (**193 das 233** SQL de teste são inéditas). Nos modelos sem parâmetros (SQL fixa, como "quantos clientes existem?") só as paráfrases são divididas, porque a SQL precisa aparecer no treino.
- Perguntas duplicadas foram removidas.

**Limitação:** as 40 SQL de teste que não são inéditas vêm dos modelos de SQL fixa. A avaliação reporta a accuracy geral (`exec_accuracy`) e a das SQL inéditas (`exec_acc_sql_inedita`), que é a que mede melhor a generalização.

## 7. Método

- **QLoRA:** modelo base em 4 bits (NF4) com adaptadores LoRA em todas as camadas lineares de atenção e MLP.
- **Prompt:** mensagem de sistema com o esquema do banco, pergunta do usuário e SQL como resposta (chat template do Qwen).
- **Métrica principal: accuracy por execução.** A SQL prevista e a SQL esperada são executadas, e conta como acerto se retornam o mesmo conjunto de linhas (ordem e arredondamento de 2 casas ignorados). Métricas auxiliares: SQL válida e exact match.
- **Segurança:** o frontend só executa `SELECT`/`WITH`, em conexão somente leitura.

## 8. Experimentos

> Preencha com os números reais gerados em `results/experimentos.csv` e `results/avaliacao.csv`.

| Execução | LoRA r | Learning rate | Épocas | Loss val | Exec. accuracy | SQL válida | Tempo (min) | VRAM pico (GB) |
|---|---|---|---|---|---|---|---|---|
| baseline (sem treino) | - | - | - | - | | | - | - |
| run1 | 8 | 2e-4 | 3 | | | | | |
| run2 | 16 | 2e-4 | 3 | | | | | |
| run3 | 8 | 1e-4 | 3 | | | | | |
| run4 | 8 | 2e-4 | 5 | | | | | |

Parâmetros de inferência testados na melhor execução (`temperature` 0.0, 0.3, 0.7 e 1.0): _preencha_.

Gráfico: `results/comparacao.png`

## 9. Avaliação qualitativa

> Cole aqui de 5 a 10 exemplos de `results/pred_*.json` (pergunta, SQL esperada, baseline, fine-tuned) e comente acertos e erros típicos.

## 10. Frontend

`python src/app.py --adapter outputs/run1/final` abre uma página com campo de pergunta, controles de `temperature`, `top-p` e máximo de tokens, a SQL gerada e a tabela com o resultado.

![Frontend](docs/imagens/frontend.png)

## 11. Conclusões e limitações

> Escreva suas conclusões: quanto o fine-tuning melhorou sobre o baseline, qual parâmetro mais influenciou, onde o modelo erra e o que melhorar (mais dados, perguntas escritas por pessoas, esquemas diferentes, modelo maior).

- O banco e as perguntas são sintéticos e limitados a um único esquema.
- O modelo não generaliza para bancos com outro esquema sem novo treino.
