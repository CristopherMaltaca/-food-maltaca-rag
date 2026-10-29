# 🍔 Food Maltaca — Assistente de Cardápio com RAG

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-FF6B35)
![Gemini](https://img.shields.io/badge/Google_Gemini-Interactions_API-4285F4?logo=google&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Um assistente de chat que responde dúvidas sobre o cardápio de um restaurante fictício (a hamburgueria **Food Maltaca**) usando **RAG**: ele busca os trechos relevantes do cardápio antes de responder e só usa o que está lá.

O ponto central do projeto é o **guardrail anti-alucinação**: se a informação não estiver no cardápio, o assistente recusa em vez de inventar preço, ingrediente ou prazo de entrega.

---

## ✨ Funcionalidades

- **Respostas fundamentadas** — toda resposta nasce dos trechos recuperados do cardápio, com o contexto injetado no prompt.
- **Recusa explícita** — perguntas fora do cardápio recebem *"Não encontrei essa informação no cardápio do Food Maltaca."*
- **Fontes visíveis** — um expander mostra quais trechos foram consultados e a distância de similaridade de cada um.
- **Memória curta** — as últimas 6 mensagens vão como histórico, então dá para perguntar "e o preço dele?" depois de citar um prato.
- **Resposta em streaming** — o texto aparece conforme o modelo gera, sem tela parada esperando a resposta completa.
- **Resiliência** — se o modelo principal falhar ou demorar, a resposta cai automaticamente para um modelo alternativo, e a interface informa qual respondeu.
- **Chave pela interface** — o usuário cola a própria Gemini API Key na barra lateral; nada de credencial no código.
- **Índice incremental** — o ChromaDB só é reconstruído quando o `cardapio.txt` muda (comparação por hash).

---

## 🔎 Como funciona

```
                    pergunta do usuário
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Embedding local (MiniLM)     │
            │  ChromaDB → 6 trechos         │
            └───────────────┬───────────────┘
                            │  contexto
                            ▼
            ┌───────────────────────────────┐
            │  Prompt = system + histórico  │
            │  + contexto + pergunta        │
            └───────────────┬───────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  Gemini 3.6 Flash             │
            │  (Interactions API)           │
            └───────────────┬───────────────┘
                            │
                            ▼
                 resposta + trechos de origem
```

**Detalhes de cada etapa:**

1. **Indexação** — `cardapio.txt` é dividido em blocos, um por prato (`## Nome do prato` + corpo). Cada bloco vira um documento com metadados: prato, categoria e preço.
2. **Embeddings** — gerados localmente pelo modelo padrão do ChromaDB (`all-MiniLM-L6-v2`, via ONNX). Nenhuma chamada de API nessa etapa, nenhum custo.
3. **Recuperação** — a pergunta é embeddada e o banco devolve os 6 trechos mais próximos por distância de cosseno.
4. **Geração** — contexto + pergunta + histórico recente vão para o `gemini-3.6-flash`, com temperatura `0.2` e um `system_instruction` que proíbe usar conhecimento externo.
5. **Interface** — o Streamlit exibe a resposta, o modelo que respondeu e os trechos consultados.

---

## 🧰 Stack

| Camada | Tecnologia |
|---|---|
| Interface | Streamlit |
| Orquestração | Python puro (sem framework de agentes) |
| Vetor store | ChromaDB (persistente em disco) |
| Embeddings | `all-MiniLM-L6-v2` (local, ONNX) |
| LLM | Google Gemini `gemini-3.6-flash` (Interactions API, SDK `google-genai`) |

---

## 📁 Estrutura

```
.
├── app.py              # interface Streamlit: chat, sidebar, fontes
├── rag.py              # chunking, indexação no ChromaDB e busca
├── llm.py              # prompt de sistema, cliente Gemini, fallback e erros
├── config.py           # constantes: modelo, k, caminhos, timeout
├── cardapio.txt        # base de conhecimento (cardápio do restaurante)
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

Arquivos gerados em tempo de execução e ignorados pelo Git: `chroma_db/`, `.index_hash`, `.venv/`, `__pycache__/`.

---

## ▶️ Rodando localmente

Pré-requisito: Python 3.11 ou superior.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Abre em `http://localhost:8501`. Na barra lateral:

1. Cole sua **Gemini API Key** (gratuita em https://aistudio.google.com/apikey).
2. Clique em **🔌 Testar conexão** para validar antes de conversar.

A primeira execução indexa o cardápio (~10 segundos). Nas seguintes, reutiliza o índice salvo.

> A chave pode ficar em um `.env` (copie o `.env.example`), mas a informada na interface tem prioridade. Nenhuma chave é gravada em disco pelo app.

---

## 💬 Exemplos de uso

| Pergunta | Comportamento esperado |
|---|---|
| Quanto custa o Smash Bacon Duplo? | Preço exato, retirado do cardápio |
| Quais opções veganas vocês têm? | Lista apenas os itens marcados como veganos |
| O hambúrguer tem glúten? | Responde pelo que está no cardápio e lembra da cozinha compartilhada |
| Tem sushi de salmão? | **Recusa** — não existe no cardápio |
| Qual é o CNPJ do restaurante? | **Recusa** — dado não presente na base |

---

## 🧠 Decisões técnicas

- **Chunk por prato, não por tamanho fixo.** Cada prato é uma unidade semântica completa (nome, descrição, preço, restrições). Dividir por número de caracteres cortaria preço do nome e pioraria a recuperação.
- **Embeddings locais, LLM na nuvem.** Só a geração custa API. A busca é gratuita, rápida e funciona offline.
- **Guardrail no prompt, não no código.** O modelo recebe instrução explícita de recusar o que não está no contexto, e a temperatura baixa reduz variação. Respostas do tipo "não encontrei" são o comportamento correto, não uma falha.
- **Histórico como steps da Interactions API.** As últimas 6 mensagens são enviadas no formato `user_input` / `model_output`, sem estado no servidor (`store=False`), o que mantém a conversa reproduzível.
- **Streaming desde a primeira versão.** A interface usa `st.write_stream` e a Interactions API em modo `stream`, então o usuário vê o texto surgindo em ~1s em vez de esperar o modelo terminar. Como o Streamlit renderiza Markdown, os `$` de "R$" são escapados durante o stream para não virarem fórmula LaTeX.
- **Fallback entre modelos, e não retentativa.** O SDK conta *timeout* como erro transitório e, com múltiplas tentativas, insiste no mesmo modelo por minutos. Por isso o cliente usa `attempts=1`: se o modelo principal devolver `503`/`429` ou estourar o tempo, a chamada vai direto para `gemini-3.5-flash-lite`. O fallback só acontece enquanto nada foi exibido — se a conexão cair no meio do texto, é melhor mostrar o erro do que emendar duas respostas.
- **Índice com hash.** Evita reindexar a cada reinício e evita índices desatualizados quando o cardápio muda.

---

## ⚠️ Limitações e próximos passos

- **Uma instância por vez.** Rodar dois processos Streamlit contra o mesmo `chroma_db` corrompe o acesso ao tenant. Para reiniciar do zero: apague `chroma_db/` e `.index_hash`.
- **Sem reranking.** Recuperação é puramente vetorial; um cross-encoder melhoraria perguntas ambíguas.
- **Sem avaliação automatizada.** O próximo passo natural é um conjunto de perguntas com respostas esperadas e medição de acerto/recusa.
- **Base estática.** Atualizações do cardápio exigem reindexação (automática, via hash).

---

## 📄 Licença

MIT — use, modifique e compartilhe à vontade. Veja [LICENSE](LICENSE).
