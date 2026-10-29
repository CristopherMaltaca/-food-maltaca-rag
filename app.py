"""
Interface web do assistente de cardápio (Streamlit).

Fluxo da aplicação:
1. Garante que o cardápio está indexado no ChromaDB.
2. Recebe a pergunta e a chave da API pela interface.
3. Busca os trechos relevantes e chama o Gemini.
4. Mostra a resposta e, opcionalmente, os trechos usados como fonte.

Executar com:  streamlit run app.py
"""

import streamlit as st

import config
import llm
import rag

st.set_page_config(
    page_title="Food Maltaca | Assistente de Cardápio",
    page_icon="🍔",
    layout="centered",
)

EXEMPLOS = [
    "Quais opções veganas vocês têm?",
    "Quanto custa o Smash Bacon Duplo?",
    "O Brownie com Sorvete tem nozes?",
    "Tem pão sem glúten? Quanto custa?",
    "Vocês entregam na Vila Mariana?",
    "Qual é o CNPJ do restaurante?",
]


try:
    with st.spinner("Preparando o cardápio..."):
        colecao = rag.obter_colecao()
except Exception as erro:
    st.error(
        "Não foi possível indexar o cardápio. Verifique sua conexão na "
        f"primeira execução e tente novamente.\n\nDetalhe técnico: {erro}"
    )
    st.stop()


with st.sidebar:
    st.title("🍔 Food Maltaca")
    st.caption("Assistente de cardápio com RAG")

    st.divider()
    st.subheader("Chave da API")
    st.text_input(
        "Gemini API Key",
        type="password",
        key="api_key_input",
        placeholder="Cole sua chave aqui",
        help="Obtenha uma chave gratuita em https://aistudio.google.com/apikey",
    )
    st.caption("A chave fica apenas na memória desta sessão e não é gravada em disco.")

    if st.button("🔌 Testar conexão", width="stretch"):
        chave_teste = (
            st.session_state.get("api_key_input", "").strip() or config.GEMINI_API_KEY
        )
        if not chave_teste:
            st.warning("Cole a chave antes de testar.")
        else:
            with st.spinner("Testando..."):
                ok, detalhe = llm.testar_conexao(chave_teste)
            if ok:
                st.success(detalhe)
            else:
                st.error(detalhe)

    st.divider()
    st.subheader("Base de conhecimento")

    if st.button("🔄 Reindexar cardápio", width="stretch"):
        with st.spinner("Reindexando..."):
            colecao = rag.obter_colecao(forcar_reindexacao=True)
        st.success("Cardápio reindexado.")

    if st.button("🧹 Limpar conversa", width="stretch"):
        st.session_state.mensagens = []
        st.rerun()

    st.caption(f"{colecao.count()} trechos indexados · modelo `{config.MODELO}`")

    st.divider()
    st.caption("RAG + ChromaDB + Gemini · respostas baseadas no cardápio")


st.title("Assistente de Cardápio")
st.caption(
    "Pergunte sobre pratos, preços, ingredientes, restrições alimentares e entrega."
)

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []


def escapar_cifrao(texto: str) -> str:
    """Evita que o Streamlit interprete 'R$ ... R$' como fórmula matemática."""
    return texto.replace("$", "\\$")


def mostrar_fontes(fontes: list[dict]) -> None:
    """Mostra os trechos do cardápio usados na resposta (transparência do RAG)."""
    if not fontes:
        return

    with st.expander(f"🔍 {len(fontes)} trechos do cardápio consultados"):
        for fonte in fontes:
            st.markdown(
                f"**{fonte['prato']}** · {fonte['categoria']} · "
                f"distância `{fonte['distancia']:.3f}`"
            )
            st.code(fonte["texto"], language="text")


if not st.session_state.mensagens:
    st.markdown("**Exemplos para testar:**")
    for exemplo in EXEMPLOS:
        st.markdown(f"- {exemplo}")


for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        st.markdown(escapar_cifrao(mensagem["content"]))
        if mensagem["role"] == "assistant":
            if mensagem.get("modelo"):
                st.caption(f"respondido por `{mensagem['modelo']}`")
            mostrar_fontes(mensagem.get("fontes", []))


pergunta = st.chat_input("Pergunte sobre o cardápio...")

if pergunta:
    chave = st.session_state.get("api_key_input", "").strip() or config.GEMINI_API_KEY

    with st.chat_message("user"):
        st.markdown(pergunta)

    if not chave:
        st.warning("Cole sua Gemini API Key na barra lateral para conversar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Consultando o cardápio..."):
                try:
                    fontes = rag.buscar(pergunta)
                    contexto = "\n\n---\n\n".join(f["texto"] for f in fontes)
                    resposta, modelo = llm.responder(
                        pergunta=pergunta,
                        contexto=contexto,
                        historico=st.session_state.mensagens,
                        api_key=chave,
                    )
                    erro = None
                except Exception as falha:
                    resposta = None
                    modelo = None
                    erro = falha

            if erro is not None:
                st.error(llm.mensagem_de_erro(erro))
            else:
                st.markdown(escapar_cifrao(resposta))
                st.caption(f"respondido por `{modelo}`")
                mostrar_fontes(fontes)

        if erro is None:
            st.session_state.mensagens.append({"role": "user", "content": pergunta})
            st.session_state.mensagens.append(
                {
                    "role": "assistant",
                    "content": resposta,
                    "fontes": fontes,
                    "modelo": modelo,
                }
            )
