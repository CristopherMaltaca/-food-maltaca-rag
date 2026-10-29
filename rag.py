"""
Camada de recuperação (o "R" de RAG).

Responsabilidades:
1. Ler o cardapio.txt e dividi-lo em um trecho por prato.
2. Indexar esses trechos no ChromaDB (banco vetorial local).
3. Buscar os trechos mais parecidos com a pergunta do usuário.
"""

import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings

import config

_colecao = None


def _hash_do_cardapio() -> str:
    """Gera uma 'impressão digital' do cardápio para detectar alterações."""
    conteudo = Path(config.CARDAPIO_PATH).read_bytes()
    return hashlib.sha256(conteudo).hexdigest()


def _campo(bloco: str, nome: str) -> str:
    """Extrai o valor de uma linha do tipo 'Preço: R$ 32,90'."""
    encontrado = re.search(rf"^{nome}:\s*(.+)$", bloco, flags=re.MULTILINE)
    return encontrado.group(1).strip() if encontrado else ""


def _slug(texto: str) -> str:
    """Converte 'Smash Bacon Duplo' em 'smash-bacon-duplo' (usado como ID)."""
    return re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")


def carregar_chunks() -> list[dict]:
    """
    Divide o cardápio em trechos, um por prato.

    O arquivo usa uma estrutura simples: cada prato começa com "##" e os
    blocos são separados por uma linha em branco. Isso é mais previsível do
    que cortar o texto a cada N caracteres, porque cada trecho fica
    semanticamente completo (nome + preço + ingredientes + restrições).
    """
    texto = Path(config.CARDAPIO_PATH).read_text(encoding="utf-8")
    blocos = re.split(r"\n\s*\n", texto)

    chunks = []
    for bloco in blocos:
        bloco = bloco.strip()
        if not bloco.startswith("##"):
            continue

        nome = bloco.splitlines()[0].replace("##", "").strip()
        chunks.append(
            {
                "id": _slug(nome),
                "texto": bloco,
                "prato": nome,
                "categoria": _campo(bloco, "Categoria"),
                "preco": _campo(bloco, "Preço"),
                "restricoes": _campo(bloco, "Restrições"),
                "alergenos": _campo(bloco, "Alérgenos"),
            }
        )

    return chunks


def _listar_nomes(cliente) -> list[str]:
    """
    Lista as collections existentes.

    O ChromaDB já retornou nomes (versões antigas) e objetos (versões novas)
    neste mesmo método; tratamos os dois casos para não quebrar.
    """
    return [c.name if hasattr(c, "name") else c for c in cliente.list_collections()]


def indexar_cardapio(forcar: bool = False):
    """
    Garante que o cardápio esteja indexado e devolve a coleção pronta.

    Só reindexa quando: o índice está vazio, o cardápio mudou (hash diferente)
    ou o parâmetro 'forcar' é True.
    """
    cliente = chromadb.PersistentClient(
        path=str(config.CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )

    hash_atual = _hash_do_cardapio()
    hash_salvo = (
        Path(config.HASH_PATH).read_text(encoding="utf-8").strip()
        if Path(config.HASH_PATH).exists()
        else ""
    )

    existe = config.COLLECTION_NAME in _listar_nomes(cliente)
    colecao = cliente.get_collection(config.COLLECTION_NAME) if existe else None

    indice_vazio = colecao is None or colecao.count() == 0
    cardapio_mudou = hash_salvo != hash_atual

    if forcar or indice_vazio or cardapio_mudou:
        if colecao is not None:
            cliente.delete_collection(config.COLLECTION_NAME)

        colecao = cliente.create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        chunks = carregar_chunks()
        colecao.add(
            ids=[c["id"] for c in chunks],
            documents=[c["texto"] for c in chunks],
            metadatas=[
                {
                    "prato": c["prato"],
                    "categoria": c["categoria"],
                    "preco": c["preco"],
                    "restricoes": c["restricoes"],
                    "alergenos": c["alergenos"],
                }
                for c in chunks
            ],
        )

        Path(config.HASH_PATH).write_text(hash_atual, encoding="utf-8")

    return colecao


def obter_colecao(forcar_reindexacao: bool = False):
    """Devolve a coleção em cache (ou cria/atualiza se necessário)."""
    global _colecao

    if _colecao is None or forcar_reindexacao:
        _colecao = indexar_cardapio(forcar=forcar_reindexacao)

    return _colecao


def buscar(pergunta: str, k: int = None) -> list[dict]:
    """
    Busca os k trechos do cardápio mais parecidos com a pergunta.

    Retorna uma lista de dicionários com o texto, o nome do prato e a
    distância (quanto menor, mais relevante).
    """
    k = k or config.K_RESULTADOS
    colecao = obter_colecao()

    k = min(k, colecao.count())

    resultado = colecao.query(
        query_texts=[pergunta],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documentos = resultado["documents"][0]
    metadados = resultado["metadatas"][0]
    distancias = resultado["distances"][0]

    return [
        {
            "texto": texto,
            "prato": meta.get("prato", ""),
            "categoria": meta.get("categoria", ""),
            "distancia": distancia,
        }
        for texto, meta, distancia in zip(documentos, metadados, distancias)
    ]
