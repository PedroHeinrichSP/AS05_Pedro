import os
import gradio as gr
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 🔐 Variáveis de ambiente
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")

# ⚙️ Inicializar Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# ⚙️ Inicializar Pinecone SDK v3
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Embedding LangChain (nova API)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    text_key="text",
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 📥 Indexar PDFs com botão
def indexar_pdfs_interface(pasta="data"):
    try:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        doc_id = 0
        arquivos_indexados = 0

        for nome_arquivo in os.listdir(pasta):
            if not nome_arquivo.endswith(".pdf"):
                continue
            pdf = PdfReader(os.path.join(pasta, nome_arquivo))
            texto = "\n".join([p.extract_text() or "" for p in pdf.pages])
            partes = splitter.split_text(texto)
            embeddings = model.encode(partes)

            for i, emb in enumerate(embeddings):
                index.upsert(vectors=[{
                    "id": f"{nome_arquivo}-{doc_id}",
                    "values": emb.tolist(),
                    "metadata": {"text": partes[i]}
                }])
                doc_id += 1
            arquivos_indexados += 1

        return f"✅ {arquivos_indexados} PDFs indexados com sucesso."
    except Exception as e:
        return f"❌ Erro ao indexar: {e}"

# 💬 Função de resposta
def responder(pergunta):
    docs = retriever.get_relevant_documents(pergunta)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Use o contexto abaixo para responder à pergunta.
CONTEXTO:
{contexto}

PERGUNTA: {pergunta}
RESPOSTA:"""

    model = genai.GenerativeModel("gemma-3-1b-it")
    resposta = model.generate_content(prompt)
    return resposta.text

# 🖼️ Interface
with gr.Blocks() as iface:
    gr.Markdown("## 🤖 Assistente de PDFs com Gemini")
    gr.Markdown("Faça perguntas com base nos PDFs que você indexar.")

    with gr.Row():
        entrada = gr.Textbox(label="Pergunta")
        saida = gr.Textbox(label="Resposta")

    btn_perguntar = gr.Button("Responder")
    btn_perguntar.click(fn=responder, inputs=entrada, outputs=saida)

    gr.Markdown("---")
    btn_indexar = gr.Button("📥 Indexar PDFs da pasta /data")
    log_indexacao = gr.Textbox(label="Log de Indexação")
    btn_indexar.click(fn=indexar_pdfs_interface, inputs=[], outputs=log_indexacao)

# 🚀 Lançar app
if __name__ == "__main__":
    iface.launch()
