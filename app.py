import os
import gradio as gr
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from pinecone import ServerlessSpec
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
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Embedding LangChain (nova API)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = PineconeVectorStore(
    pinecone_index=index,
    embedding=embedding_model,
    text_key="text",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 📥 Função para carregar e indexar PDFs
def indexar_pdfs(pasta="data"):
    print("🔄 Indexando PDFs...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc_id = 0

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

    print("✅ Indexação concluída.")

# 💬 Função de resposta
def responder(pergunta):
    docs = retriever.get_relevant_documents(pergunta)
    contexto = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Use o contexto abaixo para responder à pergunta.
CONTEXTO:
{contexto}

PERGUNTA: {pergunta}
RESPOSTA:"""

    model = genai.GenerativeModel("gemini-pro")
    resposta = model.generate_content(prompt)
    return resposta.text

# 🖼️ Interface
iface = gr.Interface(
    fn=responder,
    inputs=gr.Textbox(label="Pergunta"),
    outputs=gr.Textbox(label="Resposta"),
    title="🤖 Assistente de PDFs com Gemini",
    description="Faça perguntas sobre os documentos PDF indexados."
)

if __name__ == "__main__":
    # Descomente para indexar seus PDFs:
    # indexar_pdfs()
    iface.launch()