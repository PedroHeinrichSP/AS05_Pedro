# AS05: Implementação de Assistente Conversacional Baseado em LLM para Tópicos em Computação III
Assistente conversacional baseado em LLM (Gemini), que indexa documentos PDF em vetores semânticos e responde perguntas com base nos conteúdos usando a API Gemini da Google (modelo Gemma 3-4B)

## Bibliotecas usadas

- Python 3.10+
- [Gradio](https://gradio.app) para interface web
- [PyPDF2](https://pypi.org/project/PyPDF2/) para leitura de PDFs
- [Google Gemini API](https://developers.generativeai.google/) (modelo `gemma-3-4b-it`)
- [Pinecone](https://www.pinecone.io) para indexação vetorial e busca
- [LangChain](https://python.langchain.com) para manipulação de embeddings e integração com Pinecone
- [Sentence Transformers](https://www.sbert.net/) para geração dos embeddings dos textos

---

## Instalação

```bash
# Baixar do Github
git clone https://github.com/PedroHeinrichSP/AS05_Pedro
cd AS05_Pedro

# Criação de ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

#Dependências
pip install -r requirements.txt

#Executar a aplicação
python app.py
```

## Configuração

Crie um arquivo .env com:

GOOGLE_API_KEY=...
PINECONE_API_KEY=....
PINECONE_INDEX_NAME=...
PINECONE_REGION=...
PINECONE_CLOUD=...

Então coloque os PDFs na pasta data/

## Executar os arquivos

```bash
#Executar a aplicação
python app.py
```

A interface web abrirá no endereço: http://127.0.0.1:7860

## Como usar

- Após os PDFs estarem na página clique em Indexar PDFs
- Digite suas perguntas na caixa de texto e clique em responder

---