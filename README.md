# AS05: Implementação de Assistente Conversacional Baseado em LLM para Tópicos em Computação III
Assistente conversacional baseado em LLM (Gemini), que indexa documentos PDF em vetores semânticos e responde perguntas com base nos conteúdos

## Instalação
```bash
git clone https://github.com/PedroHeinrichSP/AS05_Pedro
cd AS05_Pedro
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Configuração

Crie um arquivo .env com:

GOOGLE_API_KEY=...
PINECONE_API_KEY=...
PINECONE_ENV=...
PINECONE_INDEX_NAME=chat-pdf-index

Então coloque os PDFs na pasta data/