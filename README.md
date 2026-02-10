# SimpleAgent with Lang Graph# 🛒 SimpleAgent: RAG Sales Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-orange)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-336791)
![Llama 3](https://img.shields.io/badge/AI-Llama_3.3-purple)

> Um agente de IA inteligente para suporte a vendas, lista de tarefas, histórico e matemática, construído com arquitetura RAG (Retrieval-Augmented Generation), Banco de Dados Vetorial na Nuvem e Verificação de Qualidade.

Este projeto foi desenvolvido como parte de um desafio prático de um projeto **PD&I** na **UFCG (Universidade Federal de Campina Grande)**.

---

## 🧠 Sobre o Projeto

O **SimpleAgent** é um assistente de IA orquestrado via **Grafos de Estado**, projetado para atuar tanto como um especialista em vendas quanto como um assistente pessoal generalista. Suas principais capacidades incluem:

1.  **Roteamento Semântico:** Um "cérebro central" analisa a entrada do usuário e decide dinamicamente o fluxo de execução: direciona para o pipeline de vendas (RAG) ou para o assistente geral (Chatbot).
2.  **RAG com Auto-Reflexão:** Na rota de vendas, ele consulta um banco vetorial (**PostgreSQL/pgvector**), mas vai além: possui um **Verificador de Qualidade** que julga se a informação recuperada é suficiente antes de responder.
3.  **Uso de Ferramentas (Tool Calling):** Diferente de chatbots passivos, este agente pode executar ações reais:
    * 🧮 **Calculadora:** Realiza operações matemáticas precisas.
    * ✅ **Gestão de Tarefas:** Adiciona e gerencia listas de pendências do usuário.
    * 📝 **Resumidor:** Compacta conversas longas para otimizar o contexto.
4.  **Ciclo de Raciocínio (ReAct):** Implementa um loop de execução onde o agente pode pensar, usar uma ferramenta, observar o resultado (via `Memory Manager`) e decidir o próximo passo autonomamente.
5.  **Memória Persistente:** Utiliza `Checkpointers` para manter o contexto da conversa, permitindo interações contínuas e coerentes.

---

## 🏗️ Arquitetura (O Grafo)

O sistema utiliza **LangGraph** para orquestrar o fluxo. Abaixo está a representação exata da máquina de estados do agente:

```mermaid
graph TD
    %% Definição de Início e Fim
    Start([Início]) --> Input[Input Node]
    Input --> Router{Router Decision}

    %% Caminho 1: RAG (Vendas)
    Router -- "É Venda?" --> RAG[RAG Node]
    RAG --> End([Fim])

    %% Caminho 2: Chatbot (Geral + Tools)
    Router -- "Geral / Comandos" --> Chatbot[Chatbot Node]

    %% Lógica de Tools (Ciclo ReAct)
    Chatbot --> ToolsCond{Tools Condition}
    
    %% Se precisar usar ferramenta
    ToolsCond -- "Sim (Calc/Tasks)" --> Tools[ToolNode]
    
    subgraph Ferramentas Disponíveis
        direction TB
        Tools
        T1[Calculator] -.-> Tools
        T2[Manage Tasks] -.-> Tools
        T3[Save Summary] -.-> Tools
    end

    %% Fluxo de volta (Loop)
    Tools --> MemManager[Memory Manager]
    MemManager --> Chatbot

    %% Se não precisar de ferramenta (Resposta Final)
    ToolsCond -- "Não (Responder)" --> End
```
---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Orquestração:** [LangGraph](https://langchain-ai.github.io/langgraph/) & [LangChain](https://www.langchain.com/)
* **LLM (Cérebro):** Llama 3.3 70B Versatile (via [Groq API](https://groq.com/))
* **Banco de Dados:** PostgreSQL Serverless ([Neon.tech](https://neon.tech/))
* **Busca Vetorial:** Extensão `pgvector` para busca de similaridade (Distância Euclidiana `<->`).
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face).

---

## 📂 Estrutura do Repositório

```bash
SimpleAgent/
├── agent.py          # O Cérebro Principal: Define o Grafo, Router e Fluxo.
├── rag_tools.py      # As Ferramentas: Conexão com DB, Busca Vetorial e Verificador.
├── ingest.py         # O ETL: Script que lê o JSON, cria vetores e salva no Banco.
├── state.py          # O Estado: Definição da memória (PrivateState).
├── faq.json          # Os Dados: Base de conhecimento (Perguntas e Respostas).
├── requirements.txt  # Dependências do projeto.
└── .env              # (Ignorado pelo Git) Variáveis de Ambiente e Chaves de API.
```
---

## 🚀 Como Executar Localmente

### 1. Pré-requisitos
* Git instalado.
* Python instalado (versão 3.10 ou superior).
* Conta na **Groq** (para obter a API Key gratuita).
* Conta na **Neon.tech** (para obter o banco Postgres gratuito).

### 2. Instalação

Abra o seu terminal e rode os comandos:

```bash
# Clone o repositório
git clone [https://github.com/alviinz/SimpleAgent.git](https://github.com/alviinz/SimpleAgent.git)
cd SimpleAgent

# Crie um ambiente virtual (Recomendado)
python -m venv .venv

# Ative o ambiente virtual
# No Linux/Mac:
source .venv/bin/activate
# No Windows:
.venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

## ⚙️ Configuração do .env

Crie um arquivo chamado `.env` na raiz do projeto e adicione suas credenciais:

```ini
# Chave da API da Groq (Modelos Llama)
GROQ_API_KEY="gsk_sua_chave_aqui..."

# URL de Conexão do Neon (Postgres)
DATABASE_URL="postgres://usuario:senha@endpoint.neon.tech/neondb?sslmode=require"
```

### 4. Ingestão de Dados (Setup do Banco)

Antes de rodar o agente, você precisa popular o banco de dados com os vetores de conhecimento. O script `ingest.py` lê o arquivo `faq.json`, converte tudo em embeddings e salva no Postgres:

```bash
python ingest.py
```
Aguarde a mensagem: "Success. Database is ready now."

### 5. Rodando o Agente

Agora inicie o chatbot no terminal:

```bash
python agent.py
```

## 🧪 Exemplos de Interação

### ✅ Caso 1: Consulta de Vendas (RAG + VectorDB)
**Usuário:** "Como adiciono um produto ao carrinho?"
**Agente:** "Para adicionar, use o formato: 'Adicionar: [Nome] | qtd: [N]'. [Fonte: FAQ Oficial]"
*(O **Router** detectou intenção de venda, buscou no **Postgres/pgvector**, e o **Verificador** aprovou a resposta baseada no FAQ).*

### 🧮 Caso 2: Uso de Ferramentas (Calculator Tool)
**Usuário:** "Se eu comprar 3 mouses de 50 reais, quanto dá?"
**Agente:** "O total será 150 reais."
*(O Router enviou para o nó **Chatbot**, que decidiu autônomamente chamar a ferramenta `calculator` para processar a conta exata).*

### 📝 Caso 3: Memória e Tarefas (State Management)
**Usuário:** "Adicione 'Ligar para o fornecedor' na minha lista de tarefas."
**Agente:** "Tarefa adicionada com sucesso! Você tem 1 tarefa pendente."
*(O agente acionou a tool `manage_tasks`, atualizou o `PrivateState` e salvou essa informação no `MemorySaver` do grafo).*

### 🛡️ Caso 4: Proteção contra Alucinação (Verifier Guardrail)
**Usuário:** "Vocês aceitam pagamento em Bitcoin?"
**Agente:** "Desculpe, procurei na base oficial e não encontrei informações sobre criptomoedas."
*(O RAG encontrou documentos irrelevantes, e a função `verify_response_quality` bloqueou a resposta para evitar que o bot inventasse uma política inexistente).*

## 🎓 Autor
Desenvolvido por Álvaro L. Estudante de Ciência da Computação @ UFCG.
