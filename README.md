# SimpleAgent with Lang Graph# 🛒 SimpleAgent: RAG Sales Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-orange)
![Postgres](https://img.shields.io/badge/Postgres-pgvector-336791)
![Llama 3](https://img.shields.io/badge/AI-Llama_3.3-purple)

> Um agente de IA inteligente para suporte a vendas, construído com arquitetura RAG (Retrieval-Augmented Generation), Banco de Dados Vetorial na Nuvem e Verificação de Qualidade.

Este projeto foi desenvolvido como parte de um desafio prático de um projeto **PD&I** na **UFCG (Universidade Federal de Campina Grande)**.

---

## 🧠 Sobre o Projeto

O **SimpleAgent** não é apenas um chatbot simples; é um **Agente Inteligente** baseado em grafos de estado. Ele é capaz de:

1.  **Entender a Intenção:** Um roteador (Router) decide se o usuário quer comprar algo ou se está falando de assuntos aleatórios.
2.  **Recuperar Informação (RAG):** Se for sobre vendas, ele busca documentos relevantes em um banco de dados PostgreSQL com `pgvector`.
3.  **Verificar a Resposta:** Antes de responder, uma "segunda mente" (LLM Verifier) checa se o documento encontrado realmente responde à pergunta, evitando alucinações.
4.  **Manter o Foco:** Se o usuário tentar falar sobre matemática, política ou outros assuntos, o agente recusa educadamente o atendimento (Scope Guard).
5.  **Memória:** Mantém o contexto da conversa, lembrando das mensagens anteriores.

---

## 🏗️ Arquitetura (O Grafo)

O sistema utiliza **LangGraph** para orquestrar o fluxo. Abaixo está a representação lógica do grafo de estados:

```mermaid
graph TD
    Start([Início]) --> Input[Input Node]
    Input --> Router{Router Decision}
    
    Router -- "É Venda?" --> RAG[RAG Node]
    Router -- "Outros Assuntos" --> Refusal[Refusal Node]
    
    RAG --> Check{Verifier Tool}
    Check -- "Contexto Válido" --> Answer[Gerar Resposta com Citação]
    Check -- "Contexto Inválido" --> Fallback[Mensagem de Erro em PT-BR]
    
    Answer --> End([Fim / Aguarda User])
    Fallback --> End
    Refusal --> End
