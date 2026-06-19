---
name: rag-research
description: Use local retrieval-augmented generation before web search to ground research in saved knowledge-base documents.
---

# RAG Research Skill

## Purpose

Use this skill when a topic may have relevant local context in the knowledge base.

## Procedure

1. Search local documents with `rag_search`.
2. Extract the most relevant passages and source file names.
3. Use web search for current facts, missing context, or verification.
4. Keep local knowledge and web findings consistent.
5. Save reusable conclusions with `save_to_knowledge` when the user asks to remember them or when future research would benefit.
6. If local knowledge conflicts with current web sources, prefer current authoritative sources and mention the discrepancy.

## Output Expectations

- Cite local file names when using local knowledge.
- Cite web sources for current facts.
- Make uncertainty explicit.
- Produce a clear, professional research report.
