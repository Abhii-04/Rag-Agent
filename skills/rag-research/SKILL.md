---
name: rag-research
description: Use local retrieval-augmented generation before web search to ground research in saved sandbox documents.
---

# RAG Research Skill

## Purpose

Use this skill when a topic may have relevant local context in `/sandbox/`.

## Procedure

1. Search `/sandbox/` documents with `rag_search`; it vectorizes supported files and stores the index under the sandbox vector-store folder.
2. Extract the most relevant passages and source file names.
3. Use web search for current facts, missing context, or verification.
4. Keep local sandbox findings and web findings consistent.
5. Save reusable conclusions with `save_to_knowledge` when the user asks to remember them or when future research would benefit.
6. If local sandbox documents conflict with current web sources, prefer current authoritative sources and mention the discrepancy.

## Output Expectations

- Cite local file names when using local sandbox documents.
- Cite web sources for current facts.
- Make uncertainty explicit.
- Produce a clear, professional research report.
