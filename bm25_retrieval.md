# BM25 Retrieval Explained

## What is BM25?

BM25 (Okapi BM25) is a keyword-based retrieval algorithm used to rank
documents based on relevance to a query.

## Intuition

BM25 scores documents based on: - Term frequency (how often a word
appears) - Inverse document frequency (rarity of the word) - Document
length normalization

## Formula

score(D, Q) = Σ IDF(q_i) \* ((f(q_i, D) \* (k1 + 1)) / (f(q_i, D) + k1
\* (1 - b + b \* \|D\| / avgdl)))

## Key Features

-   Term frequency saturation
-   Document length normalization
-   Rare words get higher importance

## BM25 vs Vector Search

  Feature    BM25              Vector Search
  ---------- ----------------- -------------------------
  Type       Keyword-based     Semantic
  Matches    Exact words       Meaning
  Speed      Fast              Slower
  Weakness   Misses synonyms   May miss exact keywords

## BM25 in RAG

Used in hybrid retrieval with vector search: - BM25 → keyword
precision - Vector → semantic understanding

## Example (Conceptual)

    docs_bm25 = bm25_retriever.get_docs(query)
    docs_vector = vector_retriever.get_docs(query)
    final_docs = merge_and_rerank(docs_bm25, docs_vector)

## Tools Using BM25

-   Elasticsearch
-   Apache Solr
-   Whoosh
-   LangChain BM25Retriever

## Mental Model

BM25 = Exact match expert Vector DB = Meaning expert Reranker = Final
judge
