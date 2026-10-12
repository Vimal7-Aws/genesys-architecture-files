# BM25 Retrieval -- Advanced Guide for RAG & Kubernetes

## 1. Overview

BM25 (Okapi BM25) is a probabilistic ranking function used in search
engines to score how relevant a document is to a query.

------------------------------------------------------------------------

## 2. Mathematical Foundation

### BM25 Formula

$$
\text{score}(D, Q) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}
$$

### IDF (Inverse Document Frequency)

$$
IDF(q) = \log \left( \frac{N - n(q) + 0.5}{n(q) + 0.5} + 1 \right)
$$

Where: - $f(q_i, D)$ = frequency of term in document\
- $|D|$ = document length\
- $avgdl$ = average document length\
- $k_1$ = term frequency scaling (\~1.2--2.0)\
- $b$ = length normalization (\~0.75)

------------------------------------------------------------------------

## 3. Visual Diagram (Conceptual)

              Query
                |
       ---------------------
       |    Tokenize       |
       ---------------------
                |
       ---------------------
       |  Compute TF       |
       ---------------------
                |
       ---------------------
       |  Apply IDF        |
       ---------------------
                |
       ---------------------
       | Length Normalize  |
       ---------------------
                |
       ---------------------
       |  Final Score      |
       ---------------------
                |
             Ranking

------------------------------------------------------------------------

## 4. BM25 in RAG Architecture

            User Query
                 |
        -------------------
        | Hybrid Retriever |
        -------------------
           /           \
      BM25 (Keyword)   Vector DB (Semantic)
           \           /
        -------------------
        |   Merge + Rank   |
        -------------------
                 |
               LLM
                 |
             Response

------------------------------------------------------------------------

## 5. LangChain Implementation (v1 / LCEL)

### Install

``` bash
pip install langchain langchain-community rank_bm25 faiss-cpu
```

### BM25 Retriever

``` python
from langchain_community.retrievers import BM25Retriever

docs = [
    "Kubernetes is a container orchestration system",
    "LangChain helps build LLM applications",
    "BM25 is used in search engines"
]

retriever = BM25Retriever.from_texts(docs)
results = retriever.invoke("What is Kubernetes?")
print(results)
```

------------------------------------------------------------------------

## 6. Hybrid Retrieval (BM25 + FAISS)

``` python
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Vector store
vector_db = FAISS.from_texts(docs, OpenAIEmbeddings())

def hybrid_search(query):
    bm25_docs = retriever.invoke(query)
    vector_docs = vector_db.similarity_search(query)
    return bm25_docs + vector_docs
```

------------------------------------------------------------------------

## 7. Production RAG on Kubernetes

### Architecture

    [User] 
       |
    [API Gateway]
       |
    [Retriever Service]
       |--- BM25 (ElasticSearch)
       |--- Vector DB (FAISS / Milvus)
       |
    [Reranker Service]
       |
    [LLM Service (vLLM / TGI)]
       |
    [Response]

### Kubernetes Deployment Tips

-   Use **Elasticsearch** for BM25
-   Use **FAISS/Milvus** for vector search
-   Deploy LLM with **vLLM** or **TGI**
-   Use **Horizontal Pod Autoscaling**
-   Cache frequent queries (Redis)

------------------------------------------------------------------------

## 8. When to Use BM25

Use BM25 when: - Exact keyword match matters - Searching logs, docs,
configs - Low latency is critical

Avoid BM25 when: - Semantic meaning is required - Queries use synonyms
or natural language

------------------------------------------------------------------------

## 9. Key Takeaways

-   BM25 = keyword precision
-   Vector DB = semantic understanding
-   Hybrid = best of both worlds
-   Essential for production-grade RAG

------------------------------------------------------------------------

## 10. Next Steps

-   Add reranking (Cross-Encoder)
-   Use hybrid scoring (weighted)
-   Deploy on Kubernetes with autoscaling
