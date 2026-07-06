# Hybrid Search Implementation Guide

Hybrid search combines vector similarity search with traditional keyword-based
retrieval to achieve the best of both worlds.

## Architecture

The hybrid retriever consists of three main stages:

1. Vector Search: Uses Milvus with COSINE similarity to find semantically similar chunks
2. BM25 Search: Uses the classic BM25 algorithm with jieba tokenization for keyword matching
3. RRF Fusion: Reciprocal Rank Fusion merges both result lists

## RRF Formula

RRF_score(d) = sum(w_i / (k + rank_i(d)))

Where:
- w_i: weight for each retriever (default: vector=0.6, bm25=0.4)
- k: smoothing parameter (default: 60)
- rank_i(d): rank of document d in retriever i's result list

## Performance Metrics

| Query Type       | Vector Only | Hybrid   | Improvement |
|------------------|-------------|----------|-------------|
| Semantic queries | 0.89        | 0.87     | -2%         |
| Keyword queries  | 0.34        | 0.88     | +159%       |
| Mixed queries    | 0.65        | 0.92     | +42%        |

## Implementation Tips

1. Use jieba for Chinese text tokenization in BM25
2. Set k=60 for RRF to avoid rank-1 dominance
3. Weight vector search slightly higher (0.6 vs 0.4) for better semantic coverage
4. Rebuild BM25 index after inserting new documents