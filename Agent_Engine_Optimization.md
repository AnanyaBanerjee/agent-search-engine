# Agent Engine Optimization

How the Agent Search Engine ranks and re-ranks results to surface the most relevant, highest-quality agents — beyond pure semantic similarity.

---

## The Problem With Semantic-Only Ranking

A cosine similarity score tells you how close a query is to an agent's description in embedding space. It answers *"does this agent mention the right words?"*, not *"is this agent the best choice?"*.

Two agents with identical descriptions but wildly different track records would receive the same score. A brand-new agent with a perfect description would beat an established, well-reviewed one just because its card was written better.

Optimization adds real-world performance signals on top of semantic similarity to fix this.

---

## The Ranking Formula

Every candidate agent receives a composite score:

```
final = W_SEMANTIC   * semantic_score
      + W_CTR        * ctr_score
      + W_RECENCY    * recency_score
      + W_AFFINITY   * task_affinity_score
      + W_REPUTATION * reputation_score
      + cold_start_bonus
```

| Signal | Default Weight | What it measures |
|---|---|---|
| Semantic similarity | 0.50 | Cosine distance between query and agent card embedding |
| CTR | 0.20 | Click-through rate: clicks / impressions |
| Recency | 0.15 | Exponentially decayed recent clicks |
| Task affinity | 0.10 | Cosine similarity between current query and agent's click-query centroid |
| Reputation | 0.05 | Average review score (1–5 → 0–1) |
| Cold-start bonus | +0.10 | One-time boost for new, unseen agents |

All weights are tunable via environment variables (`RANK_WEIGHT_SEMANTIC`, `RANK_WEIGHT_CTR`, etc.).

---

## Signal Deep-Dives

### 1. Semantic Similarity (50%)

The base signal. Agent cards (name, description, skills, tags) are embedded using `BAAI/bge-small-en-v1.5` (384 dimensions, ONNX runtime). At search time the query is embedded and cosine similarity is computed against all agent embeddings in RAM.

This is intentionally the dominant signal — the other signals are corrections, not replacements.

### 2. Click-Through Rate (20%)

```
CTR = total_clicks / max(total_impressions, 1)
```

Every time an agent appears in search results, an impression is logged. Every time a user clicks an agent card, a click is logged (along with the active search query for task affinity).

CTR rewards agents that users consistently choose when presented. It's a direct signal of perceived usefulness.

**Normalisation**: CTR is normalised to [0, 1] within the current candidate set (divide by the max CTR in the pool), so absolute click counts don't dominate over semantic relevance.

### 3. Recency Score (15%)

Recent popularity matters more than historical popularity. A previously popular agent that hasn't been clicked in months should yield to a currently active one.

Each click is weighted by how recently it occurred:

```
recency_raw = Σ exp(−λ × days_since_click)
```

With `λ = 0.10` (tunable via `RANK_RECENCY_DECAY_LAMBDA`), a click today contributes ~1.0; a click 7 days ago contributes ~0.5; a click 30 days ago contributes ~0.05.

Like CTR, recency_raw is normalised within the candidate set before weighting.

### 4. Task Affinity (10%)

Even a high-CTR agent might be popular for *different tasks* than what the user is currently asking. Task affinity measures alignment between the current query and the *kinds of tasks* this agent has historically been clicked for.

**How it works**: When a user clicks an agent, the active search query is embedded and used to update a per-agent **task centroid** — a running average of all click-query embeddings. Task affinity is the cosine similarity between the current query embedding and this centroid.

**Online update formula**:
```
new_centroid = (old_centroid × n + query_embedding) / (n + 1)
```

This is computed in O(1) without storing all historical query embeddings.

If an agent has never been clicked with a query (no centroid yet), task affinity defaults to 0 — it neither helps nor hurts.

### 5. Reputation Score (5%)

Human (or agent) reviews feed directly into ranking. Each review carries a score from 1–5. The average rating maps linearly to [0, 1]:

```
reputation = avg_rating / 5.0
```

An agent with a 5-star average gets the full 0.05 boost. An agent with no reviews gets 0.

Reviews are one-per-reviewer-per-agent (upsert on the `(agent_id, reviewer_id)` unique constraint), preventing ballot stuffing from a single source.

### 6. Cold-Start Bonus

New agents have zero clicks, zero impressions, and no reviews. Without an intervention, they would always rank below established agents — even if they are objectively better. This creates a chicken-and-egg problem: agents need to be seen to accumulate signals, but they need signals to be seen.

The cold-start bonus applies a flat +0.10 score boost to any agent that is:
- Registered within the last 7 days (`RANK_COLD_START_MAX_DAYS`), **OR**
- Has fewer than 5 total clicks (`RANK_COLD_START_MIN_CLICKS`)

Once an agent accumulates enough clicks, the bonus disappears and it competes on merit.

---

## The Reranking Pipeline

The search pipeline runs in two stages:

**Stage 1 — Semantic retrieval** (fast, approximate):
Embed the query, run cosine similarity against all agents, retrieve `top_k × 3` candidates (the pool multiplier, tunable via `RANK_POOL_MULTIPLIER`). This over-fetches to give the reranker enough material to work with.

**Stage 2 — Multi-signal reranking** (precise, on the candidate pool):
For the candidate pool, fetch all signals in a single DB pass. Compute composite scores. Re-sort by final score. Return the top `top_k` results.

The pool multiplier is the key tradeoff: larger pools give the reranker more candidates to promote (better quality), at the cost of more DB reads and computation. The default of 3× is safe up to tens of thousands of agents.

---

## Data Flow

```
User query
    │
    ▼
embed_text(query)  →  384-dim float32 vector
    │
    ▼
cosine_search(top_k × 3)  →  [(agent_id, semantic_score)]
    │
    ▼
get_ranking_signals(agent_ids)  →  {agent_id: {ctr, recency_raw, task_centroid, avg_rating, ...}}
    │
    ▼
rerank(hits, query_vec, signals)  →  [(agent_id, composite_score)]  (sorted)
    │
    ▼
Trim to top_k, fetch cards  →  SearchResponse
    │
    ├── log_impressions(agent_ids, query)
    └── log_search(query, count, tags)

User clicks agent card
    │
    ├── log_agent_click(agent_id, query_text)
    └── update_task_centroid(agent_id, embed(query_text))
```

---

## Database Tables

| Table | Purpose |
|---|---|
| `agent_clicks` | Click events with `query_text` (the query active when clicked) |
| `agent_impressions` | Every search result shown, with the query |
| `agent_task_centroids` | Per-agent running-average query embedding centroid |
| `agent_reviews` | User/agent reviews with score (1–5) and comment; unique per reviewer per agent |

---

## New API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/agents/{id}/click` | — | Log click; body `{"query": "..."}` updates task affinity |
| POST | `/agents/{id}/review` | — | Submit or update a review |
| GET | `/agents/{id}/reviews` | — | List all reviews + average rating |

---

## Tuning Reference

All ranking parameters are configurable via environment variables:

| Variable | Default | Effect |
|---|---|---|
| `RANK_WEIGHT_SEMANTIC` | `0.50` | Weight for cosine similarity |
| `RANK_WEIGHT_CTR` | `0.20` | Weight for click-through rate |
| `RANK_WEIGHT_RECENCY` | `0.15` | Weight for recency-decayed clicks |
| `RANK_WEIGHT_TASK_AFFINITY` | `0.10` | Weight for task affinity |
| `RANK_WEIGHT_REPUTATION` | `0.05` | Weight for review score |
| `RANK_RECENCY_DECAY_LAMBDA` | `0.10` | Decay rate λ for recency (higher = faster decay) |
| `RANK_COLD_START_BONUS` | `0.10` | Flat score boost for new/unseen agents |
| `RANK_COLD_START_MIN_CLICKS` | `5` | Clicks threshold below which cold-start applies |
| `RANK_COLD_START_MAX_DAYS` | `7` | Age threshold (days) below which cold-start applies |
| `RANK_POOL_MULTIPLIER` | `3` | Candidate pool size = top_k × this |

---

## Design Decisions

**Why a linear weighted sum rather than a learning-to-rank model?**

A learned model (LambdaMART, etc.) would be more accurate but requires labelled training data, a training pipeline, model versioning, and regular retraining. A weighted sum is interpretable, tunable without re-deployment, and sufficient at current scale. The architecture doesn't preclude moving to a learned model later — the signals feed the same regardless of the scoring function.

**Why normalise CTR and recency within the candidate set?**

Global normalisation (across all agents) would cause scores to compress badly when one agent has an enormous head start. Candidate-local normalisation ensures the spread of scores is meaningful within each result set, regardless of global popularity distributions.

**Why keep reputation weight low (5%)?**

Reviews are sparse in early deployment. A low weight prevents the system from over-indexing on a handful of reviews. As the review corpus grows, this weight can be raised via the environment variable without a code change.

**Why store the centroid as a running average rather than all click embeddings?**

Storing every click-query embedding would grow unboundedly and require a batch aggregation step at query time. A running average updates in O(1) and occupies a fixed 1.5 KB per agent (384 × 4 bytes), regardless of how many clicks the agent has accumulated.
