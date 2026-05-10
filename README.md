# 📚 Large-Scale Book Recommendation System

> **SDS 2412 — Analysis of Large Datasets** | Jomo Kenyatta University of Agriculture and Technology

A complete, end-to-end large-scale recommendation system built on the Book-Crossing dataset — from raw CSV files to a deployed REST API — using production-grade distributed computing tools and machine learning.

---

## 👥 Team Members

| Name | Registration Number |
|------|-------------------|
| Rodney Okoth | SCT213-C002-0063/2022 | 
| Sandra Jebet | SCT213-C002-0117/2022 | 
| Effie Auma | SCT213-C002-0077/2022 | 
| Leslie Gideon | SCT213-C002-0062/2022 | 
| Kandy Genga | SCT213-C002-00__/2022 | 

---

## 🏗️ System Architecture

This project implements **Lambda Architecture** — a design pattern that runs two data pipelines in parallel:

```
Raw Data (Books, Ratings, Users CSVs)
            │
    ┌───────┴───────┐
    │               │
 Batch Layer    Speed Layer
 (Spark)        (Stream Simulator)
    │               │
    └───────┬───────┘
            │
        ML Layer
     (ALS + TF-IDF)
            │
      Serving Layer
   (Flask REST API + LRU Cache)
            │
    Book Recommendations
```

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Batch | Apache Spark + Parquet | Accurate, scheduled processing of all 433K ratings |
| Speed | Python Stream Simulator | Real-time event processing at 15,000 events/sec |
| ML | ALS + TF-IDF Hybrid | Collaborative + content-based recommendations |
| Serving | Flask REST API + LRU Cache | Delivers Top-K recommendations with 60% cache hit rate |

---

## 📊 Dataset

**Book-Crossing Dataset**

| Stat | Value |
|------|-------|
| Total Ratings | 1.1 million |
| Unique Books | 270,000 |
| Unique Users | 278,000 |
| Explicit Ratings (1–10) | 433,000 (30%) |
| Implicit Ratings (score=0) | 770,000 (70%) |
| Matrix Sparsity | **99.8% empty** |

The core challenge: finding signal in an almost entirely empty user-item matrix.

---

## 🚀 Milestones

### Milestone 1 & 2 — Data Foundations & Distributed Processing
- Loaded and cleaned 3 raw CSVs (Books, Ratings, Users)
- Separated implicit (score=0) from explicit (score 1–10) ratings
- Saved all data to **Parquet** format (5–10x faster than CSV)
- Loaded into **Apache Spark**, partitioned into 10 shards by User-ID
- Ran **MapReduce** batch jobs to build user profiles and item popularity tables
- Confirmed near-linear scalability: 0.75s for 433K rows → projected 4.3s for 25M rows on 10 nodes

### Milestone 3 — Streaming & Real-Time Systems
- Simulated an **Apache Kafka** stream using a Python event loop
- Built a **sliding window** of 500 events for real-time trending book tracking
- Implemented **Bloom Filters** for O(1) already-read book lookups (625 bytes/user, 40x more memory-efficient than dictionaries)
- Achieved **15,000 events/sec** throughput at **0.016ms** average latency

### Milestone 4 — Scalable Machine Learning
- **Collaborative Filtering (ALS):** Matrix factorisation with 50 latent factors trained in Spark
- **Content-Based Filtering:** TF-IDF vectors (271K books × 5K features) with cosine similarity
- **Hybrid Model:** 70% ALS + 30% content-based weighting
- Results: RMSE 2.487 | MAE 1.954 | **Precision@10: 0.86**

### Milestone 5 — Optimisation & Deployment
- **LRU Cache:** 100-entry capacity, 60% hit rate, cached requests return in <1ms
- **Flask REST API:** `/recommend` and `/health` endpoints
- **Drift Monitor:** Rolling MAE window of 100 predictions, stable at 1.99

### Milestone 6 — Full Integration & Capstone
- Single end-to-end pipeline function activating all 5 layers in sequence
- **MMR Diversity Injection:** Penalises repeated authors by 0.3, preventing filter bubbles
- Tested on 20 real users end-to-end across all layers simultaneously
- Scale projection: 50 Spark nodes → batch in ~10 seconds, stream at 750K events/sec

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Streaming Throughput | 15,000 events/sec |
| Streaming Latency | 0.016ms per event |
| ALS RMSE | 2.487 |
| ALS MAE | 1.954 |
| **Precision@10** | **0.86** |
| LRU Cache Hit Rate | 60% |
| Cached Response Time | < 1ms |
| Drift Monitor MAE | Stable at 1.99 |

---

## 🔧 Tech Stack

- **Python** — core language
- **Apache Spark (PySpark)** — distributed processing and ALS training
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — TF-IDF vectorisation and cosine similarity
- **Flask** — REST API deployment
- **Parquet** — columnar storage format
- **Bloom Filters** — approximate membership testing
- **Google Colab** — development environment

---

## 📁 Project Structure

```
book-recommender-large-scale/
│
├── SDS_2412_Analysis_of_Large_Datasets.ipynb   # Main notebook (all 6 milestones)
├── README.md                                    # This file
└── SDS2412_Book_Recommender_Submission.docx    # Submission document
```

---

## ▶️ How to Run

1. Open `SDS_2412_Analysis_of_Large_Datasets.ipynb` in Google Colab
2. Download the Book-Crossing dataset and upload the three CSV files:
   - `Books.csv`
   - `Ratings.csv`
   - `Users.csv`
3. Run all cells in order from top to bottom — each milestone builds on the previous one
4. The final cell prints a complete system summary with all metrics

> **Note:** PySpark is installed automatically in the setup cell. No manual installation required.

---

## 💡 Key Innovations

- **MMR Diversity Injection** — prevents the filter bubble problem by penalising repeated authors in the top-10 list, guaranteeing variety without losing relevance
- **Hybrid ML Engine** — combines collaborative filtering (who liked what) with content-based filtering (what the book is about) for more robust recommendations
- **Production-Ready Architecture** — the same Lambda Architecture and code, deployed on a real Kafka cluster with 50 Spark nodes, would serve millions of users with no structural changes

---

## 📝 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Year column had text mixed in | `pd.to_numeric(errors='coerce')` to safely convert or discard |
| 40% of users had no age | Filled with median age instead of dropping rows |
| Kafka cannot run in a notebook | Simulated with Python event loop — identical processing logic |
| Evaluation looped 200 Spark queries | Pulled all predictions into pandas in one batch job |
| Content-based scores near zero | Documented honestly — ALS correctly dominates for active users |

---

*Built for SDS 2412 — Analysis of Large Datasets | May 2026*
