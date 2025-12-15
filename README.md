# TeamForge — Hybrid Explainable Recommendation System

## Overview

TeamForge is an intelligent recommendation system designed to match users to teams across collaborative activities such as hackathons, technical projects, sports, arts, and other group-based events.

The system goes beyond simple keyword matching by combining:

* structured skill requirements
* natural language understanding of profiles and project descriptions
* collaboration/network signals
* continuous learning from user feedback

The result is a **ranked, explainable, and adaptive recommendation pipeline** suitable for real-world applications.

---

## Key Features

* **Hybrid Recommendation Approach**

  * Skill-based filtering (hard constraints)
  * NLP-based semantic matching for free-text fields
  * Graph-based collaboration similarity
  * Learning-to-rank for final ordering

* **Explainability**

  * Every recommendation includes a human-readable explanation
  * Clear breakdown of contributing signals

* **Online Learning**

  * Learns continuously from user interactions (swipes, accepts, rejects)
  * Uses time-decayed feedback to adapt to recent behavior
  * Balances exploration and exploitation

* **Practical Constraints**

  * Prevents duplicate accounts
  * Prevents double-booking of users
  * Handles role switching and profile updates
  * Limits overexposure of popular users

---

## High-Level Architecture

```
Request (Team / Context)
        ↓
Candidate Filtering
  - Skill requirements
  - Availability checks
        ↓
Feature Computation
  - Skill match score
  - Semantic similarity score
  - Graph similarity score
        ↓
Learning-to-Rank Model
        ↓
Online Bandit Adjustment
        ↓
Final Ranked Recommendations
        ↓
Explanation Generation
```

---

## Recommendation Signals

### 1. Skill Matching

* Users and teams can have multiple skills
* Skills have levels (Beginner / Intermediate / Pro)
* Skills can be prioritized by teams
* Produces a normalized skill compatibility score

### 2. Semantic Matching (NLP)

* Uses sentence embeddings to encode:

  * user bios
  * project descriptions
  * “looking for” text
* Matches based on meaning rather than keywords
* Produces a semantic similarity score

### 3. Graph-Based Similarity

* Users are modeled as nodes in a collaboration graph
* Edges represent shared events, teams, or domains
* Node2Vec embeddings capture collaboration potential
* Produces a graph similarity score

### 4. Learning-to-Rank

* Combines all signals into a single feature vector
* Trained using historical accept / reject data
* Outputs a final relevance score for ranking

### 5. Online Learning (Bandits)

* Adjusts rankings based on recent feedback
* Uses time decay so older interactions matter less
* Enables controlled exploration of new candidates

---

## Explainability

Each recommendation includes an explanation describing why the user was suggested, for example:

* strong skill alignment with team needs
* similar interests based on profile text
* high collaboration potential from network history
* recent positive feedback boost

This improves transparency, trust, and debuggability.

---

## Project Structure

```
teamforge/
├── api/                    # API layer for backend integration
├── graph/                  # Graph construction and embeddings
├── recommender/
│   ├── pipeline.py         # Main recommendation pipeline
│   ├── skill_match.py      # Skill compatibility logic
│   ├── nlp_score.py        # Semantic similarity
│   ├── graph_embedding.py  # Graph similarity
│   ├── bandit.py           # Online learning logic
│   ├── feedback.py         # Feedback ingestion
│   └── explain.py          # Explanation generation
├── ensemble_conf.json      # Model configuration
├── main.py                 # Entry point
├── teamforge_edgecases.py  # Constraint handling
└── README.md
```

---

## How to Use (Backend Integration)

The recommendation engine is designed to be used as a service.

### Core Call

```python
recommend(team_id: int, top_k: int) -> List[Dict]
```

### Output Format

Each recommended user includes:

```json
{
  "user_id": 123,
  "score": 0.98,
  "skill_score": 0.25,
  "semantic_score": 0.21,
  "graph_sim": 0.96,
  "explanation": "Why this user was recommended"
}
```

The frontend can safely display the explanation directly.

---

## Feedback Loop

The system improves over time using feedback signals such as:

* swipe right / swipe left
* accept / reject
* successful team formation

Feedback is time-decayed so recent interactions have greater influence.

---

## Design Principles

* Modular and extensible
* Backend-agnostic
* Explainable by default
* Realistic constraints modeled explicitly
* Suitable for production iteration

---

## Future Improvements

* Event-specific or domain-specific bandits
* Diversity-aware ranking constraints
* FAISS-based candidate retrieval for large-scale usage
* Redis caching for low-latency serving
* Offline retraining pipelines

---


