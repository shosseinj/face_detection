# Face Detection and Embedding Experiments

This repository contains experiments that separate face detection from face-embedding storage/search. The project uses a detector pipeline together with a Qdrant vector database for embedding-oriented retrieval experiments.

## Structure

- `Detection/` — detection and query-side processing
- `Embedding/` — embedding generation / storage workflow
- `qdrant_storage/` — local Qdrant storage used during development

## Running Qdrant

```bash
cd Embedding
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

## Detection Example

```bash
cd Detection
python main.py --collection n3 --threshold 0.01 --webCam False
```

## Purpose

The project was used to explore the practical pipeline around face detection, embedding generation, vector indexing, and similarity search. It is an experimental repository rather than a complete access-control product.

## Upstream Attribution

The detection code and much of its Git history originate from Yakhyokhuja Valikhujaev's [RetinaFace PyTorch](https://github.com/yakhyo/retinaface-pytorch) project. This repository combines that upstream detector work with local embedding and Qdrant experiments; it is not presented as an original RetinaFace implementation.

## Data Note

Vector databases and face embeddings may contain biometric information. Public versions of this repository should contain only synthetic or explicitly shareable example data; deployment databases should not be committed to source control.


## Goal

The local work connects an upstream RetinaFace detector to an embedding and vector-search experiment, with Qdrant used to store and retrieve face representations.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
docker run --name face-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Use a separate Qdrant volume for private experiments rather than the tracked `qdrant_storage/` directory.

## Working with the Repository

Detection code is under `Detection/`; embedding and indexing code is under `Embedding/`. Start Qdrant first, configure collection and model paths, then run the documented `Detection/main.py` command. Only synthetic or consented images and embeddings should be used while testing.
