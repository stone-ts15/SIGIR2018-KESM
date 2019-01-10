Implementation of *Towards Better Text Understanding and Retrieval through Kernel Entity Salience Modeling* , SIGIR 2018.

## Dataset

17428 articles from dataset [Semantic Scholar](https://www.semanticscholar.org/) is used to train this network.

## Architecture

- `data/collect.py` extracts fields "title" and "paperAbstract" from articles, and finishes entity linking work using [TagMe](http://tagme.org/) API
- `spider/` crawls entity descriptions from [MediaWiki](https://www.wikidata.org/w/api.php) API
- `data_gen.py` yields training data using [Word2Vec](https://code.google.com/archive/p/word2vec/) API
- `model.py` implement KESM model using PyTorch
- `trained_models/` pre-trained models

## Modification

- Add *BN* layer to *CNN* part in KEE, since entities' distribution from different articles may vary.
- Add *tanh* to KESM network's output to force the output to fall in $[-1, 1]$.

## Training

**GPU memory**: less than 1G

**model size**: 325 KB

**training speed**: 1 epoch in 20 min

## Performance (on test set)

| epochs | Precision@1 | Precision@5 | Recall@1 | Recall@5 |
| ------ | ----------- | ----------- | -------- | -------- |
| 1      | 0.0000      | 0.0000      | 0.0000   | 0.0000   |
| 5      | 0.0000      | 0.0000      | 0.0000   | 0.0000   |