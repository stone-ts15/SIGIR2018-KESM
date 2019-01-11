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
- Add *tanh* to KESM network's output to force the output to fall in [-1, 1].
- Add *dropout* in two *fc* layers to avoid overfitting

## Training

**GPU memory**: less than 1G

**model size**: 325 KB

**training speed**: 1 epoch in 20 min

## Performance (on test set)

| epochs | Precision@1 | Precision@5 | Recall@1 | Recall@5 |
| ------ | ----------- | ----------- | -------- | -------- |
| 1      | 0.1736      | 0.1517      | 0.0537   | 0.2302   |
| 2      | 0.1719      | 0.1558      | 0.0509   | 0.2338   |
| 3      | 0.1658      | 0.1508      | 0.0486   | 0.2262   |
| 4      | 0.1449      | 0.1333      | 0.0443   | 0.2006   |
| 5      | 0.1471      | 0.1381      | 0.0464   | 0.2102   |
| 10     | 0.1453      | 0.1321      | 0.0454   | 0.1981   |
| 15     | 0.1442      | 0.1290      | 0.0453   | 0.1947   |

Adding dropout:

| epochs | Precision@1 | Precision@5 | Recall@1 | Recall@5 |
| ------ | ----------- | ----------- | -------- | -------- |
| 1      | 0.1637      | 0.1391      | 0.0496   | 0.2068   |
| 2      | 0.1665      | 0.1439      | 0.0529   | 0.2209   |
| 3      | 0.1488      | 0.1317      | 0.0464   | 0.2065   |
| 4      | 0.1456      | 0.1196      | 0.0428   | 0.1888   |
| 5      | 0.1269      | 0.1103      | 0.0371   | 0.1719   |
| 10     | 0.1251      | 0.1055      | 0.0380   | 0.1633   |
| 15     | 0.1084      | 0.0956      | 0.0328   | 0.1483   |

Random:

| epochs | Precision@1 | Precision@5 | Recall@1 | Recall@5 |
| ------ | ----------- | ----------- | -------- | -------- |
| -      | 0.1116      | 0.1105      | 0.0365   | 0.1771   |

