# BARS-Match Overview

BARS-Match: An Open Benchmark for Candidate Item Matching https://openbenchmark.github.io/BARS/Matching

Recommender systems generally comprise two main stages, matching and ranking. As the first-stage task, candidate item matching is designed to efficiently retrieve hundreds of item candidates out of the entire item corpus. Representative methods of candidate item matching include collaborative filtering, two-tower models, autoencoder-based models, sequential models, graph-based models, etc. To drive research in this direction, the BARS project aims to build an open benchmark for candidate item matching, which consists of:

+ [A curated list of papers on candidate item matching](./papers.md), which have been tagged into different categories, such as CF, autoencoders, two-tower models, GNNs, and so on.

+ [A collection of open datasets](https://github.com/reczoo/Datasets?tab=readme-ov-file#matching) for research on candidate item matching, and unique dataset IDs to track specific data splits for each dataset.

+ [An open-source library for candidate item matching](https://github.com/reczoo/RecBox) with key features in configurability, tunability, and reproduciblity.

+ Most importantly, [the most comprehensive benchmarking results](./leaderboard/index.md) on various models and datasets. For each result, the detailed reproducing step is recorded along with the open-source benchmarking scripts.
