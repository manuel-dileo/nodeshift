# Node Embedding Shift in Online Social Networks

Use of Temporal Graph Neural Networks to analyze how users behave before, during, and after a shocking event in an online social network (OSN) through node embedding shift.

The case study is a user migration from Steemit, the most well-known Blockchain-based OSN, to Hive, due to an hard-fork event.

This repository contains information, data and code behind the work: User migration in blockchain-based online social networks through the lens of temporal node representation shift, accepted at Machine Learning

```bibtex
@Article{Dileo2025shift,
author={Dileo, Manuel
and Zignani, Matteo},
title={User migration in blockchain-based online social networks through the lens of temporal node representation shift},
journal={Machine Learning},
year={2025},
month={Nov},
day={10},
volume={114},
number={12},
pages={272},
abstract={User migration in online social networks represents a critical phenomenon that can reshape platform dynamics and lead to abrupt structural changes, often triggered by technical, social, or competitive factors. In the context of blockchain-based online social networks, migration events can be particularly disruptive when triggered by hard forks - fundamental splits in the underlying blockchain protocol that create two incompatible versions of the platform. However, understanding how users adapt their behavior before, during, and after such events remains a challenging research question. To address this challenge, we rely on the framework of graph representation learning, with a particular focus on Temporal Graph Neural Networks (TGNNs). In particular, we analyze how node representations returned by TGNNs evolve during the migration event and examine how representation shifts can mirror changes in users‚{\"A}{\^o} behavioral patterns and platform interactions. Our study focuses on Steemit, a blockchain-based social network that experienced a significant user migration following a hard fork in its supporting blockchain infrastructure. Our findings highlight that both the prediction performance and node representation are influenced by the occurrence of the migration event. We detect shifts in node representations that correspond to changes in individual user behavior throughout the event. Furthermore, group-centric analysis reveals changes in behavior and memberships among similar users during different transition periods. Additionally, we find a level of polarization in node representations caused by the migration event, which gradually diminishes over time, resulting in more evenly distributed dimensions of node representations months after the first migration. We compare our approach against two baselines based on network statistics and pre-trained LLM embeddings, showing that TGNNs better capture the distribution shift derived by the migration. To summarize, this work offers valuable insights into user behavior dynamics during platform migrations, demonstrating the effectiveness of temporal graph learning approaches in analyzing such transitions in an automated manner.},
issn={1573-0565},
doi={10.1007/s10994-025-06905-y},
url={https://doi.org/10.1007/s10994-025-06905-y}
}
```


# Dataset
Due to privacy reasons on personal data like username and textual content, we can't release the dataset related to Steemit. To patch this problem, we provide an anonymized version of our data. This version represents the final mathematical objects that are use to feed the models. For data gathering you can refer to the [Steemit API documentation](https://developers.steem.io/). 

Data related to the period affected by the shocking event are available in this repo in the zip `steemit-hardfork-data.zip`. To use them in the notebook of experiments, you need to unzip the file in a folder with the same name on the same level. For data related to the "stable" period in 2016 you can refer to [Temporal Graph Learning for dynamic link prediction with text in Online Social Networks](https://link.springer.com/article/10.1007/s10994-023-06475-x).

# Experiments

## Command Line

To reproduce the experiments of the paper, you can use the following script:
```
python nodeshift.py --data steemit-hardfork-data --feature <struct|text> [<--sbert_only|--graph_only>]
```
The results will be available in the "results" folder. `steemit-hardfork-data` must be the folder that contains the unzipped files of `steemit-hardfork-data.zip` 
The options `--sbert_only` and `--graph_only` allow running the experiments related to baselines.

To reproduce the ablation study, you can use the following script:
```
python run_ablation.py --data steemith-hardfork-data
```

## Interactive Notebook
The notebook `TGNN-SteemitHardFork.ipynb` contains all the materials to reproduce the experiments on the period affected by the shocking event.

## Model architecture
The figure below shows the running architecture of the TGNN model.
![GNN Architecture](GNNArchitecture.drawio.png "Dynamic GNN based on ROLAND framework"). 

We report the architecture configuration in the following table: 

| Layer                     | input_channels | output_channels |
|---------------------------|----------------|-----------------|
| Preprocessing layer (MLP) | 384            | 256             |
| Preprocessing layer (MLP) | 256            | 128             |
| Graph Convolution (GCN)   | 128            | 64              |
| Embedding Update (GRU)    | 64             | 64              |
| Graph Convolution (GCN)   | 64             | 32              |
| Embedding Update (GRU)    | 32             | 32              |
| Decoder (HadamardMLP)     | 32             | 2               | 

We report the configuration of hyperparameters for the future link prediction task in the following table: 

| Hyperparameter | Value |
|----------------|-------|
| Optimizer      | Adam  |
| Learning rate  | 0.01  |
| Weight Decay   | 5e-3  |
| Epochs         | 50    |



