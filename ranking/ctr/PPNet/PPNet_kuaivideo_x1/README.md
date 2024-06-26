## PPNet_kuaishou_x1

A hands-on guide to run the PPNet model on the KuaiVideo_x1 dataset.

Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)


| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |
|:-----------------------------:|:-----------:|:--------:|:--------:|-------|
### Environments
+ Hardware

  ```python
  CPU: Intel(R) Xeon(R) Gold 6278C CPU @ 2.60GHz
  GPU: Tesla V100 32G
  RAM: 755G

  ```

+ Software

  ```python
  cuda: 10.2
  python: 3.7.10
  pytorch: 1.11.0
  pandas: 1.1.5
  numpy: 1.19.5
  scipy: 1.5.2
  sklearn: 0.22.1
  pyyaml: 6.0
  h5py: 2.8.0
  tqdm: 4.64.0
  fuxictr: 2.0.3

  ```

### Dataset
Please refer to [KuaiVideo_x1](https://github.com/reczoo/Datasets/tree/main/KuaiShou/KuaiVideo_x1) to get the dataset details.

### Code

We use the [PPNet](https://github.com/reczoo/FuxiCTR/tree/v2.0.3/model_zoo/PPNet) model code from [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/tree/v2.0.3) for this experiment.

Running steps:

1. Download [FuxiCTR-v2.0.3](https://github.com/reczoo/FuxiCTR/archive/refs/tags/v2.0.3.zip) and install all the dependencies listed in the [environments](#environments).
    
    ```bash
    pip uninstall fuxictr
    pip install fuxictr==2.0.3
    ```

2. Create a data directory and put the downloaded data files in `../data/KuaiShou/KuaiVideo_x1`.

3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [PPNet_kuaivideo_x1_tuner_config_01](./PPNet_kuaivideo_x1_tuner_config_01). Please make sure that the data paths in `dataset_config.yaml` are correctly set.

4. Run the following script to start training and evaluation.

    ```bash
    cd FuxiCTR/model_zoo/PPNet
    nohup python run_expid.py --config YOUR_PATH/PPNet/PPNet_kuaivideo_x1_tuner_config_01 --expid PPNet_kuaivideo_x1_018_71ca4227 --gpu 0 > run.log &
    tail -f run.log
    ```

### Results

| gAUC | AUC | logloss  |
|:--------------------:|:--------------------:|:--------------------:|
| 0.666808 | 0.746437 | 0.437027  |


### Logs
```python
2023-06-02 07:21:34,050 P20089 INFO Params: {
    "batch_norm": "False",
    "batch_size": "8192",
    "data_format": "csv",
    "data_root": "../data/KuaiShou/",
    "dataset_id": "kuaivideo_x1_dc7a3035",
    "debug_mode": "False",
    "early_stop_patience": "2",
    "embedding_dim": "64",
    "embedding_regularizer": "5e-05",
    "epochs": "100",
    "eval_steps": "None",
    "feature_cols": "[{'active': True, 'dtype': 'int', 'name': 'group_id', 'preprocess': 'copy_from(user_id)', 'remap': False, 'type': 'meta'}, {'active': True, 'dtype': 'str', 'name': 'user_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'name': 'item_id', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'min_categr_count': 1, 'name': 'item_emb', 'preprocess': 'copy_from(item_id)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'type': 'categorical'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'pos_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'max_len': 100, 'name': 'neg_items', 'padding': 'pre', 'share_embedding': 'item_id', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'pos_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(pos_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}, {'active': True, 'dtype': 'str', 'embedding_dim': 64, 'max_len': 100, 'min_categr_count': 1, 'name': 'neg_items_emb', 'padding': 'pre', 'preprocess': 'copy_from(neg_items)', 'pretrained_emb': '../data/KuaiShou/KuaiVideo_x1/item_visual_emb_dim64.h5', 'share_embedding': 'item_emb', 'splitter': '^', 'type': 'sequence'}]",
    "feature_config": "None",
    "feature_specs": "[{'feature_encoder': 'nn.Linear(64, 64, bias=False)', 'name': 'item_emb'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'pos_items'}, {'feature_encoder': 'layers.MaskedAveragePooling()', 'name': 'neg_items'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'pos_items_emb'}, {'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'name': 'neg_items_emb'}]",
    "gate_emb_dim": "32",
    "gate_hidden_dim": "512",
    "gate_priors": "['item_id']",
    "gpu": "6",
    "group_id": "group_id",
    "hidden_activations": "ReLU",
    "hidden_units": "[1024, 512, 256]",
    "label_col": "{'dtype': 'float', 'name': 'is_click'}",
    "learning_rate": "0.001",
    "loss": "binary_crossentropy",
    "metrics": "['gAUC', 'AUC', 'logloss']",
    "min_categr_count": "10",
    "model": "PPNet",
    "model_id": "PPNet_kuaivideo_x1_018_71ca4227",
    "model_root": "./checkpoints/PPNet_kuaivideo_x1/",
    "monitor": "{'AUC': 1, 'gAUC': 1}",
    "monitor_mode": "max",
    "net_dropout": "0.1",
    "net_regularizer": "0",
    "num_workers": "3",
    "optimizer": "adam",
    "pickle_feature_encoder": "True",
    "save_best_only": "True",
    "seed": "20222023",
    "shuffle": "True",
    "task": "binary_classification",
    "test_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "train_data": "../data/KuaiShou/KuaiVideo_x1/train.csv",
    "use_features": "None",
    "valid_data": "../data/KuaiShou/KuaiVideo_x1/test.csv",
    "verbose": "1"
}
2023-06-02 07:21:34,093 P20089 INFO Set up feature processor...
2023-06-02 07:21:34,093 P20089 WARNING Skip rebuilding ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json. Please delete it manually if rebuilding is required.
2023-06-02 07:21:34,093 P20089 INFO Load feature_map from json: ../data/KuaiShou/kuaivideo_x1_dc7a3035/feature_map.json
2023-06-02 07:21:34,094 P20089 INFO Set column index...
2023-06-02 07:21:34,094 P20089 INFO Feature specs: {
    "group_id": "{'type': 'meta'}",
    "item_emb": "{'source': '', 'type': 'categorical', 'embedding_dim': 64, 'pretrained_emb': 'pretrained_emb.h5', 'freeze_emb': True, 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'feature_encoder': 'nn.Linear(64, 64, bias=False)'}",
    "item_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406}",
    "neg_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "neg_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "pos_items": "{'source': '', 'type': 'sequence', 'feature_encoder': 'layers.MaskedAveragePooling()', 'share_embedding': 'item_id', 'padding_idx': 0, 'oov_idx': 632405, 'vocab_size': 632406, 'max_len': 100}",
    "pos_items_emb": "{'source': '', 'type': 'sequence', 'feature_encoder': ['layers.MaskedAveragePooling()', 'nn.Linear(64, 64, bias=False)'], 'embedding_dim': 64, 'share_embedding': 'item_emb', 'padding_idx': 0, 'oov_idx': 3242316, 'vocab_size': 3242317, 'max_len': 100}",
    "user_id": "{'source': '', 'type': 'categorical', 'padding_idx': 0, 'oov_idx': 10001, 'vocab_size': 10002}"
}
2023-06-02 07:21:41,703 P20089 INFO Total number of parameters: 64137665.
2023-06-02 07:21:41,703 P20089 INFO Loading data...
2023-06-02 07:21:41,703 P20089 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/train.h5
2023-06-02 07:22:07,760 P20089 INFO Train samples: total/10931092, blocks/1
2023-06-02 07:22:07,760 P20089 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/valid.h5
2023-06-02 07:22:13,961 P20089 INFO Validation samples: total/2730291, blocks/1
2023-06-02 07:22:13,961 P20089 INFO Loading train and validation data done.
2023-06-02 07:22:13,961 P20089 INFO Start training: 1335 batches/epoch
2023-06-02 07:22:13,961 P20089 INFO ************ Epoch=1 start ************
2023-06-02 07:27:14,324 P20089 INFO Train loss: 0.452990
2023-06-02 07:27:14,324 P20089 INFO Evaluation @epoch 1 - batch 1335: 
2023-06-02 07:28:19,548 P20089 INFO [Metrics] AUC: 0.727919 - gAUC: 0.639835
2023-06-02 07:28:19,550 P20089 INFO Save best model: monitor(max)=1.367754
2023-06-02 07:28:22,061 P20089 INFO ************ Epoch=1 end ************
2023-06-02 07:32:05,009 P20089 INFO Train loss: 0.442990
2023-06-02 07:32:05,014 P20089 INFO Evaluation @epoch 2 - batch 1335: 
2023-06-02 07:33:01,292 P20089 INFO [Metrics] AUC: 0.733623 - gAUC: 0.649477
2023-06-02 07:33:01,296 P20089 INFO Save best model: monitor(max)=1.383100
2023-06-02 07:33:03,715 P20089 INFO ************ Epoch=2 end ************
2023-06-02 07:35:44,584 P20089 INFO Train loss: 0.438196
2023-06-02 07:35:44,585 P20089 INFO Evaluation @epoch 3 - batch 1335: 
2023-06-02 07:36:20,917 P20089 INFO [Metrics] AUC: 0.736829 - gAUC: 0.654141
2023-06-02 07:36:20,918 P20089 INFO Save best model: monitor(max)=1.390969
2023-06-02 07:36:23,418 P20089 INFO ************ Epoch=3 end ************
2023-06-02 07:38:27,515 P20089 INFO Train loss: 0.435034
2023-06-02 07:38:27,515 P20089 INFO Evaluation @epoch 4 - batch 1335: 
2023-06-02 07:39:04,830 P20089 INFO [Metrics] AUC: 0.740345 - gAUC: 0.658251
2023-06-02 07:39:04,832 P20089 INFO Save best model: monitor(max)=1.398595
2023-06-02 07:39:07,654 P20089 INFO ************ Epoch=4 end ************
2023-06-02 07:41:11,354 P20089 INFO Train loss: 0.432669
2023-06-02 07:41:11,355 P20089 INFO Evaluation @epoch 5 - batch 1335: 
2023-06-02 07:41:48,670 P20089 INFO [Metrics] AUC: 0.742553 - gAUC: 0.660145
2023-06-02 07:41:48,672 P20089 INFO Save best model: monitor(max)=1.402698
2023-06-02 07:41:51,120 P20089 INFO ************ Epoch=5 end ************
2023-06-02 07:43:52,727 P20089 INFO Train loss: 0.430849
2023-06-02 07:43:52,727 P20089 INFO Evaluation @epoch 6 - batch 1335: 
2023-06-02 07:44:28,824 P20089 INFO [Metrics] AUC: 0.741969 - gAUC: 0.659522
2023-06-02 07:44:28,825 P20089 INFO Monitor(max)=1.401491 STOP!
2023-06-02 07:44:28,825 P20089 INFO Reduce learning rate on plateau: 0.000100
2023-06-02 07:44:28,904 P20089 INFO ************ Epoch=6 end ************
2023-06-02 07:46:36,372 P20089 INFO Train loss: 0.413364
2023-06-02 07:46:36,373 P20089 INFO Evaluation @epoch 7 - batch 1335: 
2023-06-02 07:47:13,945 P20089 INFO [Metrics] AUC: 0.746466 - gAUC: 0.665974
2023-06-02 07:47:13,947 P20089 INFO Save best model: monitor(max)=1.412440
2023-06-02 07:47:16,401 P20089 INFO ************ Epoch=7 end ************
2023-06-02 07:49:20,431 P20089 INFO Train loss: 0.407560
2023-06-02 07:49:20,432 P20089 INFO Evaluation @epoch 8 - batch 1335: 
2023-06-02 07:49:47,157 P20089 INFO [Metrics] AUC: 0.746437 - gAUC: 0.666808
2023-06-02 07:49:47,158 P20089 INFO Save best model: monitor(max)=1.413245
2023-06-02 07:49:49,759 P20089 INFO ************ Epoch=8 end ************
2023-06-02 07:51:30,264 P20089 INFO Train loss: 0.403702
2023-06-02 07:51:30,264 P20089 INFO Evaluation @epoch 9 - batch 1335: 
2023-06-02 07:51:56,818 P20089 INFO [Metrics] AUC: 0.745861 - gAUC: 0.666850
2023-06-02 07:51:56,820 P20089 INFO Monitor(max)=1.412711 STOP!
2023-06-02 07:51:56,820 P20089 INFO Reduce learning rate on plateau: 0.000010
2023-06-02 07:51:56,906 P20089 INFO ************ Epoch=9 end ************
2023-06-02 07:53:39,200 P20089 INFO Train loss: 0.395312
2023-06-02 07:53:39,200 P20089 INFO Evaluation @epoch 10 - batch 1335: 
2023-06-02 07:54:00,648 P20089 INFO [Metrics] AUC: 0.744890 - gAUC: 0.666531
2023-06-02 07:54:00,649 P20089 INFO Monitor(max)=1.411420 STOP!
2023-06-02 07:54:00,650 P20089 INFO Reduce learning rate on plateau: 0.000001
2023-06-02 07:54:00,650 P20089 INFO ********* Epoch==10 early stop *********
2023-06-02 07:54:00,736 P20089 INFO Training finished.
2023-06-02 07:54:00,737 P20089 INFO Load best model: /cache/FuxiCTR/benchmark/checkpoints/PPNet_kuaivideo_x1/kuaivideo_x1_dc7a3035/PPNet_kuaivideo_x1_018_71ca4227.model
2023-06-02 07:54:01,954 P20089 INFO ****** Validation evaluation ******
2023-06-02 07:54:22,535 P20089 INFO [Metrics] gAUC: 0.666808 - AUC: 0.746437 - logloss: 0.437027
2023-06-02 07:54:22,697 P20089 INFO ******** Test evaluation ********
2023-06-02 07:54:22,697 P20089 INFO Loading data...
2023-06-02 07:54:22,697 P20089 INFO Loading data from h5: ../data/KuaiShou/kuaivideo_x1_dc7a3035/test.h5
2023-06-02 07:54:28,240 P20089 INFO Test samples: total/2730291, blocks/1
2023-06-02 07:54:28,240 P20089 INFO Loading test data done.
2023-06-02 07:54:48,592 P20089 INFO [Metrics] gAUC: 0.666808 - AUC: 0.746437 - logloss: 0.437027

```
