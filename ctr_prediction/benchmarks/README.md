# Benchmarks

To allow extensibility of the BARS benchmark, the code implementation is not binded to any single framework or library. But we strongly recommend the following libraries due to their abundant model implementations.

+ FuxiCTR (pytorch): https://github.com/xue-pai/FuxiCTR
+ DeepCTR (tensorflow): https://github.com/shenweichen/DeepCTR


## Model List

The following models have been benchmarked with open-source code and detailed reproducing steps.

| Publication| Model  | Paper | Benchmark | 
| :-----: | :-------: |:------------|:----------:|
| WWW'07| LR  |[Predicting Clicks: Estimating the Click-Through Rate for New Ads](https://dl.acm.org/citation.cfm?id=1242643) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/LR) |
|ICDM'10 | FM  | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)| [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/FM) |
|CIKM'15| CCPM | [A Convolutional Click Prediction Model](http://www.escience.cn/system/download/73676) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/CCPM) |
| RecSys'16 | FFM | [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/FFM) |
| RecSys'16 | YoutubeDNN | [Deep Neural Networks for YouTube Recommendations](http://art.yale.edu/file_columns/0001/1132/covington.pdf) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/YoutubeDNN) |
| DLRS'16 | Wide&Deep  | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) |[:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/WideDeep) |
| ICDM'16 | IPNN | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/IPNN) |
| KDD'16 | DeepCrossing | [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)  | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepCrossing) |
| NIPS'16 | HOFM | [Higher-Order Factorization Machines](https://papers.nips.cc/paper/6144-higher-order-factorization-machines.pdf) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/HOFM) |
| IJCAI'17 | DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepFM) |
|SIGIR'17 | NFM | [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/NFM) |
|IJCAI'17 | AFM | [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/0435.pdf) |[:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/AFM)|
| ADKDD'17 | DCN  | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DCN)|
| WWW'18 | FwFM | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)  | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/FwFM) |
|KDD'18 | xDeepFM | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/xDeepFM) |
|KDD'18 | DIN | [Deep Interest Network for Click-Through Rate Prediction](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DIN) |
|CIKM'19 | FiGNN | [FiGNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction](https://arxiv.org/abs/1910.05552) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/FiGNN) |
|CIKM'19 | AutoInt/AutoInt+ | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/AutoInt) |
|RecSys'19 | FiBiNET | [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/FiBiNET) |
|WWW'19 | FGCNN | [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/abs/1904.04447) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/FGCNN) |
| AAAI'19| HFM/HFM+ | [Holographic Factorization Machines for Recommendation](https://ojs.aaai.org//index.php/AAAI/article/view/4448)  | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/HFM) |
| NeuralNetworks'20 | ONN | [Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579)  | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/ONN) |
| AAAI'20 | AFN/AFN+ | [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/AFN) |
| AAAI'20  | LorentzFM | [Learning Feature Interactions with Lorentzian Factorization](https://arxiv.org/abs/1911.09821) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/LorentzFM) |
| WSDM'20 | InterHAt | [Interpretable Click-through Rate Prediction through Hierarchical Attention](https://dl.acm.org/doi/10.1145/3336191.3371785) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/InterHAt) |
| DLP-KDD'20 | FLEN | [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/abs/1911.04690) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/FLEN) |
| CIKM'20 | DeepIM | [Deep Interaction Machine: A Simple but Effective Model for High-order Feature Interactions](https://dl.acm.org/doi/abs/10.1145/3340531.3412077) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DeepIM) |
| WWW'21 | FmFM | [FM^2: Field-matrixed Factorization Machines for Recommender Systems](https://arxiv.org/abs/2102.12994) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/FmFM) |
| WWW'21 | DCN-V2 | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DCNv2) |
| CIKM'21 | DESTINE | [Disentangled Self-Attentive Neural Networks for Click-Through Rate Prediction](https://arxiv.org/abs/2101.03654) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/DESTINE) |
| DLP-KDD'21 | MaskNet | [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619) | [:arrow_upper_right:](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/benchmarks/MaskNet) |