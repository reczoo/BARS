---
kernelspec:
  name: python
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
---

# KuaiVideo_x1

```{note}
Please use the following evaluation settings for this benchmark:
+ Dataset split: [KuaiVideo_x1](https://github.com/reczoo/Datasets/tree/main/KuaiShou/KuaiVideo_x1)
+ Rare features filtering: min_categr_count=10
+ Embedding size: 64
```

🔥 **See the benchmarking results**:

```{code-cell}
from plots import show_table_gauc, show_plot_gauc
show_plot_gauc("kuaivideo_x1.csv")
show_table_gauc("kuaivideo_x1.csv")
```
