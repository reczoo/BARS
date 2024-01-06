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

# KKBox_x1

```{note}
Please use the following evaluation settings for this benchmark:
+ Dataset split: [KKBox_x1](https://github.com/reczoo/Datasets/tree/main/KKBox/KKBox_x1)
+ Rare features filtering: min_categr_count=10
+ Embedding size: 128
```

🔥 **See the benchmarking results**:

```{code-cell}
from plots import show_table, show_plot
show_plot("kkbox_x1.csv")
show_table("kkbox_x1.csv")
```
