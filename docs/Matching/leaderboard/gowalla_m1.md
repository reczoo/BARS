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

# Gowalla_m1

```{note}
Please use the following evaluation settings for this benchmark:
+ Dataset split: [Gowalla_m1](https://github.com/reczoo/Datasets/tree/main/Gowalla/Gowalla_m1)
+ Embedding size: 64
```

🔥 **See the benchmarking results**:

```{code-cell}
from plots import show_table, show_plot
show_plot("gowalla_m1.csv")
show_table("gowalla_m1.csv")
```
