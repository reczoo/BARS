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

# Criteo_x4

Note that we have set two evaluation protocols `Criteo_x4_001` and `Criteo_x4_002` for this benchmark, which vary in the settings of rare feature filtering and embedding dimensions.

`````{tab-set}
````{tab-item} Criteo_x4_001

```{admonition} Criteo_x4_001
:class: note
Please use the following evaluation settings for this benchmark:
+ Dataset split: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4)
+ Rare features filtering: min_categr_count=10
+ Embedding size: 16
```

````
````{tab-item} Criteo_x4_002

```{admonition} Criteo_x4_002
:class: note
Please use the following evaluation settings for this benchmark:
+ Dataset split: [Criteo_x4](https://github.com/reczoo/Datasets/tree/main/Criteo/Criteo_x4)
+ Rare features filtering: min_categr_count=2
+ Embedding size: 40
```
````
`````

🔥 **See the benchmarking results on Criteo_x4_001**:

```{code-cell}
from plots import show_table, show_plot
show_plot("criteo_x4_001.csv")
show_table("criteo_x4_001.csv")
```

🔥 **See the benchmarking results on Criteo_x4_002**:

```{code-cell}
from plots import show_table, show_plot
show_plot("criteo_x4_002.csv")
show_table("criteo_x4_002.csv")
```
