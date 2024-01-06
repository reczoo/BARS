# LR-GCF_yelp18_x0

A notebook to benchmark LR_GCCF on yelp18_x0 dataset.

Author: Yi Li, Tsinghua University

Edited by [XUEPAI Team](https://github.com/xue-pai)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) 

### Environments

* Hardware

```python
CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
RAM: 125G
GPU: GeForce RTX 2080Ti, 11G memory
```

* Software

```python
python: 3.6.2
pytorch: 1.1.0
```

### Dataset

We directly use the data format transformation file `data2npy.py`（Here, we rename this file to `LR-GCCF_data_process.py`.） provided by LR-GCCF to convert the data from the LightGCN repo into the format required by the program.

You need to put the downloaded files `train.txt` and `test.txt` into the data/Yelp18/yelp18_x0 directory. 

### Code

1. The benchmark is implemented based on the original LR-GCCF code released by the authors at: https://github.com/newlei/LR-GCCF/. We use the version with commit hash: 17c160a.

2. We added the calculation of the recall metric to the hr_ndcg function of the `evaluate.py` file. At the same time, based on the calculation of the Gowalla and Yelp datasets by LR_GCCF, we have realized the calculation of the Yelp18 dataset.

3. Download the dataset from [LightGCN repo](https://github.com/kuandeng/LightGCN/tree/master/Data/yelp2018) and run the preprocessing script for format transformation.

   ```python
   cd data/Yelp18/yelp18_x0
   python LR-GCCF_data_process.py
   cd benchmarks/LR-GCCF
   ```

4. Run the following script to reproduce the result.

   ```python
   python LR_GCCF.py --gpu_id 2 --dataset yelp --run_id s0 --embed_size 64 --epoch 350 --lr 0.005
   ```

### Results

```python
HR@20 = 0.0561, Recall@20 = 0.0558, NDCG@20 = 0.0343
```

### Logs

#### train

```shell
epoch:0 time:28.6        train loss:0.6895=0.6891+ val loss:0.6549 test loss:0.655
epoch:1 time:29.1        train loss:0.5351=0.5196+ val loss:0.3474 test loss:0.3487
epoch:2 time:27.0        train loss:0.3079=0.229+ val loss:0.2211 test loss:0.2227
epoch:3 time:27.9        train loss:0.2973=0.1693+ val loss:0.2113 test loss:0.2126
epoch:4 time:28.3        train loss:0.284=0.1683+ val loss:0.2193 test loss:0.2196
epoch:5 time:29.3        train loss:0.2729=0.1785+ val loss:0.2258 test loss:0.2269
epoch:6 time:27.2        train loss:0.2679=0.1755+ val loss:0.2158 test loss:0.217
epoch:7 time:28.2        train loss:0.263=0.1642+ val loss:0.208 test loss:0.2085
epoch:8 time:26.7        train loss:0.2582=0.1568+ val loss:0.2039 test loss:0.205
epoch:9 time:28.0        train loss:0.2544=0.1528+ val loss:0.2008 test loss:0.2023
epoch:10 time:30.1       train loss:0.2517=0.1483+ val loss:0.1976 test loss:0.1984
epoch:11 time:28.3       train loss:0.2489=0.1435+ val loss:0.1945 test loss:0.1956
epoch:12 time:27.9       train loss:0.2472=0.1403+ val loss:0.1933 test loss:0.1938
epoch:13 time:29.4       train loss:0.2458=0.1379+ val loss:0.1914 test loss:0.1925
epoch:14 time:29.0       train loss:0.2445=0.1356+ val loss:0.191 test loss:0.1916
epoch:15 time:26.5       train loss:0.243=0.1331+ val loss:0.1895 test loss:0.1897
epoch:16 time:27.2       train loss:0.2421=0.1313+ val loss:0.188 test loss:0.189
epoch:17 time:29.1       train loss:0.2408=0.1294+ val loss:0.187 test loss:0.1876
epoch:18 time:29.4       train loss:0.2398=0.1276+ val loss:0.1861 test loss:0.1869
epoch:19 time:28.9       train loss:0.2388=0.1257+ val loss:0.1842 test loss:0.1858
epoch:20 time:28.2       train loss:0.2376=0.1236+ val loss:0.1839 test loss:0.1835
epoch:21 time:27.9       train loss:0.2366=0.1218+ val loss:0.1826 test loss:0.184
epoch:22 time:28.4       train loss:0.2356=0.12+ val loss:0.1817 test loss:0.1828
epoch:23 time:28.3       train loss:0.2348=0.1183+ val loss:0.1817 test loss:0.1818
epoch:24 time:27.9       train loss:0.2341=0.1168+ val loss:0.1806 test loss:0.1812
epoch:25 time:29.0       train loss:0.2335=0.1156+ val loss:0.1806 test loss:0.1814
epoch:26 time:29.2       train loss:0.2332=0.1148+ val loss:0.1801 test loss:0.181
epoch:27 time:29.0       train loss:0.2326=0.1137+ val loss:0.1798 test loss:0.1807
epoch:28 time:27.9       train loss:0.2321=0.1126+ val loss:0.1795 test loss:0.1808
epoch:29 time:27.9       train loss:0.2317=0.1117+ val loss:0.1799 test loss:0.1804
epoch:30 time:27.6       train loss:0.2312=0.1107+ val loss:0.1791 test loss:0.1802
epoch:31 time:28.7       train loss:0.2309=0.11+ val loss:0.1794 test loss:0.1802
epoch:32 time:29.0       train loss:0.2308=0.1095+ val loss:0.1791 test loss:0.1798
epoch:33 time:29.6       train loss:0.2304=0.1088+ val loss:0.1791 test loss:0.1798
epoch:34 time:28.9       train loss:0.2301=0.1081+ val loss:0.1785 test loss:0.1794
epoch:35 time:28.1       train loss:0.2297=0.1073+ val loss:0.1779 test loss:0.1787
epoch:36 time:30.1       train loss:0.2296=0.1068+ val loss:0.1791 test loss:0.179
epoch:37 time:28.2       train loss:0.2294=0.1063+ val loss:0.1776 test loss:0.1791
epoch:38 time:28.7       train loss:0.2293=0.1058+ val loss:0.1782 test loss:0.1791
epoch:39 time:28.2       train loss:0.2291=0.1054+ val loss:0.1785 test loss:0.1786
epoch:40 time:29.1       train loss:0.2288=0.1047+ val loss:0.1776 test loss:0.1783
epoch:41 time:26.9       train loss:0.2285=0.1042+ val loss:0.1775 test loss:0.1783
epoch:42 time:28.6       train loss:0.2285=0.1038+ val loss:0.1775 test loss:0.1787
epoch:43 time:31.6       train loss:0.2284=0.1036+ val loss:0.1782 test loss:0.1784
epoch:44 time:28.1       train loss:0.2281=0.103+ val loss:0.1776 test loss:0.1783
epoch:45 time:27.0       train loss:0.2283=0.1028+ val loss:0.1777 test loss:0.1787
epoch:46 time:28.0       train loss:0.228=0.1025+ val loss:0.1777 test loss:0.1783
epoch:47 time:29.7       train loss:0.2278=0.102+ val loss:0.1781 test loss:0.1781
epoch:48 time:30.2       train loss:0.2277=0.1016+ val loss:0.178 test loss:0.1782
epoch:49 time:27.4       train loss:0.2275=0.1013+ val loss:0.177 test loss:0.178
epoch:50 time:26.9       train loss:0.2275=0.101+ val loss:0.1773 test loss:0.1789
epoch:51 time:27.5       train loss:0.2274=0.1007+ val loss:0.1775 test loss:0.178
epoch:52 time:29.2       train loss:0.2272=0.1004+ val loss:0.1773 test loss:0.1787
epoch:53 time:27.1       train loss:0.2272=0.1002+ val loss:0.1781 test loss:0.1781
epoch:54 time:27.0       train loss:0.2271=0.0998+ val loss:0.1769 test loss:0.1782
epoch:55 time:29.2       train loss:0.227=0.0996+ val loss:0.1774 test loss:0.178
epoch:56 time:27.5       train loss:0.2268=0.0993+ val loss:0.1771 test loss:0.1777
epoch:57 time:26.9       train loss:0.2268=0.099+ val loss:0.1776 test loss:0.1777
epoch:58 time:30.0       train loss:0.2268=0.0989+ val loss:0.1776 test loss:0.1778
epoch:59 time:27.9       train loss:0.2266=0.0986+ val loss:0.1781 test loss:0.1776
epoch:60 time:27.0       train loss:0.2266=0.0983+ val loss:0.1771 test loss:0.178
epoch:61 time:28.1       train loss:0.2266=0.0984+ val loss:0.1777 test loss:0.1784
epoch:62 time:30.6       train loss:0.2264=0.098+ val loss:0.1773 test loss:0.1775
epoch:63 time:28.5       train loss:0.2265=0.0979+ val loss:0.1775 test loss:0.1785
epoch:64 time:29.4       train loss:0.2265=0.0978+ val loss:0.1769 test loss:0.1779
epoch:65 time:27.3       train loss:0.2265=0.0977+ val loss:0.1775 test loss:0.1777
epoch:66 time:29.0       train loss:0.2263=0.0974+ val loss:0.1776 test loss:0.1783
epoch:67 time:27.0       train loss:0.2263=0.0972+ val loss:0.1772 test loss:0.178
epoch:68 time:28.7       train loss:0.2262=0.0969+ val loss:0.1772 test loss:0.178
epoch:69 time:30.4       train loss:0.2263=0.0969+ val loss:0.1776 test loss:0.1776
epoch:70 time:27.6       train loss:0.226=0.0967+ val loss:0.1769 test loss:0.1781
epoch:71 time:29.1       train loss:0.2261=0.0966+ val loss:0.1775 test loss:0.1783
epoch:72 time:27.7       train loss:0.2261=0.0966+ val loss:0.1776 test loss:0.1785
epoch:73 time:28.1       train loss:0.226=0.0964+ val loss:0.1767 test loss:0.1777
epoch:74 time:30.0       train loss:0.226=0.0961+ val loss:0.1778 test loss:0.1781
epoch:75 time:28.3       train loss:0.226=0.0961+ val loss:0.1769 test loss:0.1782
epoch:76 time:28.4       train loss:0.2258=0.0959+ val loss:0.1777 test loss:0.1778
epoch:77 time:28.0       train loss:0.2258=0.0957+ val loss:0.1774 test loss:0.1781
epoch:78 time:29.8       train loss:0.2258=0.0956+ val loss:0.1777 test loss:0.1779
epoch:79 time:28.6       train loss:0.2258=0.0956+ val loss:0.1773 test loss:0.1779
epoch:80 time:27.3       train loss:0.2258=0.0956+ val loss:0.1776 test loss:0.1779
epoch:81 time:27.2       train loss:0.2257=0.0953+ val loss:0.1778 test loss:0.1783
epoch:82 time:27.9       train loss:0.2257=0.0953+ val loss:0.1774 test loss:0.1784
epoch:83 time:28.5       train loss:0.2257=0.0952+ val loss:0.1776 test loss:0.1778
epoch:84 time:27.3       train loss:0.2256=0.0951+ val loss:0.1776 test loss:0.1785
epoch:85 time:27.5       train loss:0.2256=0.0949+ val loss:0.1779 test loss:0.178
epoch:86 time:28.5       train loss:0.2257=0.0949+ val loss:0.1778 test loss:0.1784
epoch:87 time:28.6       train loss:0.2256=0.0947+ val loss:0.1774 test loss:0.1782
epoch:88 time:27.4       train loss:0.2256=0.0948+ val loss:0.1777 test loss:0.1791
epoch:89 time:29.4       train loss:0.2256=0.0947+ val loss:0.178 test loss:0.1784
epoch:90 time:27.2       train loss:0.2257=0.0946+ val loss:0.1781 test loss:0.1788
epoch:91 time:27.0       train loss:0.2256=0.0945+ val loss:0.1782 test loss:0.1787
epoch:92 time:28.9       train loss:0.2255=0.0944+ val loss:0.1781 test loss:0.1785
epoch:93 time:31.2       train loss:0.2255=0.0944+ val loss:0.1777 test loss:0.1785
epoch:94 time:28.5       train loss:0.2255=0.0941+ val loss:0.1782 test loss:0.1785
epoch:95 time:28.8       train loss:0.2254=0.0941+ val loss:0.1781 test loss:0.1788
epoch:96 time:28.1       train loss:0.2253=0.094+ val loss:0.178 test loss:0.1788
epoch:97 time:29.3       train loss:0.2254=0.0941+ val loss:0.178 test loss:0.1789
epoch:98 time:27.5       train loss:0.2254=0.094+ val loss:0.1782 test loss:0.1791
epoch:99 time:30.3       train loss:0.2254=0.094+ val loss:0.1782 test loss:0.1788
epoch:100 time:29.6      train loss:0.2254=0.0937+ val loss:0.178 test loss:0.1793
epoch:101 time:28.3      train loss:0.2254=0.0937+ val loss:0.1785 test loss:0.1792
epoch:102 time:28.3      train loss:0.2253=0.0938+ val loss:0.1781 test loss:0.179
epoch:103 time:28.6      train loss:0.2254=0.0936+ val loss:0.1777 test loss:0.1792
epoch:104 time:28.4      train loss:0.2253=0.0936+ val loss:0.1781 test loss:0.1796
epoch:105 time:28.9      train loss:0.2254=0.0935+ val loss:0.1777 test loss:0.1787
epoch:106 time:29.7      train loss:0.2253=0.0935+ val loss:0.1787 test loss:0.1786
epoch:107 time:26.3      train loss:0.2253=0.0934+ val loss:0.1783 test loss:0.1791
epoch:108 time:29.0      train loss:0.2254=0.0935+ val loss:0.1779 test loss:0.1789
epoch:109 time:29.6      train loss:0.2253=0.0933+ val loss:0.178 test loss:0.179
epoch:110 time:29.5      train loss:0.2252=0.0933+ val loss:0.1785 test loss:0.1794
epoch:111 time:27.5      train loss:0.2253=0.0933+ val loss:0.1781 test loss:0.1793
epoch:112 time:28.1      train loss:0.2253=0.0932+ val loss:0.1784 test loss:0.1782
epoch:113 time:27.7      train loss:0.2253=0.0932+ val loss:0.179 test loss:0.1794
epoch:114 time:29.2      train loss:0.2252=0.0931+ val loss:0.1785 test loss:0.179
epoch:115 time:27.4      train loss:0.2252=0.0931+ val loss:0.1783 test loss:0.1797
epoch:116 time:27.2      train loss:0.2252=0.0928+ val loss:0.1787 test loss:0.1791
epoch:117 time:29.9      train loss:0.2251=0.0928+ val loss:0.1789 test loss:0.1793
epoch:118 time:29.2      train loss:0.2252=0.0929+ val loss:0.1793 test loss:0.1793
epoch:119 time:26.7      train loss:0.2252=0.0928+ val loss:0.1782 test loss:0.1787
epoch:120 time:29.7      train loss:0.2252=0.0928+ val loss:0.1789 test loss:0.1791
epoch:121 time:28.3      train loss:0.2253=0.093+ val loss:0.1783 test loss:0.179
epoch:122 time:27.9      train loss:0.2251=0.0928+ val loss:0.1783 test loss:0.1793
epoch:123 time:28.5      train loss:0.2252=0.0927+ val loss:0.1784 test loss:0.179
epoch:124 time:29.9      train loss:0.2251=0.0926+ val loss:0.1782 test loss:0.1793
epoch:125 time:28.7      train loss:0.2251=0.0927+ val loss:0.1786 test loss:0.1791
epoch:126 time:28.5      train loss:0.2252=0.0927+ val loss:0.1786 test loss:0.1798
epoch:127 time:28.1      train loss:0.2251=0.0926+ val loss:0.1791 test loss:0.1793
epoch:128 time:29.7      train loss:0.2252=0.0927+ val loss:0.1786 test loss:0.1795
epoch:129 time:26.8      train loss:0.2251=0.0925+ val loss:0.1786 test loss:0.1791
epoch:130 time:28.6      train loss:0.2251=0.0925+ val loss:0.179 test loss:0.1793
epoch:131 time:28.9      train loss:0.2251=0.0926+ val loss:0.1781 test loss:0.1792
epoch:132 time:28.0      train loss:0.225=0.0923+ val loss:0.1784 test loss:0.1796
epoch:133 time:28.8      train loss:0.2251=0.0924+ val loss:0.1783 test loss:0.1793
epoch:134 time:27.0      train loss:0.225=0.0923+ val loss:0.1786 test loss:0.1793
epoch:135 time:28.7      train loss:0.225=0.0923+ val loss:0.1787 test loss:0.1797
epoch:136 time:29.6      train loss:0.2251=0.0922+ val loss:0.1785 test loss:0.1799
epoch:137 time:27.7      train loss:0.2251=0.0924+ val loss:0.1787 test loss:0.1791
epoch:138 time:28.1      train loss:0.2251=0.0923+ val loss:0.1788 test loss:0.1795
epoch:139 time:29.3      train loss:0.225=0.0922+ val loss:0.1787 test loss:0.1795
epoch:140 time:29.4      train loss:0.225=0.0922+ val loss:0.1794 test loss:0.1795
epoch:141 time:29.0      train loss:0.225=0.0921+ val loss:0.1789 test loss:0.1799
epoch:142 time:27.6      train loss:0.225=0.0922+ val loss:0.1792 test loss:0.1797
epoch:143 time:27.7      train loss:0.2251=0.0922+ val loss:0.179 test loss:0.1797
epoch:144 time:32.2      train loss:0.225=0.0921+ val loss:0.1796 test loss:0.1797
epoch:145 time:29.4      train loss:0.2249=0.092+ val loss:0.1789 test loss:0.1796
epoch:146 time:27.9      train loss:0.2251=0.092+ val loss:0.1797 test loss:0.1799
epoch:147 time:26.9      train loss:0.225=0.0921+ val loss:0.1791 test loss:0.1796
epoch:148 time:28.4      train loss:0.2249=0.0919+ val loss:0.1789 test loss:0.1794
epoch:149 time:28.6      train loss:0.2251=0.092+ val loss:0.1789 test loss:0.18
epoch:150 time:27.6      train loss:0.225=0.092+ val loss:0.1795 test loss:0.1797
epoch:151 time:29.1      train loss:0.2249=0.0919+ val loss:0.1793 test loss:0.1797
epoch:152 time:27.9      train loss:0.225=0.0919+ val loss:0.1793 test loss:0.1797
epoch:153 time:28.1      train loss:0.2249=0.0919+ val loss:0.179 test loss:0.1801
epoch:154 time:28.1      train loss:0.2251=0.092+ val loss:0.1792 test loss:0.18
epoch:155 time:29.3      train loss:0.225=0.0919+ val loss:0.1791 test loss:0.1803
epoch:156 time:29.0      train loss:0.225=0.0918+ val loss:0.1787 test loss:0.1799
epoch:157 time:29.3      train loss:0.225=0.092+ val loss:0.1797 test loss:0.1802
epoch:158 time:28.1      train loss:0.225=0.0919+ val loss:0.1794 test loss:0.1803
epoch:159 time:29.2      train loss:0.2249=0.0916+ val loss:0.1792 test loss:0.1797
epoch:160 time:27.2      train loss:0.2249=0.0917+ val loss:0.18 test loss:0.1799
epoch:161 time:29.1      train loss:0.2249=0.0917+ val loss:0.1786 test loss:0.1796
epoch:162 time:29.0      train loss:0.225=0.0918+ val loss:0.1795 test loss:0.18
epoch:163 time:28.5      train loss:0.2249=0.0918+ val loss:0.1792 test loss:0.1797
epoch:164 time:28.9      train loss:0.225=0.0917+ val loss:0.179 test loss:0.18
epoch:165 time:28.0      train loss:0.225=0.0918+ val loss:0.1793 test loss:0.1802
epoch:166 time:29.2      train loss:0.2249=0.0916+ val loss:0.1793 test loss:0.1797
epoch:167 time:30.3      train loss:0.225=0.0916+ val loss:0.1792 test loss:0.1796
epoch:168 time:27.3      train loss:0.2251=0.0918+ val loss:0.1795 test loss:0.1802
epoch:169 time:28.0      train loss:0.225=0.0917+ val loss:0.179 test loss:0.1802
epoch:170 time:28.7      train loss:0.2249=0.0915+ val loss:0.1792 test loss:0.1796
epoch:171 time:29.3      train loss:0.2249=0.0916+ val loss:0.18 test loss:0.1794
epoch:172 time:28.5      train loss:0.225=0.0918+ val loss:0.1789 test loss:0.1802
epoch:173 time:27.2      train loss:0.2249=0.0915+ val loss:0.1797 test loss:0.1803
epoch:174 time:28.1      train loss:0.225=0.0917+ val loss:0.1797 test loss:0.18
epoch:175 time:28.3      train loss:0.2249=0.0914+ val loss:0.1793 test loss:0.1802
epoch:176 time:29.8      train loss:0.2249=0.0914+ val loss:0.1796 test loss:0.18
epoch:177 time:28.8      train loss:0.2249=0.0914+ val loss:0.1793 test loss:0.1795
epoch:178 time:27.8      train loss:0.2249=0.0915+ val loss:0.1796 test loss:0.1804
epoch:179 time:29.3      train loss:0.2249=0.0916+ val loss:0.1796 test loss:0.1797
epoch:180 time:28.4      train loss:0.225=0.0915+ val loss:0.1794 test loss:0.1798
epoch:181 time:26.6      train loss:0.225=0.0914+ val loss:0.1797 test loss:0.1803
epoch:182 time:29.4      train loss:0.2249=0.0916+ val loss:0.1796 test loss:0.1801
epoch:183 time:28.3      train loss:0.2249=0.0914+ val loss:0.1792 test loss:0.18
epoch:184 time:27.8      train loss:0.2249=0.0914+ val loss:0.1798 test loss:0.1801
epoch:185 time:28.7      train loss:0.2249=0.0914+ val loss:0.1795 test loss:0.1799
epoch:186 time:30.2      train loss:0.225=0.0915+ val loss:0.1798 test loss:0.1805
epoch:187 time:32.1      train loss:0.2249=0.0915+ val loss:0.1795 test loss:0.1802
epoch:188 time:29.9      train loss:0.2249=0.0914+ val loss:0.1794 test loss:0.181
epoch:189 time:28.6      train loss:0.2248=0.0913+ val loss:0.1795 test loss:0.1803
epoch:190 time:29.2      train loss:0.2249=0.0913+ val loss:0.1794 test loss:0.1804
epoch:191 time:28.1      train loss:0.2248=0.0913+ val loss:0.1797 test loss:0.1803
epoch:192 time:29.6      train loss:0.225=0.0916+ val loss:0.1792 test loss:0.1806
epoch:193 time:29.2      train loss:0.2248=0.0913+ val loss:0.1793 test loss:0.1797
epoch:194 time:29.3      train loss:0.2248=0.0912+ val loss:0.1796 test loss:0.1808
epoch:195 time:27.9      train loss:0.2249=0.0913+ val loss:0.1795 test loss:0.18
epoch:196 time:27.6      train loss:0.2249=0.0913+ val loss:0.1798 test loss:0.1805
epoch:197 time:28.6      train loss:0.225=0.0915+ val loss:0.1796 test loss:0.1807
epoch:198 time:31.2      train loss:0.2249=0.0913+ val loss:0.1796 test loss:0.1797
epoch:199 time:27.8      train loss:0.2249=0.0913+ val loss:0.1801 test loss:0.1809
epoch:200 time:28.0      train loss:0.2249=0.0913+ val loss:0.1801 test loss:0.1805
epoch:201 time:27.6      train loss:0.2248=0.0911+ val loss:0.1793 test loss:0.1802
epoch:202 time:29.6      train loss:0.225=0.0914+ val loss:0.1801 test loss:0.181
epoch:203 time:29.0      train loss:0.225=0.0914+ val loss:0.1792 test loss:0.1809
epoch:204 time:27.7      train loss:0.2248=0.0912+ val loss:0.18 test loss:0.1802
epoch:205 time:28.0      train loss:0.2248=0.0912+ val loss:0.1796 test loss:0.1803
epoch:206 time:28.6      train loss:0.2249=0.0913+ val loss:0.1799 test loss:0.1807
epoch:207 time:28.8      train loss:0.2248=0.0911+ val loss:0.1795 test loss:0.1805
epoch:208 time:33.3      train loss:0.225=0.0913+ val loss:0.18 test loss:0.1805
epoch:209 time:29.9      train loss:0.2249=0.0912+ val loss:0.1793 test loss:0.1801
epoch:210 time:29.9      train loss:0.2249=0.0914+ val loss:0.18 test loss:0.1805
epoch:211 time:30.4      train loss:0.225=0.0914+ val loss:0.1801 test loss:0.1802
epoch:212 time:27.3      train loss:0.2249=0.0912+ val loss:0.1799 test loss:0.1802
epoch:213 time:30.3      train loss:0.2248=0.0913+ val loss:0.1794 test loss:0.1806
epoch:214 time:29.5      train loss:0.2248=0.0911+ val loss:0.1798 test loss:0.1807
epoch:215 time:29.9      train loss:0.2249=0.0912+ val loss:0.1801 test loss:0.1801
epoch:216 time:32.5      train loss:0.2249=0.0912+ val loss:0.1797 test loss:0.1803
epoch:217 time:30.4      train loss:0.2248=0.091+ val loss:0.1795 test loss:0.1802
epoch:218 time:29.1      train loss:0.2247=0.0911+ val loss:0.1797 test loss:0.1801
epoch:219 time:30.3      train loss:0.2248=0.0911+ val loss:0.1794 test loss:0.18
epoch:220 time:28.2      train loss:0.2249=0.0912+ val loss:0.18 test loss:0.1804
epoch:221 time:32.1      train loss:0.2248=0.0912+ val loss:0.1803 test loss:0.1802
epoch:222 time:28.0      train loss:0.2248=0.0911+ val loss:0.1797 test loss:0.1806
epoch:223 time:32.0      train loss:0.2249=0.0912+ val loss:0.1793 test loss:0.1803
epoch:224 time:35.9      train loss:0.2249=0.0912+ val loss:0.1796 test loss:0.1808
epoch:225 time:29.8      train loss:0.2247=0.091+ val loss:0.1796 test loss:0.1803
epoch:226 time:28.9      train loss:0.2249=0.0912+ val loss:0.1804 test loss:0.1807
epoch:227 time:29.0      train loss:0.2248=0.0912+ val loss:0.1797 test loss:0.1795
epoch:228 time:32.4      train loss:0.2249=0.0912+ val loss:0.1796 test loss:0.1801
epoch:229 time:30.7      train loss:0.2248=0.091+ val loss:0.1799 test loss:0.1806
epoch:230 time:34.4      train loss:0.2249=0.0912+ val loss:0.1802 test loss:0.1805
epoch:231 time:34.8      train loss:0.2249=0.0911+ val loss:0.1801 test loss:0.1805
epoch:232 time:28.5      train loss:0.2248=0.0911+ val loss:0.1798 test loss:0.1801
epoch:233 time:30.1      train loss:0.2248=0.091+ val loss:0.1801 test loss:0.1803
epoch:234 time:29.1      train loss:0.2248=0.0911+ val loss:0.1798 test loss:0.1804
epoch:235 time:28.3      train loss:0.2248=0.091+ val loss:0.1793 test loss:0.1807
epoch:236 time:30.9      train loss:0.2248=0.0911+ val loss:0.1796 test loss:0.1804
epoch:237 time:33.0      train loss:0.2249=0.0911+ val loss:0.1801 test loss:0.1804
epoch:238 time:29.9      train loss:0.2249=0.0911+ val loss:0.1799 test loss:0.1802
epoch:239 time:31.8      train loss:0.2249=0.091+ val loss:0.1803 test loss:0.1803
epoch:240 time:28.7      train loss:0.2248=0.091+ val loss:0.18 test loss:0.1802
epoch:241 time:29.8      train loss:0.2248=0.091+ val loss:0.1792 test loss:0.1805
epoch:242 time:30.0      train loss:0.2249=0.0911+ val loss:0.1798 test loss:0.1807
epoch:243 time:31.1      train loss:0.2248=0.091+ val loss:0.1796 test loss:0.1804
epoch:244 time:31.3      train loss:0.2249=0.091+ val loss:0.1791 test loss:0.18
epoch:245 time:28.4      train loss:0.2249=0.0911+ val loss:0.1795 test loss:0.1797
epoch:246 time:30.6      train loss:0.2249=0.0911+ val loss:0.1798 test loss:0.1807
epoch:247 time:33.1      train loss:0.2249=0.0911+ val loss:0.1802 test loss:0.1807
epoch:248 time:30.0      train loss:0.2248=0.0911+ val loss:0.1799 test loss:0.1808
epoch:249 time:29.0      train loss:0.2248=0.091+ val loss:0.1802 test loss:0.1803
epoch:250 time:29.0      train loss:0.2248=0.091+ val loss:0.18 test loss:0.1802
epoch:251 time:31.3      train loss:0.225=0.0912+ val loss:0.1798 test loss:0.1807
epoch:252 time:31.2      train loss:0.2247=0.0909+ val loss:0.1797 test loss:0.1806
epoch:253 time:28.8      train loss:0.2248=0.091+ val loss:0.1804 test loss:0.1803
epoch:254 time:34.0      train loss:0.2247=0.0909+ val loss:0.1793 test loss:0.1799
epoch:255 time:30.0      train loss:0.2249=0.091+ val loss:0.1798 test loss:0.1802
epoch:256 time:29.2      train loss:0.2248=0.091+ val loss:0.1793 test loss:0.1801
epoch:257 time:34.0      train loss:0.2247=0.0908+ val loss:0.1797 test loss:0.1804
epoch:258 time:28.1      train loss:0.2248=0.091+ val loss:0.1801 test loss:0.1803
epoch:259 time:29.5      train loss:0.2247=0.0909+ val loss:0.1798 test loss:0.1802
epoch:260 time:31.1      train loss:0.2248=0.0909+ val loss:0.1803 test loss:0.1803
epoch:261 time:29.3      train loss:0.2248=0.091+ val loss:0.1799 test loss:0.1805
epoch:262 time:29.2      train loss:0.2249=0.0911+ val loss:0.1798 test loss:0.1809
epoch:263 time:30.0      train loss:0.2248=0.091+ val loss:0.1791 test loss:0.1797
epoch:264 time:30.1      train loss:0.2249=0.0909+ val loss:0.1797 test loss:0.1797
epoch:265 time:33.1      train loss:0.2249=0.0911+ val loss:0.1799 test loss:0.1801
epoch:266 time:28.6      train loss:0.2249=0.091+ val loss:0.1803 test loss:0.1803
epoch:267 time:28.2      train loss:0.2248=0.091+ val loss:0.1804 test loss:0.1805
epoch:268 time:30.4      train loss:0.2248=0.091+ val loss:0.1796 test loss:0.1806
epoch:269 time:29.9      train loss:0.2248=0.0909+ val loss:0.1801 test loss:0.1804
epoch:270 time:31.2      train loss:0.2248=0.0909+ val loss:0.1798 test loss:0.1809
epoch:271 time:30.3      train loss:0.2248=0.0911+ val loss:0.18 test loss:0.1804
epoch:272 time:31.6      train loss:0.2248=0.0909+ val loss:0.1802 test loss:0.18
epoch:273 time:29.4      train loss:0.2249=0.091+ val loss:0.1801 test loss:0.1807
epoch:274 time:28.9      train loss:0.2248=0.0909+ val loss:0.1801 test loss:0.1808
epoch:275 time:30.4      train loss:0.2248=0.0909+ val loss:0.1798 test loss:0.1804
epoch:276 time:28.9      train loss:0.2247=0.0908+ val loss:0.1802 test loss:0.1805
epoch:277 time:30.7      train loss:0.225=0.0911+ val loss:0.1803 test loss:0.1807
epoch:278 time:29.4      train loss:0.2248=0.0909+ val loss:0.1799 test loss:0.1805
epoch:279 time:31.6      train loss:0.2247=0.0908+ val loss:0.1799 test loss:0.1799
epoch:280 time:30.3      train loss:0.2249=0.0909+ val loss:0.1795 test loss:0.1802
epoch:281 time:30.2      train loss:0.2248=0.0909+ val loss:0.1801 test loss:0.181
epoch:282 time:29.6      train loss:0.2249=0.091+ val loss:0.18 test loss:0.1811
epoch:283 time:31.1      train loss:0.2248=0.0909+ val loss:0.1796 test loss:0.1806
epoch:284 time:29.0      train loss:0.2248=0.0908+ val loss:0.1801 test loss:0.1808
epoch:285 time:33.5      train loss:0.2247=0.0909+ val loss:0.18 test loss:0.1809
epoch:286 time:31.0      train loss:0.2248=0.0907+ val loss:0.1796 test loss:0.1802
epoch:287 time:29.2      train loss:0.2248=0.091+ val loss:0.1802 test loss:0.1814
epoch:288 time:30.0      train loss:0.2248=0.0909+ val loss:0.1799 test loss:0.1805
epoch:289 time:28.4      train loss:0.2248=0.0909+ val loss:0.18 test loss:0.1802
epoch:290 time:29.3      train loss:0.2248=0.0909+ val loss:0.1801 test loss:0.1803
epoch:291 time:31.1      train loss:0.2248=0.0909+ val loss:0.1802 test loss:0.1804
epoch:292 time:29.4      train loss:0.2248=0.0909+ val loss:0.1799 test loss:0.1793
epoch:293 time:30.5      train loss:0.2248=0.0908+ val loss:0.1796 test loss:0.1805
epoch:294 time:30.3      train loss:0.2249=0.091+ val loss:0.18 test loss:0.1808
epoch:295 time:30.1      train loss:0.2248=0.0909+ val loss:0.18 test loss:0.1798
epoch:296 time:31.5      train loss:0.2249=0.0909+ val loss:0.1798 test loss:0.1799
epoch:297 time:27.7      train loss:0.2248=0.0909+ val loss:0.18 test loss:0.1805
epoch:298 time:30.0      train loss:0.2248=0.0907+ val loss:0.1801 test loss:0.1805
epoch:299 time:28.7      train loss:0.2247=0.0908+ val loss:0.1795 test loss:0.1808
epoch:300 time:30.7      train loss:0.2249=0.0909+ val loss:0.1803 test loss:0.1801
epoch:301 time:28.3      train loss:0.2249=0.091+ val loss:0.1796 test loss:0.1804
epoch:302 time:31.9      train loss:0.2248=0.0908+ val loss:0.1798 test loss:0.1806
epoch:303 time:30.1      train loss:0.2247=0.0908+ val loss:0.1801 test loss:0.1803
epoch:304 time:29.4      train loss:0.2248=0.0908+ val loss:0.1792 test loss:0.1806
epoch:305 time:28.1      train loss:0.2248=0.0908+ val loss:0.1798 test loss:0.1798
epoch:306 time:32.2      train loss:0.2248=0.0909+ val loss:0.1797 test loss:0.1812
epoch:307 time:32.1      train loss:0.2248=0.0909+ val loss:0.1796 test loss:0.1808
epoch:308 time:28.6      train loss:0.2247=0.0908+ val loss:0.1798 test loss:0.1805
epoch:309 time:30.5      train loss:0.2248=0.0909+ val loss:0.1803 test loss:0.1803
epoch:310 time:29.0      train loss:0.2248=0.0909+ val loss:0.1794 test loss:0.1811
epoch:311 time:28.2      train loss:0.2249=0.0908+ val loss:0.1802 test loss:0.1808
epoch:312 time:30.2      train loss:0.2248=0.0908+ val loss:0.1801 test loss:0.1805
epoch:313 time:29.6      train loss:0.2247=0.0907+ val loss:0.1795 test loss:0.1805
epoch:314 time:29.6      train loss:0.2248=0.0909+ val loss:0.1801 test loss:0.1807
epoch:315 time:28.3      train loss:0.2249=0.091+ val loss:0.18 test loss:0.1811
epoch:316 time:29.7      train loss:0.2247=0.0908+ val loss:0.1801 test loss:0.1804
epoch:317 time:30.6      train loss:0.2248=0.0908+ val loss:0.18 test loss:0.18
epoch:318 time:29.6      train loss:0.2248=0.0909+ val loss:0.18 test loss:0.1806
epoch:319 time:30.6      train loss:0.2248=0.0908+ val loss:0.1796 test loss:0.1804
epoch:320 time:28.0      train loss:0.225=0.091+ val loss:0.1803 test loss:0.1811
epoch:321 time:31.6      train loss:0.2248=0.091+ val loss:0.1802 test loss:0.1808
epoch:322 time:34.8      train loss:0.2247=0.0907+ val loss:0.1803 test loss:0.1807
epoch:323 time:29.7      train loss:0.2248=0.0908+ val loss:0.1802 test loss:0.1804
epoch:324 time:30.0      train loss:0.2248=0.0909+ val loss:0.1803 test loss:0.1802
epoch:325 time:30.0      train loss:0.2248=0.0908+ val loss:0.1796 test loss:0.1805
epoch:326 time:29.9      train loss:0.2248=0.0908+ val loss:0.1799 test loss:0.1803
epoch:327 time:29.5      train loss:0.2248=0.0908+ val loss:0.1802 test loss:0.1806
epoch:328 time:28.8      train loss:0.2248=0.0908+ val loss:0.1803 test loss:0.1806
epoch:329 time:28.3      train loss:0.2248=0.0908+ val loss:0.1798 test loss:0.1802
epoch:330 time:29.2      train loss:0.2248=0.0908+ val loss:0.1804 test loss:0.1803
epoch:331 time:30.1      train loss:0.2248=0.0909+ val loss:0.1798 test loss:0.181
epoch:332 time:29.6      train loss:0.2247=0.0907+ val loss:0.1797 test loss:0.1805
epoch:333 time:27.6      train loss:0.2247=0.0907+ val loss:0.1794 test loss:0.1807
epoch:334 time:29.9      train loss:0.2247=0.0908+ val loss:0.1802 test loss:0.1803
epoch:335 time:30.3      train loss:0.2248=0.0908+ val loss:0.1801 test loss:0.1808
epoch:336 time:27.8      train loss:0.2248=0.0909+ val loss:0.1798 test loss:0.1805
epoch:337 time:30.6      train loss:0.2248=0.0909+ val loss:0.1797 test loss:0.181
epoch:338 time:28.3      train loss:0.2248=0.0908+ val loss:0.18 test loss:0.1808
epoch:339 time:29.6      train loss:0.2249=0.0909+ val loss:0.1799 test loss:0.1807
epoch:340 time:30.0      train loss:0.2248=0.0908+ val loss:0.1801 test loss:0.1804
epoch:341 time:29.8      train loss:0.2247=0.0907+ val loss:0.1801 test loss:0.1806
epoch:342 time:29.3      train loss:0.2248=0.0908+ val loss:0.1799 test loss:0.1809
epoch:343 time:30.2      train loss:0.2248=0.0908+ val loss:0.1798 test loss:0.1805
epoch:344 time:28.7      train loss:0.2247=0.0906+ val loss:0.18 test loss:0.181
epoch:345 time:30.4      train loss:0.2247=0.0907+ val loss:0.1808 test loss:0.1807
epoch:346 time:28.3      train loss:0.2249=0.0909+ val loss:0.1802 test loss:0.1805
epoch:347 time:31.5      train loss:0.2248=0.0909+ val loss:0.1797 test loss:0.1805
epoch:348 time:30.7      train loss:0.2248=0.0908+ val loss:0.1801 test loss:0.1798
epoch:349 time:28.9      train loss:0.2248=0.0907+ val loss:0.18 test loss:0.1801

```

#### test

```shell
s0
has results save path
has model save path
--------test processing-------
-0.0003578 0.0421327 3.96e-05 0.0108496 0.0002385 0.007829 0.0002654 0.0056104 4.64e-05 0.0166055
-6.29e-05 0.024128 0.0001966 0.0110474 0.0001631 0.0059425 0.0002333 0.0051151 0.0001325 0.0115583
epoch:130time:119.51     test hit:0.0554 ndcg:0.034 recall:0.0551
-0.0003689 0.0421742 3.23e-05 0.0108534 0.0002286 0.007831 0.0002579 0.0056109 3.75e-05 0.0166174
-6.86e-05 0.0241487 0.0001888 0.0110531 0.0001572 0.0059433 0.0002242 0.0051162 0.0001254 0.0115653
epoch:131time:117.72     test hit:0.0554 ndcg:0.0341 recall:0.055
-0.0003878 0.0422146 2.34e-05 0.0108581 0.0002102 0.007837 0.0002484 0.0056131 2.36e-05 0.0166308
-7.88e-05 0.0241684 0.0001762 0.0110621 0.0001504 0.0059455 0.0002107 0.0051205 0.0001146 0.0115741
epoch:132time:116.08     test hit:0.0557 ndcg:0.0344 recall:0.0554
-0.0003861 0.0422353 1.99e-05 0.0108614 0.0002098 0.0078355 0.0002449 0.0056151 2.21e-05 0.0166369
-8.2e-05 0.0241781 0.0001769 0.0110616 0.0001481 0.0059472 0.0002113 0.0051197 0.0001136 0.0115767
epoch:133time:116.83     test hit:0.0557 ndcg:0.0343 recall:0.0553
-0.0003654 0.0422416 2.54e-05 0.0108548 0.0002263 0.0078245 0.0002508 0.0056075 3.43e-05 0.0166321
-7.79e-05 0.0241808 0.0001912 0.0110532 0.0001535 0.0059407 0.0002247 0.0051109 0.0001228 0.0115714
epoch:134time:115.77     test hit:0.0559 ndcg:0.0343 recall:0.0556
-0.0003533 0.0422691 4.32e-05 0.0108563 0.0002402 0.0078286 0.0002686 0.0056086 4.97e-05 0.0166407
-6.33e-05 0.0241926 0.0002026 0.0110583 0.0001684 0.0059414 0.000236 0.0051144 0.0001359 0.0115767
epoch:135time:118.37     test hit:0.056 ndcg:0.0344 recall:0.0556
-0.0003666 0.0422967 3.62e-05 0.0108609 0.0002297 0.0078311 0.000262 0.0056129 4.03e-05 0.0166505
-6.55e-05 0.0242046 0.0001956 0.0110615 0.0001633 0.0059448 0.0002281 0.005117 0.0001304 0.011582
epoch:136time:115.31     test hit:0.0559 ndcg:0.0342 recall:0.0555
-0.0003641 0.0423193 2.68e-05 0.0108527 0.0002297 0.0078209 0.000253 0.0056033 3.64e-05 0.0166491
-7.31e-05 0.0242097 0.0001942 0.0110557 0.0001546 0.0059364 0.0002264 0.0051085 0.0001255 0.0115776
epoch:137time:118.23     test hit:0.056 ndcg:0.0342 recall:0.0556
-0.0003688 0.0423563 2.26e-05 0.0108593 0.0002267 0.0078292 0.000249 0.0056078 3.24e-05 0.0166632
-7.35e-05 0.0242226 0.0001908 0.0110657 0.0001508 0.0059406 0.0002231 0.0051143 0.0001228 0.0115858
epoch:138time:115.31     test hit:0.0563 ndcg:0.0343 recall:0.0559
-0.0003805 0.0423648 1.67e-05 0.010857 0.0002169 0.0078241 0.0002428 0.0056048 2.4e-05 0.0166627
-7.5e-05 0.0242272 0.0001853 0.0110622 0.0001469 0.005938 0.000217 0.0051103 0.0001185 0.0115844
epoch:139time:118.38     test hit:0.0556 ndcg:0.0341 recall:0.0552
-0.0003722 0.0423839 2.1e-05 0.0108564 0.0002262 0.0078228 0.0002472 0.0056035 3.06e-05 0.0166667
-6.51e-05 0.0242363 0.000195 0.0110623 0.0001517 0.0059368 0.0002252 0.0051094 0.0001267 0.0115862
epoch:140time:115.61     test hit:0.0557 ndcg:0.0341 recall:0.0554
-0.000372 0.0423966 3.23e-05 0.0108587 0.0002262 0.0078246 0.0002572 0.0056063 3.59e-05 0.0166716
-5.47e-05 0.024239 0.0001959 0.011064 0.0001605 0.005939 0.0002256 0.0051113 0.0001318 0.0115883
epoch:141time:120.76     test hit:0.0558 ndcg:0.0342 recall:0.0555
-0.0003823 0.0424089 3.2e-05 0.01085 0.0002183 0.0078131 0.0002571 0.0055971 3.13e-05 0.0166673
-5.2e-05 0.0242443 0.0001934 0.0110563 0.0001618 0.005931 0.000221 0.005102 0.0001311 0.0115834
epoch:142time:118.18     test hit:0.0562 ndcg:0.0344 recall:0.0559
-0.0003667 0.0424552 3.23e-05 0.010856 0.0002293 0.0078225 0.0002583 0.0055996 3.83e-05 0.0166834
-5.2e-05 0.0242663 0.0002028 0.0110686 0.0001636 0.0059337 0.0002305 0.0051086 0.0001362 0.0115943
epoch:143time:117.69     test hit:0.0557 ndcg:0.0343 recall:0.0554
-0.0003659 0.0424729 2.15e-05 0.0108544 0.0002257 0.0078161 0.0002482 0.0055957 3.24e-05 0.0166848
-6.45e-05 0.0242813 0.0001971 0.0110654 0.0001541 0.0059306 0.000226 0.0051034 0.0001282 0.0115952
epoch:144time:117.91     test hit:0.0558 ndcg:0.0343 recall:0.0555
-0.000373 0.0425067 1.8e-05 0.0108603 0.0002184 0.0078224 0.0002446 0.0055997 2.7e-05 0.0166973
-6.81e-05 0.0243007 0.0001897 0.0110729 0.0001494 0.0059342 0.0002185 0.0051084 0.0001224 0.011604
epoch:145time:117.71     test hit:0.0561 ndcg:0.0344 recall:0.0558
-0.000383 0.0425203 1.56e-05 0.01086 0.0002109 0.0078183 0.0002421 0.0055988 2.14e-05 0.0166994
-6.77e-05 0.0243059 0.0001837 0.0110703 0.0001456 0.0059332 0.0002113 0.0051052 0.0001182 0.0116037
epoch:146time:116.88     test hit:0.0558 ndcg:0.0342 recall:0.0555
-0.0004024 0.0425232 3.8e-06 0.0108524 0.0001932 0.0078037 0.0002294 0.0055909 6e-06 0.0166926
-6.88e-05 0.0243071 0.0001724 0.0110591 0.0001375 0.0059262 0.0001983 0.0050939 0.0001098 0.0115966
epoch:147time:118.19     test hit:0.0558 ndcg:0.0342 recall:0.0554
-0.0003853 0.0425717 1.06e-05 0.0108617 0.000208 0.0078176 0.0002365 0.0055971 1.74e-05 0.0167121
-5.84e-05 0.0243256 0.0001856 0.0110748 0.0001454 0.005932 0.0002105 0.0051041 0.0001208 0.0116092
epoch:148time:116.91     test hit:0.0561 ndcg:0.0343 recall:0.0557
-0.0003768 0.0425861 1.42e-05 0.0108616 0.0002126 0.0078158 0.0002396 0.0055963 2.24e-05 0.016715
-5.28e-05 0.0243317 0.0001937 0.0110741 0.0001505 0.0059314 0.0002171 0.0051029 0.0001271 0.01161
epoch:149time:118.91     test hit:0.0567 ndcg:0.0345 recall:0.0563
-0.0003811 0.0425987 1.83e-05 0.0108572 0.0002098 0.0078102 0.0002437 0.0055923 2.26e-05 0.0167147
-5e-05 0.0243349 0.0001938 0.0110699 0.0001545 0.0059275 0.0002165 0.005099 0.0001287 0.0116078
epoch:150time:117.41     test hit:0.0567 ndcg:0.0344 recall:0.0563
-0.0003938 0.0426166 7.5e-06 0.0108601 0.0001985 0.0078115 0.0002334 0.0055942 1.14e-05 0.0167206
-5.73e-05 0.0243446 0.0001866 0.0110719 0.0001466 0.0059292 0.0002083 0.0051002 0.0001211 0.0116115
epoch:151time:117.28     test hit:0.0564 ndcg:0.0345 recall:0.0561
-0.0003702 0.0426262 1.88e-05 0.0108564 0.0002211 0.0078075 0.0002455 0.0055912 2.88e-05 0.0167204
-4.82e-05 0.0243469 0.0002023 0.0110693 0.0001553 0.0059263 0.0002246 0.0050971 0.0001335 0.0116099
epoch:152time:117.97     test hit:0.0562 ndcg:0.0345 recall:0.0558
-0.0003668 0.0426487 2.31e-05 0.0108569 0.0002248 0.0078083 0.0002494 0.0055903 3.26e-05 0.0167261
-4.09e-05 0.0243561 0.0002068 0.0110721 0.0001595 0.0059257 0.000228 0.0050974 0.0001384 0.0116128
epoch:153time:117.49     test hit:0.0561 ndcg:0.0344 recall:0.0557
-0.0003721 0.0426684 2.52e-05 0.010857 0.0002224 0.0078094 0.0002511 0.0055902 3.17e-05 0.0167313
-3.75e-05 0.0243654 0.0002056 0.0110743 0.0001617 0.0059254 0.0002266 0.0050984 0.0001391 0.0116159
epoch:154time:116.03     test hit:0.0562 ndcg:0.0344 recall:0.0559
-0.0003874 0.042687 1.8e-05 0.0108578 0.0002129 0.0078099 0.0002443 0.0055904 2.2e-05 0.0167363
-3.81e-05 0.0243724 0.0001999 0.0110761 0.0001576 0.0059253 0.0002199 0.0050988 0.0001348 0.0116182
epoch:155time:116.27     test hit:0.056 ndcg:0.0344 recall:0.0556
-0.0003979 0.0426959 1.26e-05 0.0108576 0.0002056 0.0078084 0.0002393 0.00559 1.49e-05 0.016738
-4.3e-05 0.0243774 0.0001933 0.0110757 0.0001537 0.0059249 0.0002138 0.0050978 0.0001294 0.011619
epoch:156time:116.05     test hit:0.0562 ndcg:0.0344 recall:0.0558
-0.0004108 0.0427045 4.7e-06 0.0108479 0.0001938 0.0077944 0.0002309 0.0055787 4.6e-06 0.0167314
-4.64e-05 0.0243826 0.000184 0.0110657 0.000147 0.0059152 0.0002039 0.0050864 0.0001221 0.0116125
epoch:157time:116.13     test hit:0.0561 ndcg:0.0343 recall:0.0558
-0.0003915 0.0427515 1.26e-05 0.0108606 0.0002059 0.0078084 0.0002388 0.0055887 1.65e-05 0.0167523
-3.75e-05 0.0244055 0.0001945 0.0110805 0.0001543 0.005924 0.000214 0.0050972 0.0001313 0.0116268
epoch:158time:116.39     test hit:0.0561 ndcg:0.0342 recall:0.0557
-0.0003976 0.0427631 1.39e-05 0.0108647 0.0001971 0.0078072 0.0002389 0.0055916 1.31e-05 0.0167567
-3.29e-05 0.0244163 0.0001894 0.0110801 0.0001566 0.0059268 0.0002092 0.0050965 0.0001306 0.0116299
epoch:159time:116.59     test hit:0.0564 ndcg:0.0343 recall:0.0561
-0.0003838 0.0427642 2.14e-05 0.0108573 0.0002093 0.0077939 0.0002464 0.0055834 2.33e-05 0.0167497
-2.38e-05 0.0244217 0.0002008 0.0110696 0.000164 0.0059197 0.0002198 0.0050861 0.0001402 0.0116243
epoch:160time:116.36     test hit:0.0563 ndcg:0.0343 recall:0.0559
-0.0003791 0.042805 1.92e-05 0.0108639 0.0002125 0.0078081 0.0002446 0.0055881 2.43e-05 0.0167663
-2.8e-05 0.0244352 0.0002041 0.0110848 0.0001627 0.005924 0.0002225 0.0050966 0.0001403 0.0116352
epoch:161time:116.2      test hit:0.0564 ndcg:0.0343 recall:0.056
-0.000387 0.0428089 9.1e-06 0.0108555 0.0002053 0.0078005 0.0002342 0.0055808 1.54e-05 0.0167615
-4.13e-05 0.0244324 0.000197 0.0110784 0.0001524 0.0059173 0.000216 0.0050908 0.0001311 0.0116297
epoch:162time:116.2      test hit:0.0562 ndcg:0.0343 recall:0.0559
-0.0003863 0.0428314 6.8e-06 0.0108575 0.000207 0.0078042 0.0002318 0.0055823 1.48e-05 0.0167689
-4.44e-05 0.0244391 0.0001963 0.0110828 0.0001495 0.0059183 0.0002161 0.0050938 0.0001294 0.0116335
epoch:163time:116.25     test hit:0.0563 ndcg:0.0343 recall:0.0559
-0.0004092 0.0428389 1e-07 0.0108588 0.000189 0.007799 0.0002245 0.005583 1.1e-06 0.01677
-4.57e-05 0.024442 0.0001832 0.0110785 0.0001441 0.0059189 0.0002023 0.0050903 0.000121 0.0116324
epoch:164time:116.4      test hit:0.0564 ndcg:0.0343 recall:0.056
-0.000424 0.0428443 -6.6e-06 0.0108529 0.0001768 0.0077884 0.0002178 0.0055755 -9e-06 0.0167653
-5.1e-05 0.0244482 0.0001739 0.0110716 0.0001388 0.0059126 0.000193 0.0050815 0.0001137 0.0116285
epoch:165time:116.66     test hit:0.0564 ndcg:0.0344 recall:0.056
-0.0004214 0.0428887 -1.16e-05 0.0108654 0.0001776 0.0078111 0.0002127 0.0055856 -1.07e-05 0.0167878
-5.32e-05 0.0244687 0.0001739 0.0110945 0.0001349 0.0059218 0.000193 0.0050985 0.0001122 0.0116459
epoch:166time:116.14     test hit:0.0563 ndcg:0.0344 recall:0.056
-0.0004237 0.0428845 -1.77e-05 0.0108645 0.000172 0.0078027 0.0002064 0.0055843 -1.57e-05 0.0167841
-5.64e-05 0.0244709 0.0001704 0.0110863 0.0001302 0.0059208 0.000188 0.0050924 0.0001081 0.0116426
epoch:167time:116.63     test hit:0.0566 ndcg:0.0344 recall:0.0562
-0.0004202 0.0428795 -1.16e-05 0.0108552 0.0001762 0.0077895 0.0002127 0.0055755 -1.07e-05 0.016775
-4.16e-05 0.0244658 0.0001783 0.0110752 0.0001394 0.005913 0.0001949 0.0050822 0.0001177 0.011634
epoch:168time:116.37     test hit:0.0564 ndcg:0.0343 recall:0.056
-0.0004082 0.0429104 -3.8e-06 0.0108639 0.000188 0.0078029 0.0002204 0.0055827 -9e-07 0.01679
-3.22e-05 0.0244774 0.0001897 0.0110888 0.0001465 0.0059193 0.0002056 0.0050923 0.0001274 0.0116445
epoch:169time:116.95     test hit:0.0563 ndcg:0.0343 recall:0.0559
-0.0004192 0.0429129 -1.4e-06 0.0108625 0.0001792 0.0077994 0.000222 0.0055818 -4.9e-06 0.0167892
-3.27e-05 0.0244812 0.0001827 0.0110855 0.0001467 0.0059184 0.0001982 0.00509 0.0001237 0.0116438
epoch:170time:116.43     test hit:0.0566 ndcg:0.0343 recall:0.0562
-0.0004269 0.0429101 -2.9e-06 0.0108553 0.0001733 0.0077886 0.0002207 0.0055762 -8.9e-06 0.0167826
-3.38e-05 0.0244783 0.0001796 0.0110756 0.0001457 0.005913 0.0001944 0.0050821 0.0001215 0.0116373
epoch:171time:116.85     test hit:0.0563 ndcg:0.0343 recall:0.056
-0.0004144 0.0429373 -2e-06 0.010854 0.0001847 0.0077908 0.0002232 0.0055748 -2.1e-06 0.0167893
-2.89e-05 0.0244878 0.0001906 0.011079 0.0001486 0.0059114 0.0002045 0.005084 0.0001287 0.0116405
epoch:172time:116.57     test hit:0.0561 ndcg:0.0342 recall:0.0557
-0.0004076 0.0429644 -3.3e-06 0.0108552 0.0001912 0.0077931 0.000223 0.0055747 8e-07 0.0167969
-2.87e-05 0.0244982 0.0001955 0.0110831 0.000148 0.0059113 0.0002094 0.0050856 0.000131 0.0116446
epoch:173time:116.75     test hit:0.0563 ndcg:0.0343 recall:0.056
-0.0004041 0.0429798 -2.8e-06 0.0108526 0.0001944 0.0077857 0.0002238 0.0055694 2.8e-06 0.016797
-2.72e-05 0.0245105 0.0001982 0.0110796 0.0001493 0.0059072 0.000212 0.0050792 0.0001331 0.0116441
epoch:174time:116.82     test hit:0.0565 ndcg:0.0344 recall:0.0562
-0.0004058 0.0430211 5.3e-06 0.0108672 0.0001918 0.0078023 0.0002316 0.0055813 5.7e-06 0.016818
-2.02e-05 0.0245295 0.0001953 0.0110965 0.0001567 0.0059178 0.0002099 0.0050918 0.0001354 0.0116589
epoch:175time:116.54     test hit:0.0566 ndcg:0.0344 recall:0.0563
-0.0004184 0.0430119 -4e-06 0.0108598 0.0001762 0.0077892 0.000221 0.0055738 -6.3e-06 0.0168087
-3.06e-05 0.0245245 0.0001833 0.0110849 0.0001485 0.0059114 0.0001981 0.0050815 0.0001248 0.0116506
epoch:176time:116.6      test hit:0.0567 ndcg:0.0345 recall:0.0563
-0.0004068 0.0430211 -5.8e-06 0.0108632 0.0001815 0.0077932 0.0002191 0.005577 -3e-06 0.0168137
-3.54e-05 0.0245263 0.0001864 0.0110886 0.0001469 0.0059142 0.0002018 0.005085 0.0001249 0.0116535
epoch:177time:116.82     test hit:0.0564 ndcg:0.0343 recall:0.056
-0.0004049 0.0430227 -1.05e-05 0.0108597 0.0001817 0.0077893 0.0002144 0.0055725 -4.8e-06 0.0168111
-4e-05 0.0245249 0.0001888 0.0110865 0.0001435 0.0059107 0.0002031 0.0050814 0.0001239 0.0116509
epoch:178time:116.34     test hit:0.0561 ndcg:0.0343 recall:0.0558
-0.0004028 0.043033 -1e-05 0.0108564 0.0001832 0.0077858 0.000214 0.0055682 -3.9e-06 0.0168109
-3.67e-05 0.0245278 0.0001922 0.0110852 0.0001451 0.005907 0.0002051 0.0050782 0.0001264 0.0116495
epoch:179time:116.62     test hit:0.0561 ndcg:0.0343 recall:0.0558
-0.0004092 0.0430595 -5.7e-06 0.0108641 0.0001791 0.0077983 0.0002176 0.0055765 -4.6e-06 0.0168247
-3.56e-05 0.0245385 0.0001872 0.0110964 0.0001476 0.0059135 0.0002011 0.0050885 0.0001251 0.0116592
epoch:180time:116.72     test hit:0.0563 ndcg:0.0345 recall:0.056
-0.0004122 0.0430412 -1e-05 0.0108583 0.000177 0.0077885 0.0002133 0.0055734 -8e-06 0.0168154
-3.85e-05 0.0245357 0.0001837 0.0110855 0.0001434 0.0059101 0.0001988 0.0050821 0.0001218 0.0116534
epoch:181time:116.86     test hit:0.0567 ndcg:0.0345 recall:0.0564
-0.0003984 0.0430404 9e-07 0.0108533 0.0001932 0.0077806 0.0002249 0.0055677 5.1e-06 0.0168106
-2.48e-05 0.0245385 0.0001957 0.0110796 0.0001525 0.0059053 0.000211 0.0050757 0.0001336 0.0116498
epoch:182time:116.49     test hit:0.0564 ndcg:0.0343 recall:0.0561
-0.0004111 0.043071 1.1e-06 0.0108638 0.0001824 0.0077968 0.0002248 0.0055772 -7e-07 0.0168272
-1.91e-05 0.0245489 0.000189 0.0110948 0.0001532 0.0059136 0.0002029 0.0050883 0.0001315 0.0116614
epoch:183time:116.34     test hit:0.0568 ndcg:0.0346 recall:0.0564
-0.0004117 0.0430703 1.4e-06 0.0108625 0.0001804 0.0077931 0.000225 0.0055771 -1.2e-06 0.0168258
-1.61e-05 0.0245467 0.0001905 0.0110906 0.0001542 0.0059133 0.0002031 0.005086 0.0001329 0.0116591
epoch:184time:116.38     test hit:0.0565 ndcg:0.0345 recall:0.0562
-0.0004108 0.0430756 -1e-06 0.0108562 0.0001822 0.0077842 0.0002231 0.0055714 -1.7e-06 0.0168219
-1.38e-05 0.0245433 0.0001947 0.0110834 0.0001543 0.0059081 0.0002058 0.005079 0.0001352 0.0116534
epoch:185time:116.39     test hit:0.0566 ndcg:0.0345 recall:0.0562
-0.0004296 0.0430901 -1.52e-05 0.0108545 0.0001675 0.007779 0.0002086 0.0055663 -1.72e-05 0.0168225
-2.25e-05 0.0245499 0.0001842 0.0110821 0.0001431 0.0059045 0.0001942 0.0050739 0.0001247 0.0116526
epoch:186time:116.59     test hit:0.0567 ndcg:0.0346 recall:0.0564
-0.0004292 0.0431311 -1.64e-05 0.010861 0.0001682 0.0077896 0.0002067 0.0055687 -1.77e-05 0.0168377
-2.05e-05 0.0245691 0.0001856 0.0110955 0.0001421 0.0059074 0.0001937 0.0050811 0.0001252 0.0116633
epoch:187time:116.52     test hit:0.0566 ndcg:0.0345 recall:0.0563
-0.0004239 0.0431365 -1.33e-05 0.0108578 0.0001745 0.0077846 0.00021 0.0055663 -1.32e-05 0.0168364
-1.41e-05 0.024573 0.0001941 0.0110909 0.0001455 0.0059046 0.0001998 0.0050779 0.0001313 0.0116616
epoch:188time:116.42     test hit:0.0566 ndcg:0.0344 recall:0.0562
-0.0004241 0.0431411 -1.55e-05 0.0108621 0.0001727 0.007788 0.0002077 0.005572 -1.48e-05 0.0168409
-1.48e-05 0.024574 0.0001931 0.0110927 0.0001433 0.0059091 0.000198 0.0050812 0.0001299 0.0116642
epoch:189time:116.76     test hit:0.0564 ndcg:0.0344 recall:0.056
-0.0004297 0.0431467 -2.26e-05 0.0108661 0.0001655 0.0077921 0.0002009 0.0055764 -2.15e-05 0.0168454
-2.57e-05 0.0245762 0.000184 0.0110962 0.000136 0.0059129 0.0001902 0.0050846 0.0001211 0.0116675
epoch:190time:116.55     test hit:0.0564 ndcg:0.0344 recall:0.056
-0.0004247 0.0431309 -1.9e-05 0.0108569 0.0001679 0.0077844 0.0002052 0.0055697 -1.77e-05 0.0168355
-2.34e-05 0.0245661 0.0001858 0.0110882 0.0001391 0.0059064 0.0001921 0.0050787 0.0001234 0.0116599
epoch:191time:116.6      test hit:0.0565 ndcg:0.0345 recall:0.0562
-0.0004298 0.0431297 -1.76e-05 0.0108507 0.000166 0.0077773 0.0002076 0.0055641 -1.85e-05 0.0168305
-2.27e-05 0.0245664 0.0001871 0.0110821 0.0001425 0.0059014 0.0001935 0.0050736 0.0001251 0.0116559
epoch:192time:116.58     test hit:0.0564 ndcg:0.0345 recall:0.056
-0.000421 0.0431681 -1.11e-05 0.0108621 0.0001771 0.0077915 0.0002147 0.005573 -1.01e-05 0.0168487
-1.38e-05 0.0245823 0.000197 0.0110968 0.0001496 0.0059094 0.0002031 0.0050842 0.000134 0.0116682
epoch:193time:116.62     test hit:0.0567 ndcg:0.0344 recall:0.0563
-0.0004158 0.0431706 -4.4e-06 0.0108603 0.0001845 0.0077845 0.0002215 0.0055703 -3.5e-06 0.0168465
-7.8e-06 0.0245842 0.0002046 0.0110919 0.0001553 0.005907 0.0002102 0.0050786 0.0001406 0.0116654
epoch:194time:116.72     test hit:0.0569 ndcg:0.0345 recall:0.0565
-0.0004256 0.0431647 -1.04e-05 0.0108577 0.0001754 0.0077801 0.000216 0.0055681 -1.11e-05 0.0168427
-1.53e-05 0.0245854 0.000196 0.0110882 0.0001493 0.005905 0.0002021 0.0050755 0.000133 0.0116635
epoch:195time:116.69     test hit:0.0562 ndcg:0.0344 recall:0.0558
-0.0004332 0.0431746 -2.01e-05 0.0108585 0.0001666 0.0077823 0.0002063 0.0055676 -2.01e-05 0.0168458
-2.68e-05 0.0245923 0.0001877 0.0110914 0.000142 0.005905 0.0001949 0.0050769 0.0001244 0.0116664
epoch:196time:116.47     test hit:0.0563 ndcg:0.0343 recall:0.0559
-0.0004247 0.0431847 -2.19e-05 0.0108557 0.0001713 0.0077818 0.0002036 0.0055641 -1.79e-05 0.0168466
-2.93e-05 0.0245955 0.0001914 0.0110922 0.00014 0.0059019 0.0001983 0.0050763 0.0001251 0.0116665
epoch:197time:115.97     test hit:0.0562 ndcg:0.0344 recall:0.0559
-0.0004205 0.0432035 -1.46e-05 0.0108579 0.0001783 0.0077849 0.0002109 0.005565 -1.15e-05 0.0168529
-1.79e-05 0.0246056 0.0001973 0.0110967 0.0001474 0.0059027 0.0002039 0.0050782 0.0001327 0.0116708
epoch:198time:116.18     test hit:0.0562 ndcg:0.0344 recall:0.0558
-0.0004238 0.0432027 -8e-07 0.0108573 0.0001821 0.0077758 0.0002249 0.0055643 -4.4e-06 0.0168501
7e-07 0.0246088 0.0002033 0.0110884 0.0001605 0.005902 0.0002094 0.0050716 0.0001435 0.0116677
epoch:199time:116.04     test hit:0.0561 ndcg:0.0343 recall:0.0558
```
