# LR-GCF_gowalla_x0

A notebook to benchmark LR-GCCF on gowalla_x0 dataset.

Author: Yi Li, Tsinghua University

Edited by [XUEPAI Team](https://github.com/xue-pai)

### Index

[Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results)| [Logs](#Logs) 

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

We directly use the data format transformation file `data2npy.py`（Here, we rename the file to `LR-GCCF_data_process.py`.） provided by LR-GCCF to convert the data from the LightGCN repo into the format required by the program.

You need to put the downloaded files `train.txt` and `test.txt` into the data/Gowalla/gowalla_x0 directory. 

### Code

1. The benchmark is implemented based on the original LR-GCCF code released by the authors at: https://github.com/newlei/LR-GCCF/. We use the version with commit hash: 17c160a.
2. We added the calculation of the recall metric to the hr_ndcg function of the `evaluate.py` file.

3. Download the dataset from [LightGCN repo](https://github.com/kuandeng/LightGCN/tree/master/Data/gowalla) and run the preprocessing script for format transformation.

   ```python
   cd data/Gowalla/gowalla_x0
   python LR-GCCF_data_process.py
   cd benchmarks/LR-GCCF
   ```

4. Run the following script to reproduce the result.

   ```python
   python LR_GCCF.py --gpu_id 1 --dataset gowalla --run_id s0 --embed_size 64 --epoch 350 --lr 0.005
   ```

### Results

```python
HR@20 = 0.1555, Recall@20 = 0.1519, NDCG@20 = 0.1285
```

### Logs

#### train

```shell
epoch:0 time:22.2	 train loss:0.6933=0.6931+ val loss:0.6928 test loss:0.6928
epoch:1 time:21.7	 train loss:0.6919=0.6916+ val loss:0.6847 test loss:0.6856
epoch:2 time:22.4	 train loss:0.6777=0.6763+ val loss:0.6427 test loss:0.6475
epoch:3 time:23.0	 train loss:0.6256=0.6195+ val loss:0.5509 test loss:0.5618
epoch:4 time:21.4	 train loss:0.5313=0.514+ val loss:0.436 test loss:0.4487
epoch:5 time:22.7	 train loss:0.4303=0.3936+ val loss:0.3404 test loss:0.3497
epoch:6 time:21.7	 train loss:0.3603=0.2975+ val loss:0.2783 test loss:0.2838
epoch:7 time:23.9	 train loss:0.3248=0.2347+ val loss:0.2413 test loss:0.244
epoch:8 time:21.9	 train loss:0.3084=0.1961+ val loss:0.2167 test loss:0.2194
epoch:9 time:22.6	 train loss:0.2967=0.1705+ val loss:0.2012 test loss:0.2041
epoch:10 time:23.0	 train loss:0.2862=0.1539+ val loss:0.1911 test loss:0.1952
epoch:11 time:23.6	 train loss:0.2763=0.1439+ val loss:0.1864 test loss:0.1912
epoch:12 time:21.6	 train loss:0.2677=0.1388+ val loss:0.185 test loss:0.1937
epoch:13 time:20.1	 train loss:0.2605=0.1365+ val loss:0.186 test loss:0.195
epoch:14 time:23.4	 train loss:0.2555=0.1363+ val loss:0.187 test loss:0.1965
epoch:15 time:20.9	 train loss:0.2515=0.1362+ val loss:0.1876 test loss:0.1985
epoch:16 time:23.5	 train loss:0.2482=0.1352+ val loss:0.1875 test loss:0.1989
epoch:17 time:20.3	 train loss:0.2453=0.1332+ val loss:0.1853 test loss:0.1964
epoch:18 time:21.2	 train loss:0.2429=0.1303+ val loss:0.1829 test loss:0.1936
epoch:19 time:20.8	 train loss:0.2404=0.1267+ val loss:0.1814 test loss:0.1903
epoch:20 time:22.4	 train loss:0.2387=0.1235+ val loss:0.1788 test loss:0.1884
epoch:21 time:22.8	 train loss:0.2374=0.1207+ val loss:0.1766 test loss:0.1836
epoch:22 time:22.0	 train loss:0.2356=0.1178+ val loss:0.1752 test loss:0.1829
epoch:23 time:22.0	 train loss:0.2341=0.1156+ val loss:0.1749 test loss:0.183
epoch:24 time:23.1	 train loss:0.2333=0.1142+ val loss:0.1746 test loss:0.1814
epoch:25 time:21.8	 train loss:0.2323=0.1129+ val loss:0.1738 test loss:0.1832
epoch:26 time:22.9	 train loss:0.2312=0.1116+ val loss:0.1731 test loss:0.1809
epoch:27 time:22.8	 train loss:0.23=0.1101+ val loss:0.1729 test loss:0.1814
epoch:28 time:22.4	 train loss:0.2294=0.1091+ val loss:0.1727 test loss:0.1808
epoch:29 time:22.5	 train loss:0.2287=0.108+ val loss:0.1722 test loss:0.1794
epoch:30 time:20.8	 train loss:0.2279=0.1067+ val loss:0.1717 test loss:0.1795
epoch:31 time:21.4	 train loss:0.2271=0.1054+ val loss:0.1719 test loss:0.1803
epoch:32 time:23.0	 train loss:0.2263=0.1041+ val loss:0.171 test loss:0.1789
epoch:33 time:22.4	 train loss:0.2259=0.1034+ val loss:0.1705 test loss:0.178
epoch:34 time:22.3	 train loss:0.2254=0.1025+ val loss:0.1704 test loss:0.1785
epoch:35 time:24.2	 train loss:0.2248=0.1016+ val loss:0.1704 test loss:0.1785
epoch:36 time:21.4	 train loss:0.2244=0.1009+ val loss:0.1705 test loss:0.1791
epoch:37 time:22.8	 train loss:0.2239=0.1+ val loss:0.1703 test loss:0.1772
epoch:38 time:22.2	 train loss:0.2235=0.0993+ val loss:0.1697 test loss:0.1771
epoch:39 time:21.0	 train loss:0.223=0.0985+ val loss:0.1701 test loss:0.1773
epoch:40 time:21.4	 train loss:0.2227=0.0979+ val loss:0.1703 test loss:0.1784
epoch:41 time:21.4	 train loss:0.2222=0.0971+ val loss:0.1695 test loss:0.1771
epoch:42 time:23.6	 train loss:0.222=0.0967+ val loss:0.1698 test loss:0.177
epoch:43 time:22.4	 train loss:0.2216=0.096+ val loss:0.1698 test loss:0.1773
epoch:44 time:21.8	 train loss:0.2212=0.0954+ val loss:0.1692 test loss:0.1782
epoch:45 time:22.2	 train loss:0.221=0.0948+ val loss:0.1702 test loss:0.1772
epoch:46 time:22.9	 train loss:0.2208=0.0945+ val loss:0.1699 test loss:0.1773
epoch:47 time:20.2	 train loss:0.2204=0.0938+ val loss:0.1696 test loss:0.1772
epoch:48 time:22.4	 train loss:0.2201=0.0933+ val loss:0.1691 test loss:0.178
epoch:49 time:20.9	 train loss:0.2199=0.0928+ val loss:0.1696 test loss:0.1769
epoch:50 time:22.3	 train loss:0.2197=0.0925+ val loss:0.169 test loss:0.1769
epoch:51 time:23.8	 train loss:0.2195=0.092+ val loss:0.1691 test loss:0.1772
epoch:52 time:21.6	 train loss:0.2193=0.0918+ val loss:0.1693 test loss:0.1773
epoch:53 time:21.8	 train loss:0.2192=0.0914+ val loss:0.1696 test loss:0.1774
epoch:54 time:20.3	 train loss:0.2187=0.0908+ val loss:0.1691 test loss:0.1774
epoch:55 time:21.9	 train loss:0.2185=0.0904+ val loss:0.1694 test loss:0.1761
epoch:56 time:22.6	 train loss:0.2185=0.0901+ val loss:0.1692 test loss:0.1765
epoch:57 time:21.7	 train loss:0.2182=0.0897+ val loss:0.1692 test loss:0.1769
epoch:58 time:21.7	 train loss:0.218=0.0893+ val loss:0.1688 test loss:0.1784
epoch:59 time:21.1	 train loss:0.218=0.0892+ val loss:0.1696 test loss:0.1781
epoch:60 time:22.3	 train loss:0.2178=0.0888+ val loss:0.1698 test loss:0.1773
epoch:61 time:21.1	 train loss:0.2176=0.0885+ val loss:0.1698 test loss:0.1774
epoch:62 time:24.1	 train loss:0.2173=0.0881+ val loss:0.1697 test loss:0.1772
epoch:63 time:22.9	 train loss:0.2173=0.088+ val loss:0.1692 test loss:0.1781
epoch:64 time:20.7	 train loss:0.2171=0.0876+ val loss:0.1696 test loss:0.1771
epoch:65 time:22.1	 train loss:0.2169=0.0872+ val loss:0.1697 test loss:0.1773
epoch:66 time:20.5	 train loss:0.2168=0.087+ val loss:0.1693 test loss:0.1767
epoch:67 time:21.4	 train loss:0.2168=0.0868+ val loss:0.1693 test loss:0.1774
epoch:68 time:22.9	 train loss:0.2168=0.0866+ val loss:0.1695 test loss:0.1774
epoch:69 time:22.2	 train loss:0.2165=0.0862+ val loss:0.1697 test loss:0.1777
epoch:70 time:22.3	 train loss:0.2163=0.086+ val loss:0.1697 test loss:0.1781
epoch:71 time:21.4	 train loss:0.2162=0.0857+ val loss:0.1695 test loss:0.1766
epoch:72 time:21.0	 train loss:0.2159=0.0853+ val loss:0.1703 test loss:0.1776
epoch:73 time:22.5	 train loss:0.2161=0.0853+ val loss:0.1699 test loss:0.1794
epoch:74 time:23.0	 train loss:0.216=0.0851+ val loss:0.1693 test loss:0.1778
epoch:75 time:22.3	 train loss:0.2157=0.0848+ val loss:0.1695 test loss:0.1775
epoch:76 time:23.5	 train loss:0.216=0.0849+ val loss:0.1701 test loss:0.1785
epoch:77 time:22.1	 train loss:0.2157=0.0845+ val loss:0.1696 test loss:0.1785
epoch:78 time:23.7	 train loss:0.2156=0.0843+ val loss:0.1696 test loss:0.1782
epoch:79 time:23.6	 train loss:0.2153=0.084+ val loss:0.17 test loss:0.1783
epoch:80 time:21.3	 train loss:0.2155=0.084+ val loss:0.1706 test loss:0.1784
epoch:81 time:20.9	 train loss:0.2152=0.0836+ val loss:0.1701 test loss:0.1776
epoch:82 time:21.2	 train loss:0.215=0.0833+ val loss:0.1704 test loss:0.178
epoch:83 time:21.2	 train loss:0.2152=0.0834+ val loss:0.1702 test loss:0.1793
epoch:84 time:24.0	 train loss:0.215=0.0831+ val loss:0.1697 test loss:0.1781
epoch:85 time:22.7	 train loss:0.2152=0.0832+ val loss:0.1705 test loss:0.1779
epoch:86 time:22.5	 train loss:0.215=0.083+ val loss:0.1705 test loss:0.1781
epoch:87 time:22.6	 train loss:0.2148=0.0827+ val loss:0.1704 test loss:0.1784
epoch:88 time:23.7	 train loss:0.2146=0.0825+ val loss:0.1707 test loss:0.1786
epoch:89 time:22.6	 train loss:0.2147=0.0824+ val loss:0.1704 test loss:0.1787
epoch:90 time:24.7	 train loss:0.2146=0.0822+ val loss:0.1709 test loss:0.1789
epoch:91 time:21.5	 train loss:0.2146=0.0822+ val loss:0.1707 test loss:0.1782
epoch:92 time:23.7	 train loss:0.2146=0.082+ val loss:0.1709 test loss:0.1789
epoch:93 time:23.2	 train loss:0.2144=0.0819+ val loss:0.1707 test loss:0.179
epoch:94 time:22.1	 train loss:0.2144=0.0817+ val loss:0.1708 test loss:0.1795
epoch:95 time:22.0	 train loss:0.2142=0.0814+ val loss:0.1706 test loss:0.1792
epoch:96 time:22.0	 train loss:0.2143=0.0815+ val loss:0.1713 test loss:0.1797
epoch:97 time:21.2	 train loss:0.2142=0.0813+ val loss:0.1706 test loss:0.1795
epoch:98 time:23.6	 train loss:0.2143=0.0814+ val loss:0.171 test loss:0.179
epoch:99 time:21.2	 train loss:0.214=0.081+ val loss:0.1713 test loss:0.1796
epoch:100 time:21.6	 train loss:0.214=0.0809+ val loss:0.1709 test loss:0.179
epoch:101 time:22.3	 train loss:0.2142=0.081+ val loss:0.1713 test loss:0.1792
epoch:102 time:24.2	 train loss:0.214=0.0808+ val loss:0.171 test loss:0.1791
epoch:103 time:21.2	 train loss:0.2139=0.0807+ val loss:0.1708 test loss:0.1792
epoch:104 time:22.2	 train loss:0.2139=0.0806+ val loss:0.1711 test loss:0.1802
epoch:105 time:24.4	 train loss:0.2137=0.0804+ val loss:0.1711 test loss:0.1803
epoch:106 time:21.3	 train loss:0.2138=0.0804+ val loss:0.1714 test loss:0.1803
epoch:107 time:23.7	 train loss:0.2138=0.0803+ val loss:0.1712 test loss:0.179
epoch:108 time:23.0	 train loss:0.2137=0.0801+ val loss:0.1715 test loss:0.1797
epoch:109 time:19.9	 train loss:0.2135=0.0799+ val loss:0.1712 test loss:0.1794
epoch:110 time:21.2	 train loss:0.2137=0.0801+ val loss:0.1713 test loss:0.1802
epoch:111 time:22.6	 train loss:0.2136=0.0799+ val loss:0.1707 test loss:0.1799
epoch:112 time:22.9	 train loss:0.2136=0.0798+ val loss:0.1717 test loss:0.1806
epoch:113 time:21.4	 train loss:0.2136=0.0798+ val loss:0.1717 test loss:0.1805
epoch:114 time:22.3	 train loss:0.2136=0.0798+ val loss:0.1713 test loss:0.1798
epoch:115 time:21.0	 train loss:0.2133=0.0795+ val loss:0.1715 test loss:0.1798
epoch:116 time:20.9	 train loss:0.2135=0.0796+ val loss:0.1715 test loss:0.1804
epoch:117 time:22.8	 train loss:0.2135=0.0795+ val loss:0.1717 test loss:0.1794
epoch:118 time:21.3	 train loss:0.2134=0.0793+ val loss:0.1717 test loss:0.18
epoch:119 time:23.1	 train loss:0.2135=0.0794+ val loss:0.1714 test loss:0.1796
epoch:120 time:24.0	 train loss:0.2134=0.0793+ val loss:0.1719 test loss:0.1794
epoch:121 time:22.9	 train loss:0.2133=0.0792+ val loss:0.1722 test loss:0.1802
epoch:122 time:21.1	 train loss:0.2133=0.0791+ val loss:0.1713 test loss:0.1803
epoch:123 time:23.5	 train loss:0.2132=0.079+ val loss:0.1712 test loss:0.1796
epoch:124 time:20.5	 train loss:0.2131=0.0788+ val loss:0.1721 test loss:0.1799
epoch:125 time:22.8	 train loss:0.213=0.0787+ val loss:0.1715 test loss:0.1805
epoch:126 time:22.8	 train loss:0.2131=0.0788+ val loss:0.1719 test loss:0.1806
epoch:127 time:21.0	 train loss:0.2132=0.0788+ val loss:0.1721 test loss:0.1804
epoch:128 time:23.4	 train loss:0.2129=0.0786+ val loss:0.1718 test loss:0.1802
epoch:129 time:23.0	 train loss:0.2131=0.0787+ val loss:0.1721 test loss:0.1803
epoch:130 time:22.8	 train loss:0.2129=0.0785+ val loss:0.1718 test loss:0.1803
epoch:131 time:21.3	 train loss:0.2131=0.0786+ val loss:0.1728 test loss:0.1799
epoch:132 time:20.7	 train loss:0.213=0.0785+ val loss:0.1721 test loss:0.1807
epoch:133 time:22.5	 train loss:0.2128=0.0782+ val loss:0.1725 test loss:0.1813
epoch:134 time:22.7	 train loss:0.213=0.0783+ val loss:0.1725 test loss:0.1807
epoch:135 time:22.4	 train loss:0.2128=0.0781+ val loss:0.1721 test loss:0.1804
epoch:136 time:21.3	 train loss:0.2128=0.0781+ val loss:0.1725 test loss:0.1811
epoch:137 time:22.3	 train loss:0.213=0.0783+ val loss:0.1722 test loss:0.1807
epoch:138 time:22.3	 train loss:0.2126=0.0779+ val loss:0.1726 test loss:0.1814
epoch:139 time:21.2	 train loss:0.2128=0.078+ val loss:0.1725 test loss:0.1809
epoch:140 time:21.0	 train loss:0.2129=0.0781+ val loss:0.1725 test loss:0.1809
epoch:141 time:25.6	 train loss:0.2128=0.078+ val loss:0.1723 test loss:0.1821
epoch:142 time:23.7	 train loss:0.2128=0.078+ val loss:0.1729 test loss:0.1814
epoch:143 time:22.3	 train loss:0.2127=0.0778+ val loss:0.1725 test loss:0.1806
epoch:144 time:20.9	 train loss:0.2128=0.0779+ val loss:0.1723 test loss:0.181
epoch:145 time:21.0	 train loss:0.2127=0.0778+ val loss:0.1723 test loss:0.181
epoch:146 time:21.0	 train loss:0.2128=0.0779+ val loss:0.1725 test loss:0.1807
epoch:147 time:21.5	 train loss:0.2128=0.0778+ val loss:0.1726 test loss:0.1812
epoch:148 time:24.9	 train loss:0.2126=0.0777+ val loss:0.1726 test loss:0.1814
epoch:149 time:22.6	 train loss:0.2126=0.0776+ val loss:0.1727 test loss:0.1811
epoch:150 time:21.8	 train loss:0.2125=0.0775+ val loss:0.1725 test loss:0.181
epoch:151 time:21.2	 train loss:0.2126=0.0775+ val loss:0.1728 test loss:0.1803
epoch:152 time:22.5	 train loss:0.2126=0.0775+ val loss:0.1727 test loss:0.1813
epoch:153 time:21.2	 train loss:0.2125=0.0774+ val loss:0.1723 test loss:0.1813
epoch:154 time:23.2	 train loss:0.2125=0.0774+ val loss:0.1728 test loss:0.181
epoch:155 time:20.6	 train loss:0.2126=0.0774+ val loss:0.1725 test loss:0.1814
epoch:156 time:22.2	 train loss:0.2125=0.0773+ val loss:0.1729 test loss:0.1801
epoch:157 time:22.1	 train loss:0.2125=0.0772+ val loss:0.1724 test loss:0.1808
epoch:158 time:22.0	 train loss:0.2125=0.0772+ val loss:0.1728 test loss:0.1817
epoch:159 time:21.9	 train loss:0.2126=0.0774+ val loss:0.1733 test loss:0.1802
epoch:160 time:22.3	 train loss:0.2125=0.0772+ val loss:0.1729 test loss:0.1815
epoch:161 time:21.3	 train loss:0.2123=0.0771+ val loss:0.173 test loss:0.1812
epoch:162 time:22.3	 train loss:0.2125=0.0771+ val loss:0.1731 test loss:0.1807
epoch:163 time:21.0	 train loss:0.2125=0.0772+ val loss:0.1722 test loss:0.1818
epoch:164 time:22.9	 train loss:0.2125=0.0771+ val loss:0.1728 test loss:0.1814
epoch:165 time:22.5	 train loss:0.2124=0.077+ val loss:0.1731 test loss:0.182
epoch:166 time:19.8	 train loss:0.2124=0.077+ val loss:0.1731 test loss:0.1811
epoch:167 time:21.0	 train loss:0.2124=0.077+ val loss:0.1727 test loss:0.1813
epoch:168 time:19.5	 train loss:0.2126=0.0771+ val loss:0.1727 test loss:0.1818
epoch:169 time:20.9	 train loss:0.2126=0.0772+ val loss:0.1725 test loss:0.1816
epoch:170 time:21.5	 train loss:0.2123=0.0769+ val loss:0.173 test loss:0.1813
epoch:171 time:20.6	 train loss:0.2124=0.077+ val loss:0.1735 test loss:0.1809
epoch:172 time:22.0	 train loss:0.2124=0.0769+ val loss:0.1731 test loss:0.1822
epoch:173 time:22.9	 train loss:0.2124=0.0769+ val loss:0.1733 test loss:0.1814
epoch:174 time:20.9	 train loss:0.2123=0.0768+ val loss:0.1727 test loss:0.1805
epoch:175 time:22.2	 train loss:0.2122=0.0766+ val loss:0.173 test loss:0.1813
epoch:176 time:24.5	 train loss:0.2125=0.0769+ val loss:0.1727 test loss:0.1813
epoch:177 time:23.7	 train loss:0.2123=0.0767+ val loss:0.1737 test loss:0.1821
epoch:178 time:25.0	 train loss:0.2122=0.0767+ val loss:0.1733 test loss:0.1817
epoch:179 time:22.1	 train loss:0.2123=0.0767+ val loss:0.1729 test loss:0.181
epoch:180 time:22.4	 train loss:0.2123=0.0767+ val loss:0.1731 test loss:0.1809
epoch:181 time:22.8	 train loss:0.2123=0.0766+ val loss:0.1728 test loss:0.1813
epoch:182 time:21.6	 train loss:0.2123=0.0766+ val loss:0.1735 test loss:0.1825
epoch:183 time:24.6	 train loss:0.2123=0.0766+ val loss:0.1736 test loss:0.1816
epoch:184 time:20.9	 train loss:0.2123=0.0766+ val loss:0.1731 test loss:0.1813
epoch:185 time:23.7	 train loss:0.2122=0.0765+ val loss:0.1734 test loss:0.1817
epoch:186 time:21.4	 train loss:0.2123=0.0766+ val loss:0.1736 test loss:0.1816
epoch:187 time:24.1	 train loss:0.2122=0.0764+ val loss:0.1732 test loss:0.1818
epoch:188 time:22.4	 train loss:0.212=0.0762+ val loss:0.1732 test loss:0.1812
epoch:189 time:24.8	 train loss:0.2122=0.0764+ val loss:0.1735 test loss:0.1819
epoch:190 time:21.3	 train loss:0.2121=0.0763+ val loss:0.1734 test loss:0.1819
epoch:191 time:21.3	 train loss:0.212=0.0762+ val loss:0.1735 test loss:0.1819
epoch:192 time:21.3	 train loss:0.2122=0.0764+ val loss:0.1734 test loss:0.1823
epoch:193 time:22.1	 train loss:0.2122=0.0764+ val loss:0.1731 test loss:0.1815
epoch:194 time:20.9	 train loss:0.2121=0.0763+ val loss:0.1731 test loss:0.1818
epoch:195 time:22.3	 train loss:0.2122=0.0764+ val loss:0.1734 test loss:0.1825
epoch:196 time:22.8	 train loss:0.2121=0.0763+ val loss:0.1731 test loss:0.1822
epoch:197 time:21.2	 train loss:0.2123=0.0764+ val loss:0.1733 test loss:0.1818
epoch:198 time:21.7	 train loss:0.2121=0.0763+ val loss:0.1735 test loss:0.1824
epoch:199 time:23.7	 train loss:0.2121=0.0763+ val loss:0.1735 test loss:0.1826
epoch:200 time:22.8	 train loss:0.2122=0.0763+ val loss:0.1739 test loss:0.1828
epoch:201 time:21.0	 train loss:0.2122=0.0763+ val loss:0.1736 test loss:0.1813
epoch:202 time:22.5	 train loss:0.2121=0.0762+ val loss:0.1732 test loss:0.1816
epoch:203 time:21.0	 train loss:0.212=0.0761+ val loss:0.1735 test loss:0.1824
epoch:204 time:20.9	 train loss:0.212=0.0761+ val loss:0.1733 test loss:0.1825
epoch:205 time:23.3	 train loss:0.2121=0.0761+ val loss:0.1733 test loss:0.1824
epoch:206 time:23.6	 train loss:0.2121=0.0761+ val loss:0.1732 test loss:0.1826
epoch:207 time:24.0	 train loss:0.2121=0.0761+ val loss:0.1731 test loss:0.182
epoch:208 time:25.5	 train loss:0.2121=0.0761+ val loss:0.1736 test loss:0.1827
epoch:209 time:24.2	 train loss:0.212=0.076+ val loss:0.1733 test loss:0.1817
epoch:210 time:26.4	 train loss:0.212=0.076+ val loss:0.1734 test loss:0.1815
epoch:211 time:22.6	 train loss:0.212=0.0759+ val loss:0.1732 test loss:0.1815
epoch:212 time:22.0	 train loss:0.212=0.0759+ val loss:0.1732 test loss:0.1825
epoch:213 time:25.6	 train loss:0.2121=0.076+ val loss:0.1738 test loss:0.1817
epoch:214 time:22.6	 train loss:0.212=0.0759+ val loss:0.1735 test loss:0.1825
epoch:215 time:23.1	 train loss:0.2119=0.0758+ val loss:0.1736 test loss:0.183
epoch:216 time:22.5	 train loss:0.212=0.076+ val loss:0.173 test loss:0.1823
epoch:217 time:30.3	 train loss:0.212=0.076+ val loss:0.1738 test loss:0.182
epoch:218 time:25.5	 train loss:0.2121=0.076+ val loss:0.1739 test loss:0.1824
epoch:219 time:23.0	 train loss:0.2119=0.0758+ val loss:0.1731 test loss:0.1819
epoch:220 time:27.0	 train loss:0.212=0.0759+ val loss:0.1732 test loss:0.1829
epoch:221 time:22.7	 train loss:0.212=0.0758+ val loss:0.1737 test loss:0.1824
epoch:222 time:23.2	 train loss:0.2119=0.0757+ val loss:0.1732 test loss:0.1829
epoch:223 time:33.4	 train loss:0.212=0.0758+ val loss:0.1733 test loss:0.1818
epoch:224 time:22.5	 train loss:0.212=0.0759+ val loss:0.1739 test loss:0.1828
epoch:225 time:22.3	 train loss:0.212=0.0758+ val loss:0.1737 test loss:0.1826
epoch:226 time:22.3	 train loss:0.212=0.0758+ val loss:0.1741 test loss:0.183
epoch:227 time:22.4	 train loss:0.2119=0.0758+ val loss:0.174 test loss:0.182
epoch:228 time:24.6	 train loss:0.2121=0.0759+ val loss:0.1735 test loss:0.1825
epoch:229 time:22.1	 train loss:0.2118=0.0756+ val loss:0.1735 test loss:0.1812
epoch:230 time:22.3	 train loss:0.2119=0.0757+ val loss:0.174 test loss:0.1825
epoch:231 time:22.5	 train loss:0.2118=0.0756+ val loss:0.1738 test loss:0.1822
epoch:232 time:22.3	 train loss:0.2119=0.0757+ val loss:0.1734 test loss:0.1814
epoch:233 time:21.6	 train loss:0.2119=0.0757+ val loss:0.1737 test loss:0.1826
epoch:234 time:22.9	 train loss:0.2119=0.0757+ val loss:0.1737 test loss:0.1834
epoch:235 time:20.1	 train loss:0.212=0.0757+ val loss:0.1738 test loss:0.1827
epoch:236 time:21.2	 train loss:0.2119=0.0756+ val loss:0.1736 test loss:0.1829
epoch:237 time:23.3	 train loss:0.2119=0.0756+ val loss:0.1738 test loss:0.1826
epoch:238 time:21.2	 train loss:0.212=0.0757+ val loss:0.1735 test loss:0.1829
epoch:239 time:20.6	 train loss:0.2118=0.0755+ val loss:0.1741 test loss:0.1824
epoch:240 time:20.5	 train loss:0.2118=0.0755+ val loss:0.1739 test loss:0.1826
epoch:241 time:22.0	 train loss:0.2118=0.0755+ val loss:0.1736 test loss:0.183
epoch:242 time:19.6	 train loss:0.2119=0.0756+ val loss:0.1741 test loss:0.1826
epoch:243 time:22.5	 train loss:0.2119=0.0756+ val loss:0.1741 test loss:0.1831
epoch:244 time:21.8	 train loss:0.2119=0.0757+ val loss:0.1736 test loss:0.1834
epoch:245 time:22.4	 train loss:0.2118=0.0755+ val loss:0.174 test loss:0.1834
epoch:246 time:21.6	 train loss:0.212=0.0756+ val loss:0.1736 test loss:0.1824
epoch:247 time:23.5	 train loss:0.212=0.0756+ val loss:0.174 test loss:0.1829
epoch:248 time:21.8	 train loss:0.2118=0.0754+ val loss:0.174 test loss:0.1821
epoch:249 time:21.7	 train loss:0.2118=0.0755+ val loss:0.1739 test loss:0.1832
epoch:250 time:22.7	 train loss:0.2119=0.0756+ val loss:0.174 test loss:0.1827
epoch:251 time:23.0	 train loss:0.2117=0.0753+ val loss:0.1741 test loss:0.1823
epoch:252 time:21.8	 train loss:0.2118=0.0755+ val loss:0.1738 test loss:0.182
epoch:253 time:23.7	 train loss:0.212=0.0756+ val loss:0.174 test loss:0.1831
epoch:254 time:22.9	 train loss:0.2117=0.0753+ val loss:0.174 test loss:0.182
epoch:255 time:21.9	 train loss:0.2119=0.0756+ val loss:0.1738 test loss:0.1827
epoch:256 time:20.2	 train loss:0.2117=0.0754+ val loss:0.1733 test loss:0.1829
epoch:257 time:21.4	 train loss:0.2118=0.0754+ val loss:0.1739 test loss:0.1828
epoch:258 time:23.7	 train loss:0.2118=0.0754+ val loss:0.1739 test loss:0.182
epoch:259 time:22.8	 train loss:0.2119=0.0755+ val loss:0.1739 test loss:0.1821
epoch:260 time:21.7	 train loss:0.2118=0.0755+ val loss:0.1735 test loss:0.1829
epoch:261 time:21.0	 train loss:0.2118=0.0754+ val loss:0.1742 test loss:0.1823
epoch:262 time:23.0	 train loss:0.212=0.0755+ val loss:0.1738 test loss:0.1825
epoch:263 time:22.9	 train loss:0.2117=0.0753+ val loss:0.1741 test loss:0.1824
epoch:264 time:22.4	 train loss:0.2117=0.0753+ val loss:0.1739 test loss:0.1834
epoch:265 time:20.9	 train loss:0.2118=0.0754+ val loss:0.1743 test loss:0.1826
epoch:266 time:18.9	 train loss:0.2118=0.0753+ val loss:0.1739 test loss:0.1822
epoch:267 time:23.1	 train loss:0.2118=0.0753+ val loss:0.1738 test loss:0.1829
epoch:268 time:23.6	 train loss:0.2118=0.0754+ val loss:0.1733 test loss:0.1833
epoch:269 time:23.4	 train loss:0.2118=0.0753+ val loss:0.1739 test loss:0.1823
epoch:270 time:23.2	 train loss:0.2119=0.0754+ val loss:0.174 test loss:0.1823
epoch:271 time:22.6	 train loss:0.2117=0.0753+ val loss:0.1737 test loss:0.1828
epoch:272 time:22.7	 train loss:0.2116=0.0751+ val loss:0.1736 test loss:0.1826
epoch:273 time:23.6	 train loss:0.2117=0.0752+ val loss:0.1737 test loss:0.182
epoch:274 time:21.8	 train loss:0.2117=0.0752+ val loss:0.1739 test loss:0.1831
epoch:275 time:23.2	 train loss:0.2117=0.0752+ val loss:0.1738 test loss:0.1822
epoch:276 time:20.1	 train loss:0.2117=0.0753+ val loss:0.1738 test loss:0.182
epoch:277 time:20.8	 train loss:0.2117=0.0752+ val loss:0.1744 test loss:0.1825
epoch:278 time:20.9	 train loss:0.2117=0.0752+ val loss:0.174 test loss:0.1824
epoch:279 time:21.1	 train loss:0.2118=0.0752+ val loss:0.1741 test loss:0.1824
epoch:280 time:23.0	 train loss:0.2118=0.0753+ val loss:0.1742 test loss:0.1829
epoch:281 time:22.0	 train loss:0.2117=0.0752+ val loss:0.174 test loss:0.1831
epoch:282 time:23.1	 train loss:0.2116=0.0751+ val loss:0.1743 test loss:0.1827
epoch:283 time:26.2	 train loss:0.2117=0.0752+ val loss:0.1738 test loss:0.1821
epoch:284 time:22.2	 train loss:0.2118=0.0752+ val loss:0.1737 test loss:0.1829
epoch:285 time:22.3	 train loss:0.2118=0.0752+ val loss:0.1736 test loss:0.1819
epoch:286 time:22.4	 train loss:0.2118=0.0753+ val loss:0.1741 test loss:0.1825
epoch:287 time:21.2	 train loss:0.2117=0.0752+ val loss:0.1744 test loss:0.1817
epoch:288 time:20.9	 train loss:0.2117=0.0751+ val loss:0.1741 test loss:0.1832
epoch:289 time:20.8	 train loss:0.2119=0.0753+ val loss:0.1742 test loss:0.1829
epoch:290 time:21.9	 train loss:0.2117=0.0752+ val loss:0.1741 test loss:0.183
epoch:291 time:22.8	 train loss:0.2118=0.0752+ val loss:0.1735 test loss:0.1827
epoch:292 time:23.2	 train loss:0.2116=0.0751+ val loss:0.1742 test loss:0.1835
epoch:293 time:21.4	 train loss:0.2118=0.0752+ val loss:0.174 test loss:0.1823
epoch:294 time:23.7	 train loss:0.2117=0.0751+ val loss:0.174 test loss:0.1827
epoch:295 time:20.3	 train loss:0.2118=0.0752+ val loss:0.1738 test loss:0.1835
epoch:296 time:20.5	 train loss:0.2118=0.0752+ val loss:0.1743 test loss:0.1824
epoch:297 time:21.9	 train loss:0.2117=0.0751+ val loss:0.1739 test loss:0.183
epoch:298 time:21.5	 train loss:0.2117=0.0751+ val loss:0.1742 test loss:0.1826
epoch:299 time:26.2	 train loss:0.2119=0.0753+ val loss:0.1739 test loss:0.1826
epoch:300 time:33.2	 train loss:0.2117=0.0751+ val loss:0.1737 test loss:0.1834
epoch:301 time:36.5	 train loss:0.2116=0.0751+ val loss:0.1739 test loss:0.1829
epoch:302 time:35.3	 train loss:0.2117=0.0752+ val loss:0.1744 test loss:0.183
epoch:303 time:33.1	 train loss:0.2118=0.0751+ val loss:0.174 test loss:0.1831
epoch:304 time:40.6	 train loss:0.2117=0.075+ val loss:0.1742 test loss:0.1818
epoch:305 time:40.0	 train loss:0.2117=0.0751+ val loss:0.1738 test loss:0.1825
epoch:306 time:27.6	 train loss:0.2116=0.075+ val loss:0.1736 test loss:0.1829
epoch:307 time:20.9	 train loss:0.2117=0.075+ val loss:0.1741 test loss:0.1832
epoch:308 time:25.0	 train loss:0.2117=0.075+ val loss:0.1739 test loss:0.1828
epoch:309 time:23.0	 train loss:0.2116=0.075+ val loss:0.1739 test loss:0.1828
epoch:310 time:21.8	 train loss:0.2117=0.0751+ val loss:0.1738 test loss:0.1838
epoch:311 time:22.2	 train loss:0.2117=0.0751+ val loss:0.1739 test loss:0.1834
epoch:312 time:21.7	 train loss:0.2117=0.0751+ val loss:0.1742 test loss:0.1832
epoch:313 time:22.2	 train loss:0.2116=0.075+ val loss:0.1743 test loss:0.183
epoch:314 time:21.4	 train loss:0.2116=0.075+ val loss:0.1748 test loss:0.1829
epoch:315 time:21.5	 train loss:0.2117=0.075+ val loss:0.174 test loss:0.1834
epoch:316 time:27.8	 train loss:0.2118=0.0751+ val loss:0.1742 test loss:0.1834
epoch:317 time:22.1	 train loss:0.2116=0.075+ val loss:0.1739 test loss:0.1831
epoch:318 time:22.8	 train loss:0.2116=0.075+ val loss:0.1741 test loss:0.1835
epoch:319 time:23.2	 train loss:0.2118=0.0751+ val loss:0.1743 test loss:0.1823
epoch:320 time:22.0	 train loss:0.2117=0.075+ val loss:0.1742 test loss:0.1817
epoch:321 time:23.4	 train loss:0.2117=0.075+ val loss:0.1743 test loss:0.1828
epoch:322 time:26.3	 train loss:0.2116=0.0749+ val loss:0.1744 test loss:0.183
epoch:323 time:21.8	 train loss:0.2117=0.075+ val loss:0.174 test loss:0.1832
epoch:324 time:20.6	 train loss:0.2116=0.0749+ val loss:0.1744 test loss:0.1825
epoch:325 time:21.5	 train loss:0.2118=0.0751+ val loss:0.1737 test loss:0.1828
epoch:326 time:21.6	 train loss:0.2116=0.075+ val loss:0.1742 test loss:0.1827
epoch:327 time:21.8	 train loss:0.2118=0.0751+ val loss:0.1741 test loss:0.1826
epoch:328 time:22.1	 train loss:0.2117=0.0751+ val loss:0.1744 test loss:0.1835
epoch:329 time:20.9	 train loss:0.2116=0.075+ val loss:0.174 test loss:0.1833
epoch:330 time:22.1	 train loss:0.2117=0.0751+ val loss:0.1741 test loss:0.1832
epoch:331 time:22.0	 train loss:0.2118=0.0751+ val loss:0.1743 test loss:0.1832
epoch:332 time:22.0	 train loss:0.2116=0.0749+ val loss:0.174 test loss:0.1832
epoch:333 time:21.7	 train loss:0.2117=0.075+ val loss:0.1744 test loss:0.1829
epoch:334 time:22.0	 train loss:0.2118=0.0751+ val loss:0.1746 test loss:0.1832
epoch:335 time:24.4	 train loss:0.2118=0.0751+ val loss:0.1742 test loss:0.1839
epoch:336 time:22.0	 train loss:0.2116=0.0749+ val loss:0.1742 test loss:0.1826
epoch:337 time:21.9	 train loss:0.2116=0.0749+ val loss:0.1741 test loss:0.1835
epoch:338 time:20.7	 train loss:0.2117=0.075+ val loss:0.1738 test loss:0.1827
epoch:339 time:20.7	 train loss:0.2117=0.075+ val loss:0.1739 test loss:0.1828
epoch:340 time:21.0	 train loss:0.2116=0.0749+ val loss:0.1742 test loss:0.1833
epoch:341 time:21.0	 train loss:0.2116=0.0748+ val loss:0.1746 test loss:0.1825
epoch:342 time:22.8	 train loss:0.2118=0.075+ val loss:0.1742 test loss:0.1835
epoch:343 time:22.2	 train loss:0.2117=0.075+ val loss:0.1741 test loss:0.1832
epoch:344 time:20.2	 train loss:0.2116=0.0749+ val loss:0.1744 test loss:0.1832
epoch:345 time:23.4	 train loss:0.2116=0.0749+ val loss:0.1746 test loss:0.1827
epoch:346 time:22.3	 train loss:0.2116=0.0749+ val loss:0.1742 test loss:0.1832
epoch:347 time:22.8	 train loss:0.2116=0.0749+ val loss:0.1741 test loss:0.1833
epoch:348 time:21.0	 train loss:0.2117=0.075+ val loss:0.1739 test loss:0.184
epoch:349 time:22.1	 train loss:0.2117=0.075+ val loss:0.174 test loss:0.1836
```

#### test

```shell
s0 
has results save path
has model save path
--------test processing-------
0.0010034 0.0407954 0.0012253 0.0133548 0.0009533 0.0092796 0.0010888 0.0062299 0.0010677 0.017415
-0.0001184 0.0221085 -0.0001621 0.0113166 0.0002688 0.0063348 0.0002207 0.0049179 5.23e-05 0.0111695
epoch:130time:119.25     test hit:0.1517 ndcg:0.1257 recall:0.1481
0.0010075 0.0408407 0.0012295 0.0133566 0.0009584 0.0092764 0.001093 0.0062265 0.0010721 0.0174251
-0.0001167 0.0221256 -0.0001585 0.0113192 0.0002722 0.0063327 0.0002249 0.004915 5.55e-05 0.0111732
epoch:131time:115.75     test hit:0.1518 ndcg:0.1258 recall:0.1482
0.00101 0.0408847 0.001233 0.0133585 0.0009617 0.0092734 0.0010963 0.0062233 0.0010753 0.0174349
-0.0001155 0.0221419 -0.0001563 0.0113218 0.0002747 0.0063307 0.0002276 0.0049122 5.76e-05 0.0111767
epoch:132time:117.1      test hit:0.1519 ndcg:0.1259 recall:0.1484
0.0010101 0.0409309 0.0012357 0.0133624 0.000963 0.0092727 0.0010988 0.0062219 0.0010769 0.017447
-0.0001144 0.0221594 -0.0001556 0.0113264 0.0002768 0.0063303 0.0002288 0.004911 5.89e-05 0.0111818
epoch:133time:115.56     test hit:0.1521 ndcg:0.1259 recall:0.1485
0.0010071 0.0409738 0.0012348 0.013364 0.0009611 0.0092708 0.0010978 0.0062193 0.0010752 0.017457
-0.0001172 0.0221747 -0.0001574 0.0113295 0.0002759 0.0063286 0.0002275 0.0049091 5.72e-05 0.0111855
epoch:134time:116.77     test hit:0.1521 ndcg:0.126 recall:0.1486
0.001004 0.041016 0.0012332 0.013366 0.000959 0.0092695 0.0010965 0.0062171 0.0010732 0.0174671
-0.0001203 0.0221895 -0.0001595 0.011333 0.0002749 0.0063273 0.000226 0.0049077 5.53e-05 0.0111894
epoch:135time:115.62     test hit:0.1522 ndcg:0.126 recall:0.1486
0.0010022 0.0410577 0.0012325 0.0133685 0.0009579 0.0092684 0.001096 0.0062152 0.0010722 0.0174774
-0.000123 0.0222049 -0.0001609 0.0113367 0.0002745 0.0063262 0.0002254 0.0049063 5.4e-05 0.0111935
epoch:136time:116.63     test hit:0.1523 ndcg:0.126 recall:0.1487
0.0010021 0.0410972 0.0012322 0.0133694 0.0009582 0.0092659 0.0010962 0.0062119 0.0010722 0.0174861
-0.0001249 0.0222186 -0.000161 0.0113393 0.0002747 0.0063241 0.0002258 0.004904 5.36e-05 0.0111965
epoch:137time:116.18     test hit:0.1525 ndcg:0.1261 recall:0.149
0.0010041 0.0411361 0.0012336 0.0133702 0.0009602 0.0092631 0.0010982 0.0062085 0.001074 0.0174945
-0.0001249 0.022232 -0.0001596 0.0113415 0.0002764 0.0063219 0.0002278 0.0049014 5.5e-05 0.0111992
epoch:138time:116.11     test hit:0.1525 ndcg:0.1262 recall:0.1489
0.001006 0.0411751 0.0012352 0.0133704 0.0009628 0.0092594 0.0011004 0.0062044 0.0010761 0.0175023
-0.0001248 0.0222453 -0.0001575 0.0113431 0.0002784 0.0063191 0.0002304 0.0048983 5.66e-05 0.0112015
epoch:139time:115.66     test hit:0.1522 ndcg:0.1261 recall:0.1487
0.0010049 0.0412129 0.0012346 0.0133709 0.0009628 0.0092565 0.0011002 0.0062009 0.0010756 0.0175103
-0.0001275 0.0222588 -0.0001578 0.0113451 0.0002781 0.0063167 0.0002307 0.0048958 5.59e-05 0.0112041
epoch:140time:116.49     test hit:0.1524 ndcg:0.1262 recall:0.1488
0.0010067 0.0412505 0.0012347 0.0133721 0.0009652 0.009254 0.0011008 0.0061977 0.0010768 0.0175186
-0.00013 0.0222725 -0.0001565 0.0113474 0.0002779 0.0063147 0.0002326 0.0048934 5.6e-05 0.011207
epoch:141time:118.74     test hit:0.1524 ndcg:0.1262 recall:0.1488
0.0010103 0.0412916 0.001236 0.0133755 0.0009696 0.0092538 0.0011025 0.0061964 0.0010796 0.0175293
-0.0001306 0.022288 -0.0001533 0.0113516 0.0002789 0.0063143 0.0002361 0.0048926 5.78e-05 0.0112117
epoch:142time:122.34     test hit:0.1523 ndcg:0.1262 recall:0.1488
0.0010144 0.0413317 0.0012381 0.0133788 0.0009746 0.0092533 0.001105 0.006195 0.001083 0.0175397
-0.0001296 0.0223032 -0.0001493 0.0113557 0.0002809 0.0063138 0.0002402 0.0048916 6.06e-05 0.0112161
epoch:143time:118.16     test hit:0.1526 ndcg:0.1264 recall:0.1491
0.0010163 0.0413685 0.0012393 0.0133798 0.0009774 0.0092512 0.0011063 0.006192 0.0010848 0.0175479
-0.0001302 0.0223168 -0.0001474 0.0113583 0.0002817 0.0063118 0.0002424 0.0048895 6.16e-05 0.0112191
epoch:144time:118.23     test hit:0.1525 ndcg:0.1264 recall:0.149
0.0010145 0.0414044 0.0012382 0.013381 0.0009768 0.0092489 0.0011052 0.0061891 0.0010837 0.0175558
-0.0001328 0.022331 -0.0001484 0.0113607 0.0002803 0.0063099 0.0002416 0.0048874 6.02e-05 0.0112223
epoch:145time:120.86     test hit:0.1526 ndcg:0.1264 recall:0.149
0.0010085 0.0414395 0.0012337 0.0133822 0.0009721 0.0092459 0.0011006 0.0061857 0.0010787 0.0175633
-0.0001388 0.0223459 -0.0001529 0.0113627 0.0002758 0.0063078 0.0002375 0.0048846 5.54e-05 0.0112253
epoch:146time:117.55     test hit:0.1528 ndcg:0.1264 recall:0.1493
0.0010012 0.0414747 0.0012296 0.0133836 0.0009666 0.0092429 0.0010965 0.0061825 0.0010735 0.0175709
-0.0001433 0.022361 -0.0001577 0.0113648 0.0002722 0.006306 0.0002332 0.004882 5.11e-05 0.0112285
epoch:147time:121.63     test hit:0.1527 ndcg:0.1264 recall:0.1491
0.0009983 0.0415113 0.0012283 0.0133867 0.0009648 0.0092417 0.0010953 0.0061808 0.0010717 0.0175801
-0.000145 0.0223767 -0.0001594 0.0113682 0.0002712 0.0063053 0.0002321 0.0048805 4.97e-05 0.0112327
epoch:148time:118.29     test hit:0.1528 ndcg:0.1265 recall:0.1492
0.0010021 0.0415471 0.0012319 0.0133892 0.0009685 0.0092407 0.0010988 0.0061791 0.0010753 0.017589
-0.0001427 0.0223907 -0.0001565 0.0113714 0.0002745 0.0063044 0.0002356 0.0048792 5.27e-05 0.0112365
epoch:149time:121.26     test hit:0.1528 ndcg:0.1265 recall:0.1492
0.0010097 0.0415832 0.001238 0.0133925 0.0009756 0.0092407 0.0011049 0.0061784 0.001082 0.0175987
-0.000138 0.0224043 -0.0001507 0.0113753 0.0002801 0.0063043 0.0002422 0.0048786 5.84e-05 0.0112407
epoch:150time:117.69     test hit:0.1529 ndcg:0.1266 recall:0.1493
0.0010139 0.041614 0.0012423 0.0133937 0.0009804 0.0092388 0.0011093 0.0061766 0.0010865 0.0176058
-0.0001351 0.0224151 -0.0001473 0.0113771 0.0002837 0.0063031 0.0002463 0.0048771 6.19e-05 0.0112431
epoch:151time:118.15     test hit:0.153 ndcg:0.1267 recall:0.1494
0.0010104 0.0416412 0.0012412 0.0133934 0.0009784 0.0092358 0.0011082 0.0061737 0.0010845 0.017611
-0.0001379 0.0224247 -0.0001497 0.0113777 0.0002824 0.0063009 0.0002444 0.0048748 5.98e-05 0.0112446
epoch:152time:119.47     test hit:0.1532 ndcg:0.1267 recall:0.1496
0.0010046 0.0416712 0.0012378 0.0133945 0.000974 0.0092343 0.0011048 0.0061718 0.0010803 0.0176179
-0.0001424 0.0224361 -0.0001541 0.0113798 0.000279 0.0062996 0.0002405 0.0048734 5.57e-05 0.0112473
epoch:153time:121.19     test hit:0.153 ndcg:0.1268 recall:0.1495
0.0009983 0.0417002 0.0012332 0.0133949 0.000969 0.0092318 0.0011002 0.0061691 0.0010752 0.017624
-0.000148 0.022447 -0.0001587 0.0113812 0.0002748 0.0062978 0.0002363 0.0048713 5.11e-05 0.0112494
epoch:154time:117.81     test hit:0.1532 ndcg:0.1269 recall:0.1496
0.000996 0.0417307 0.0012309 0.0133965 0.0009679 0.00923 0.001098 0.0061669 0.0010732 0.017631
-0.0001515 0.0224588 -0.0001599 0.0113833 0.0002728 0.0062965 0.0002354 0.0048696 4.92e-05 0.0112521
epoch:155time:116.69     test hit:0.1535 ndcg:0.127 recall:0.1499
0.0009984 0.0417615 0.0012318 0.0133986 0.000971 0.009229 0.0010991 0.0061653 0.001075 0.0176386
-0.0001519 0.022471 -0.0001579 0.011386 0.0002734 0.0062957 0.0002379 0.0048684 5.04e-05 0.0112553
epoch:156time:117.11     test hit:0.1535 ndcg:0.1271 recall:0.1499
0.0010027 0.0417904 0.0012344 0.0133999 0.0009757 0.0092277 0.0011019 0.0061632 0.0010787 0.0176453
-0.0001517 0.0224824 -0.0001551 0.0113884 0.0002753 0.0062944 0.0002415 0.0048671 5.25e-05 0.0112581
epoch:157time:118.51     test hit:0.1535 ndcg:0.1271 recall:0.1499
0.0010026 0.0418179 0.0012329 0.013401 0.0009758 0.0092257 0.0011004 0.0061609 0.0010779 0.0176514
-0.0001555 0.0224928 -0.0001561 0.0113901 0.0002733 0.006293 0.000241 0.0048653 5.07e-05 0.0112603
epoch:158time:118.7      test hit:0.1534 ndcg:0.1272 recall:0.1498
0.0009999 0.0418449 0.0012302 0.0134014 0.0009738 0.0092236 0.0010978 0.0061582 0.0010754 0.017657
-0.0001603 0.0225028 -0.0001589 0.0113917 0.0002705 0.0062913 0.0002389 0.0048633 4.75e-05 0.0112623
epoch:159time:118.04     test hit:0.1535 ndcg:0.1272 recall:0.1499
0.0009986 0.0418739 0.0012289 0.0134025 0.0009725 0.0092221 0.0010968 0.0061563 0.0010742 0.0176637
-0.0001633 0.0225135 -0.000161 0.0113939 0.0002693 0.0062901 0.0002376 0.0048619 4.57e-05 0.0112649
epoch:160time:118.41     test hit:0.1536 ndcg:0.1273 recall:0.1501
0.0009992 0.0419038 0.0012294 0.0134053 0.000973 0.009222 0.0010977 0.0061557 0.0010748 0.0176717
-0.0001642 0.0225246 -0.0001615 0.0113971 0.0002699 0.00629 0.0002378 0.0048613 4.55e-05 0.0112683
epoch:161time:121.15     test hit:0.154 ndcg:0.1274 recall:0.1504
0.0010047 0.0419324 0.0012339 0.0134075 0.0009775 0.0092218 0.001102 0.0061549 0.0010795 0.0176791
-0.0001625 0.0225358 -0.0001588 0.0114 0.0002734 0.0062897 0.0002414 0.0048607 4.84e-05 0.0112716
epoch:162time:118.36     test hit:0.1538 ndcg:0.1273 recall:0.1502
0.0010097 0.0419591 0.0012382 0.0134091 0.0009821 0.0092207 0.0011061 0.0061534 0.001084 0.0176856
-0.0001611 0.0225461 -0.0001558 0.0114022 0.0002764 0.0062889 0.0002451 0.0048595 5.11e-05 0.0112742
epoch:163time:118.04     test hit:0.1538 ndcg:0.1273 recall:0.1502
0.0010143 0.0419844 0.001242 0.01341 0.0009866 0.0092191 0.0011097 0.0061512 0.0010882 0.0176912
-0.0001594 0.0225557 -0.0001528 0.0114039 0.0002792 0.0062875 0.0002486 0.004858 5.39e-05 0.0112763
epoch:164time:118.87     test hit:0.1536 ndcg:0.1272 recall:0.15
0.0010176 0.0420092 0.0012444 0.0134102 0.0009899 0.0092174 0.0011119 0.0061486 0.001091 0.0176964
-0.0001586 0.0225648 -0.0001505 0.0114057 0.0002808 0.0062858 0.0002512 0.0048564 5.57e-05 0.0112782
epoch:165time:119.29     test hit:0.1536 ndcg:0.1273 recall:0.15
0.0010181 0.0420333 0.0012452 0.0134109 0.0009908 0.0092159 0.0011124 0.0061467 0.0010916 0.0177017
-0.0001597 0.022574 -0.0001504 0.0114074 0.0002809 0.0062845 0.0002517 0.004855 5.56e-05 0.0112803
epoch:166time:117.92     test hit:0.1538 ndcg:0.1273 recall:0.1502
0.0010184 0.0420564 0.0012471 0.013411 0.0009922 0.0092141 0.0011144 0.0061446 0.001093 0.0177066
-0.000159 0.0225826 -0.0001498 0.0114086 0.0002823 0.006283 0.0002528 0.0048535 5.66e-05 0.011282
epoch:167time:118.47     test hit:0.1537 ndcg:0.1273 recall:0.1501
0.0010191 0.0420768 0.0012492 0.0134093 0.000994 0.0092107 0.0011166 0.0061413 0.0010947 0.0177095
-0.000158 0.0225905 -0.0001488 0.0114082 0.0002841 0.0062804 0.0002544 0.004851 5.79e-05 0.0112826
epoch:168time:118.76     test hit:0.1536 ndcg:0.1272 recall:0.15
0.0010166 0.0420959 0.0012492 0.0134073 0.0009928 0.0092063 0.0011169 0.0061376 0.0010939 0.0177118
-0.0001587 0.0225987 -0.0001505 0.0114071 0.0002843 0.0062775 0.0002535 0.004848 5.72e-05 0.0112829
epoch:169time:121.37     test hit:0.154 ndcg:0.1274 recall:0.1504
0.0010112 0.0421206 0.0012472 0.0134092 0.0009892 0.0092051 0.0011152 0.0061366 0.0010907 0.0177179
-0.0001612 0.0226097 -0.0001543 0.0114089 0.0002826 0.006277 0.0002503 0.0048469 5.44e-05 0.0112857
epoch:170time:118.12     test hit:0.1539 ndcg:0.1274 recall:0.1503
0.0010076 0.0421493 0.0012437 0.0134133 0.0009858 0.0092057 0.0011118 0.0061368 0.0010872 0.0177263
-0.0001668 0.0226228 -0.0001581 0.0114125 0.0002788 0.0062776 0.0002469 0.0048468 5.02e-05 0.01129
epoch:171time:118.45     test hit:0.1541 ndcg:0.1276 recall:0.1505
0.0010076 0.0421787 0.0012429 0.0134176 0.0009859 0.0092065 0.0011111 0.0061368 0.0010869 0.0177349
-0.0001699 0.0226364 -0.000159 0.0114165 0.0002776 0.0062783 0.0002464 0.0048468 4.88e-05 0.0112946
epoch:172time:118.92     test hit:0.1543 ndcg:0.1277 recall:0.1507
0.0010116 0.0422044 0.0012445 0.0134194 0.0009893 0.0092061 0.0011126 0.0061358 0.0010895 0.0177414
-0.0001712 0.0226465 -0.000157 0.011419 0.0002784 0.0062778 0.0002488 0.0048462 4.97e-05 0.0112974
epoch:173time:117.66     test hit:0.1542 ndcg:0.1277 recall:0.1506
0.0010147 0.0422293 0.0012461 0.0134208 0.0009918 0.0092059 0.001114 0.0061349 0.0010917 0.0177477
-0.0001714 0.0226556 -0.0001554 0.0114215 0.0002794 0.0062772 0.0002509 0.0048457 5.09e-05 0.0113
epoch:174time:116.18     test hit:0.1543 ndcg:0.1277 recall:0.1507
0.0010176 0.0422526 0.0012468 0.0134214 0.0009938 0.0092047 0.0011146 0.0061329 0.0010932 0.0177529
-0.0001729 0.0226646 -0.0001545 0.0114232 0.0002798 0.0062759 0.0002525 0.0048445 5.12e-05 0.0113021
epoch:175time:119.06     test hit:0.1543 ndcg:0.1276 recall:0.1507
0.001019 0.0422715 0.0012466 0.0134202 0.0009951 0.0092012 0.0011143 0.0061293 0.0010937 0.0177556
-0.0001748 0.022672 -0.0001538 0.011423 0.0002794 0.0062733 0.0002536 0.0048417 5.11e-05 0.0113026
epoch:176time:118.67     test hit:0.1545 ndcg:0.1278 recall:0.1509
0.0010215 0.0422913 0.001248 0.0134207 0.0009975 0.0091993 0.0011156 0.0061275 0.0010957 0.0177597
-0.0001739 0.0226797 -0.0001518 0.0114238 0.0002807 0.0062722 0.0002558 0.0048402 5.27e-05 0.011304
epoch:177time:119.44     test hit:0.1543 ndcg:0.1278 recall:0.1507
0.0010209 0.0423127 0.0012479 0.013422 0.0009977 0.0091984 0.0011153 0.0061264 0.0010955 0.0177649
-0.0001731 0.0226884 -0.0001515 0.0114256 0.0002808 0.0062716 0.0002562 0.0048392 5.31e-05 0.0113062
epoch:178time:119.48     test hit:0.1547 ndcg:0.128 recall:0.1511
0.0010218 0.0423339 0.001248 0.0134226 0.0009988 0.0091973 0.0011153 0.006125 0.001096 0.0177697
-0.000173 0.0226967 -0.0001509 0.0114271 0.0002808 0.0062706 0.000257 0.0048383 5.35e-05 0.0113082
epoch:179time:121.69     test hit:0.1548 ndcg:0.128 recall:0.1512
0.0010237 0.0423517 0.0012494 0.0134221 0.0010004 0.009195 0.0011166 0.0061226 0.0010975 0.0177729
-0.000172 0.0227033 -0.0001498 0.0114275 0.0002819 0.0062689 0.0002583 0.0048364 5.46e-05 0.0113091
epoch:180time:118.6      test hit:0.1547 ndcg:0.1279 recall:0.1511
0.0010229 0.0423701 0.0012484 0.0134216 0.0009996 0.0091931 0.0011156 0.0061201 0.0010966 0.0177762
-0.0001737 0.0227107 -0.0001508 0.0114282 0.0002808 0.0062671 0.0002577 0.0048349 5.35e-05 0.0113103
epoch:181time:121.21     test hit:0.1549 ndcg:0.128 recall:0.1513
0.001025 0.0423907 0.0012503 0.0134229 0.0010008 0.0091928 0.0011172 0.0061192 0.0010983 0.0177814
-0.0001734 0.0227191 -0.0001503 0.0114303 0.000282 0.0062666 0.0002587 0.0048344 5.42e-05 0.0113126
epoch:182time:119.42     test hit:0.1551 ndcg:0.128 recall:0.1515
0.0010275 0.0424128 0.0012521 0.0134247 0.0010024 0.0091931 0.0011186 0.0061187 0.0011002 0.0177873
-0.0001737 0.022728 -0.0001498 0.0114329 0.0002827 0.0062664 0.0002598 0.0048344 5.47e-05 0.0113155
epoch:183time:121.27     test hit:0.155 ndcg:0.128 recall:0.1514
0.0010304 0.0424318 0.0012547 0.0134244 0.0010047 0.0091909 0.001121 0.0061162 0.0011027 0.0177909
-0.0001727 0.0227363 -0.0001486 0.0114337 0.0002845 0.0062646 0.0002616 0.0048326 5.62e-05 0.0113168
epoch:184time:119.36     test hit:0.1548 ndcg:0.128 recall:0.1512
0.0010333 0.0424519 0.0012559 0.0134254 0.0010068 0.0091896 0.0011217 0.0061145 0.0011044 0.0177953
-0.0001734 0.0227452 -0.0001476 0.0114352 0.0002845 0.0062636 0.000263 0.0048312 5.67e-05 0.0113189
epoch:185time:121.27     test hit:0.1553 ndcg:0.1282 recall:0.1517
0.0010314 0.0424732 0.001253 0.0134276 0.0010048 0.0091887 0.0011182 0.0061133 0.0011018 0.0178007
-0.0001776 0.0227558 -0.00015 0.0114373 0.000281 0.0062632 0.0002608 0.0048301 5.35e-05 0.0113216
epoch:186time:119.33     test hit:0.155 ndcg:0.1281 recall:0.1514
0.001034 0.0424942 0.001254 0.0134305 0.0010067 0.0091888 0.0011189 0.0061133 0.0011034 0.0178067
-0.0001783 0.0227647 -0.0001491 0.0114398 0.0002812 0.0062636 0.000262 0.0048298 5.4e-05 0.0113245
epoch:187time:121.57     test hit:0.155 ndcg:0.1282 recall:0.1514
0.0010385 0.042515 0.0012581 0.0134339 0.0010107 0.0091901 0.0011229 0.0061143 0.0011076 0.0178133
-0.0001754 0.0227723 -0.0001464 0.0114429 0.0002845 0.0062646 0.0002651 0.0048303 5.69e-05 0.0113276
epoch:188time:119.47     test hit:0.155 ndcg:0.1283 recall:0.1514
0.0010414 0.0425308 0.0012642 0.0134343 0.0010135 0.0091883 0.0011287 0.0061127 0.0011119 0.0178165
-0.0001704 0.0227776 -0.0001444 0.0114433 0.0002897 0.0062636 0.0002676 0.0048289 6.06e-05 0.0113284
epoch:189time:121.49     test hit:0.1551 ndcg:0.1282 recall:0.1515
0.001041 0.042543 0.0012675 0.0134332 0.0010137 0.0091855 0.0011317 0.0061107 0.0011135 0.0178181
-0.0001689 0.0227815 -0.0001449 0.0114423 0.0002922 0.0062619 0.0002675 0.0048271 6.15e-05 0.0113283
epoch:190time:120.38     test hit:0.1552 ndcg:0.1283 recall:0.1515
0.0010406 0.0425584 0.0012693 0.0134334 0.0010132 0.0091846 0.0011328 0.0061097 0.001114 0.0178215
-0.0001702 0.0227868 -0.0001464 0.0114433 0.0002925 0.0062611 0.0002664 0.0048265 6.06e-05 0.0113295
epoch:191time:122.23     test hit:0.1551 ndcg:0.1283 recall:0.1515
0.0010442 0.042572 0.0012739 0.0134321 0.0010162 0.0091822 0.0011367 0.0061075 0.0011177 0.0178234
-0.0001684 0.0227909 -0.0001447 0.0114429 0.0002954 0.0062593 0.0002686 0.0048248 6.27e-05 0.0113295
epoch:192time:117.52     test hit:0.1551 ndcg:0.1282 recall:0.1515
0.0010482 0.0425875 0.0012777 0.0134319 0.0010193 0.0091804 0.00114 0.0061057 0.0011213 0.0178264
-0.0001671 0.0227966 -0.0001428 0.0114435 0.0002978 0.006258 0.0002711 0.0048234 6.47e-05 0.0113304
epoch:193time:119.47     test hit:0.1552 ndcg:0.1283 recall:0.1516
0.0010498 0.0426035 0.0012773 0.0134323 0.00102 0.0091793 0.0011393 0.0061043 0.0011216 0.0178298
-0.0001702 0.0228026 -0.0001432 0.0114445 0.0002968 0.0062571 0.0002714 0.0048223 6.37e-05 0.0113317
epoch:194time:117.67     test hit:0.1555 ndcg:0.1284 recall:0.1519
0.0010542 0.0426206 0.0012781 0.0134334 0.0010228 0.0091789 0.00114 0.0061032 0.0011238 0.017834
-0.0001729 0.0228094 -0.000142 0.0114463 0.0002968 0.0062566 0.0002735 0.0048217 6.38e-05 0.0113335
epoch:195time:120.05     test hit:0.1553 ndcg:0.1283 recall:0.1517
0.0010541 0.0426388 0.001275 0.0134359 0.0010217 0.0091793 0.0011368 0.0061033 0.0011219 0.0178393
-0.0001789 0.0228172 -0.000144 0.0114486 0.0002935 0.0062571 0.0002721 0.0048217 6.07e-05 0.0113362
epoch:196time:120.2      test hit:0.1552 ndcg:0.1283 recall:0.1516
0.0010508 0.0426526 0.0012708 0.0134352 0.0010184 0.0091766 0.0011324 0.0061009 0.0011181 0.0178414
-0.0001844 0.0228234 -0.0001471 0.0114482 0.0002898 0.0062554 0.0002693 0.0048196 5.69e-05 0.0113367
epoch:197time:121.76     test hit:0.1551 ndcg:0.1284 recall:0.1515
0.0010492 0.0426672 0.001271 0.0134345 0.0010176 0.0091744 0.0011325 0.0060987 0.0011176 0.0178437
-0.0001846 0.0228299 -0.0001473 0.0114483 0.0002903 0.0062537 0.000269 0.0048179 5.68e-05 0.0113375
epoch:198time:119.35     test hit:0.1553 ndcg:0.1285 recall:0.1517
0.0010517 0.0426853 0.0012732 0.0134359 0.0010203 0.0091747 0.0011348 0.0060984 0.00112 0.0178486
-0.0001833 0.0228371 -0.0001451 0.0114503 0.0002923 0.0062536 0.0002712 0.0048179 5.88e-05 0.0113398
epoch:199time:120.96     test hit:0.1555 ndcg:0.1285 recall:0.1519
```
