# LR-GCCF_amazonbooks_x0

A notebook to benchmark LR_GCCF on amazonbooks_x0 dataset.

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

We directly use the data format transformation file `data2npy.py` （Here, we rename this file to `LR-GCCF_data_process.py`.）provided by LR-GCCF to convert the data from the LightGCN repo into the format required by the program.

You need to put the downloaded files `train.txt` and `test.txt` into the data/AmazonBooks/amazonbooks_x0 directory. 

### Code

1. The benchmark is implemented based on the original LR-GCCF code released by the authors at: https://github.com/newlei/LR-GCCF/. We use the version with commit hash: 17c160a.

2. We added the calculation of the recall metric to the hr_ndcg function of the `evaluate.py` file.

3. Download the dataset from [LightGCN repo](https://github.com/kuandeng/LightGCN/tree/master/Data/amazon-book) and run the preprocessing script for format transformation.

   ```python
   cd data/AmazonBooks/amazonbooks_x0
   python LR-GCCF_data_process.py
   cd benchmarks/LR-GCCF
   ```

4. Run the following script to reproduce the result.

   ```python
   python LR_GCCF.py --gpu_id 3 --dataset amazons --run_id s0 --embed_size 64 --epoch 350 --lr 0.001
   ```

### Results

```python
HR@20 = 0.0349, Recall@20 = 0.0335, NDCG@20 = 0.0265
```

### Logs

#### train

```shell
epoch:0 time:70.9	 train loss:0.6932=0.6931+ test loss:0.6929
epoch:1 time:72.5	 train loss:0.6889=0.6883+ test loss:0.678
epoch:2 time:67.1	 train loss:0.6357=0.6301+ test loss:0.5879
epoch:3 time:67.1	 train loss:0.5158=0.4924+ test loss:0.4818
epoch:4 time:66.0	 train loss:0.433=0.3856+ test loss:0.4272
epoch:5 time:66.3	 train loss:0.3941=0.3295+ test loss:0.3972
epoch:6 time:70.0	 train loss:0.3693=0.2951+ test loss:0.3751
epoch:7 time:69.7	 train loss:0.3506=0.2704+ test loss:0.3581
epoch:8 time:67.9	 train loss:0.3359=0.2511+ test loss:0.3435
epoch:9 time:67.1	 train loss:0.3243=0.2357+ test loss:0.331
epoch:10 time:69.6	 train loss:0.3153=0.2234+ test loss:0.3212
epoch:11 time:66.3	 train loss:0.3075=0.2128+ test loss:0.3126
epoch:12 time:71.5	 train loss:0.3011=0.2038+ test loss:0.3053
epoch:13 time:69.5	 train loss:0.2959=0.1963+ test loss:0.2989
epoch:14 time:69.0	 train loss:0.2913=0.1898+ test loss:0.2934
epoch:15 time:68.9	 train loss:0.2872=0.1839+ test loss:0.2887
epoch:16 time:68.6	 train loss:0.2836=0.1787+ test loss:0.2841
epoch:17 time:68.1	 train loss:0.2806=0.1741+ test loss:0.2806
epoch:18 time:67.0	 train loss:0.2779=0.1702+ test loss:0.2772
epoch:19 time:68.3	 train loss:0.2754=0.1663+ test loss:0.2742
epoch:20 time:67.8	 train loss:0.2729=0.1627+ test loss:0.272
epoch:21 time:68.8	 train loss:0.2711=0.1597+ test loss:0.269
epoch:22 time:68.3	 train loss:0.2691=0.1568+ test loss:0.2673
epoch:23 time:69.8	 train loss:0.2673=0.154+ test loss:0.2656
epoch:24 time:67.6	 train loss:0.2657=0.1515+ test loss:0.2633
epoch:25 time:67.6	 train loss:0.2643=0.1492+ test loss:0.262
epoch:26 time:65.9	 train loss:0.2628=0.1469+ test loss:0.2601
epoch:27 time:69.8	 train loss:0.2615=0.1447+ test loss:0.259
epoch:28 time:69.6	 train loss:0.2602=0.1427+ test loss:0.2572
epoch:29 time:66.5	 train loss:0.2591=0.1408+ test loss:0.256
epoch:30 time:69.8	 train loss:0.258=0.139+ test loss:0.2551
epoch:31 time:66.3	 train loss:0.2569=0.1373+ test loss:0.2542
epoch:32 time:66.2	 train loss:0.2559=0.1356+ test loss:0.2531
epoch:33 time:69.9	 train loss:0.2551=0.1342+ test loss:0.2522
epoch:34 time:67.2	 train loss:0.2541=0.1326+ test loss:0.2516
epoch:35 time:69.4	 train loss:0.2534=0.1313+ test loss:0.2506
epoch:36 time:68.4	 train loss:0.2526=0.13+ test loss:0.2497
epoch:37 time:66.7	 train loss:0.2519=0.1288+ test loss:0.2492
epoch:38 time:68.8	 train loss:0.2511=0.1275+ test loss:0.2484
epoch:39 time:69.5	 train loss:0.2504=0.1263+ test loss:0.248
epoch:40 time:66.9	 train loss:0.2498=0.1252+ test loss:0.2473
epoch:41 time:66.9	 train loss:0.2492=0.1242+ test loss:0.2472
epoch:42 time:70.4	 train loss:0.2485=0.1231+ test loss:0.2465
epoch:43 time:67.8	 train loss:0.2481=0.1222+ test loss:0.2462
epoch:44 time:67.2	 train loss:0.2475=0.1212+ test loss:0.2454
epoch:45 time:69.6	 train loss:0.2471=0.1204+ test loss:0.245
epoch:46 time:69.2	 train loss:0.2466=0.1195+ test loss:0.245
epoch:47 time:68.0	 train loss:0.2461=0.1187+ test loss:0.2447
epoch:48 time:69.1	 train loss:0.2456=0.1178+ test loss:0.2446
epoch:49 time:68.2	 train loss:0.2452=0.117+ test loss:0.2437
epoch:50 time:67.6	 train loss:0.2448=0.1163+ test loss:0.2436
epoch:51 time:69.9	 train loss:0.2445=0.1157+ test loss:0.2433
epoch:52 time:67.6	 train loss:0.244=0.1149+ test loss:0.2431
epoch:53 time:70.4	 train loss:0.2438=0.1143+ test loss:0.2433
epoch:54 time:68.2	 train loss:0.2434=0.1137+ test loss:0.2433
epoch:55 time:67.7	 train loss:0.243=0.113+ test loss:0.2428
epoch:56 time:68.4	 train loss:0.2428=0.1125+ test loss:0.2427
epoch:57 time:67.8	 train loss:0.2425=0.1119+ test loss:0.2426
epoch:58 time:65.2	 train loss:0.2421=0.1113+ test loss:0.2426
epoch:59 time:70.0	 train loss:0.2419=0.1108+ test loss:0.242
epoch:60 time:67.4	 train loss:0.2415=0.1102+ test loss:0.2425
epoch:61 time:69.1	 train loss:0.2414=0.1098+ test loss:0.2424
epoch:62 time:67.1	 train loss:0.2411=0.1093+ test loss:0.242
epoch:63 time:67.3	 train loss:0.2408=0.1088+ test loss:0.2418
epoch:64 time:66.6	 train loss:0.2407=0.1084+ test loss:0.2417
epoch:65 time:69.6	 train loss:0.2405=0.108+ test loss:0.2419
epoch:66 time:66.4	 train loss:0.2402=0.1076+ test loss:0.2414
epoch:67 time:66.6	 train loss:0.2401=0.1072+ test loss:0.2415
epoch:68 time:70.1	 train loss:0.2399=0.1068+ test loss:0.2419
epoch:69 time:68.4	 train loss:0.2397=0.1064+ test loss:0.2418
epoch:70 time:68.2	 train loss:0.2395=0.1061+ test loss:0.2418
epoch:71 time:69.3	 train loss:0.2393=0.1057+ test loss:0.2421
epoch:72 time:69.9	 train loss:0.2391=0.1053+ test loss:0.2417
epoch:73 time:67.3	 train loss:0.239=0.105+ test loss:0.2416
epoch:74 time:67.1	 train loss:0.2388=0.1047+ test loss:0.2416
epoch:75 time:70.6	 train loss:0.2387=0.1044+ test loss:0.2413
epoch:76 time:67.1	 train loss:0.2385=0.104+ test loss:0.2413
epoch:77 time:67.3	 train loss:0.2383=0.1037+ test loss:0.2416
epoch:78 time:68.1	 train loss:0.2382=0.1034+ test loss:0.2414
epoch:79 time:72.0	 train loss:0.238=0.1031+ test loss:0.2415
epoch:80 time:68.8	 train loss:0.2379=0.1029+ test loss:0.2414
epoch:81 time:66.7	 train loss:0.2378=0.1026+ test loss:0.2414
epoch:82 time:68.9	 train loss:0.2377=0.1023+ test loss:0.2414
epoch:83 time:67.9	 train loss:0.2376=0.1021+ test loss:0.2412
epoch:84 time:66.5	 train loss:0.2375=0.1019+ test loss:0.2415
epoch:85 time:71.1	 train loss:0.2374=0.1017+ test loss:0.2416
epoch:86 time:68.1	 train loss:0.2373=0.1014+ test loss:0.2414
epoch:87 time:72.9	 train loss:0.2371=0.1011+ test loss:0.2415
epoch:88 time:67.3	 train loss:0.2371=0.101+ test loss:0.2415
epoch:89 time:70.8	 train loss:0.237=0.1008+ test loss:0.2415
epoch:90 time:66.5	 train loss:0.2369=0.1005+ test loss:0.2413
epoch:91 time:71.4	 train loss:0.2367=0.1002+ test loss:0.2415
epoch:92 time:66.9	 train loss:0.2367=0.1001+ test loss:0.2413
epoch:93 time:67.7	 train loss:0.2366=0.0999+ test loss:0.2415
epoch:94 time:70.4	 train loss:0.2365=0.0997+ test loss:0.2416
epoch:95 time:67.3	 train loss:0.2363=0.0994+ test loss:0.2415
epoch:96 time:69.2	 train loss:0.2363=0.0993+ test loss:0.2418
epoch:97 time:68.1	 train loss:0.2363=0.0992+ test loss:0.2418
epoch:98 time:70.8	 train loss:0.2362=0.099+ test loss:0.2418
epoch:99 time:69.3	 train loss:0.2361=0.0988+ test loss:0.2417
epoch:100 time:68.6	 train loss:0.2359=0.0985+ test loss:0.2418
epoch:101 time:67.9	 train loss:0.2359=0.0984+ test loss:0.2418
epoch:102 time:69.3	 train loss:0.2358=0.0982+ test loss:0.2418
epoch:103 time:68.1	 train loss:0.2357=0.0981+ test loss:0.2418
epoch:104 time:68.1	 train loss:0.2357=0.0979+ test loss:0.2416
epoch:105 time:70.7	 train loss:0.2356=0.0977+ test loss:0.2418
epoch:106 time:69.8	 train loss:0.2355=0.0976+ test loss:0.2416
epoch:107 time:68.1	 train loss:0.2355=0.0975+ test loss:0.2418
epoch:108 time:70.8	 train loss:0.2354=0.0973+ test loss:0.2416
epoch:109 time:68.4	 train loss:0.2353=0.0972+ test loss:0.242
epoch:110 time:67.0	 train loss:0.2353=0.0971+ test loss:0.2417
epoch:111 time:70.9	 train loss:0.2352=0.0969+ test loss:0.2415
epoch:112 time:67.9	 train loss:0.2352=0.0968+ test loss:0.242
epoch:113 time:70.7	 train loss:0.2352=0.0967+ test loss:0.2418
epoch:114 time:69.5	 train loss:0.2351=0.0966+ test loss:0.2418
epoch:115 time:68.4	 train loss:0.235=0.0964+ test loss:0.2418
epoch:116 time:67.3	 train loss:0.235=0.0963+ test loss:0.2418
epoch:117 time:69.0	 train loss:0.2349=0.0961+ test loss:0.2421
epoch:118 time:68.1	 train loss:0.2349=0.096+ test loss:0.2418
epoch:119 time:67.3	 train loss:0.2348=0.0959+ test loss:0.2418
epoch:120 time:69.0	 train loss:0.2348=0.0959+ test loss:0.242
epoch:121 time:68.5	 train loss:0.2347=0.0957+ test loss:0.242
epoch:122 time:69.2	 train loss:0.2347=0.0956+ test loss:0.2421
epoch:123 time:68.9	 train loss:0.2346=0.0955+ test loss:0.2422
epoch:124 time:71.5	 train loss:0.2346=0.0954+ test loss:0.2421
epoch:125 time:68.9	 train loss:0.2346=0.0953+ test loss:0.2423
epoch:126 time:67.1	 train loss:0.2345=0.0952+ test loss:0.2421
epoch:127 time:69.2	 train loss:0.2346=0.0952+ test loss:0.2421
epoch:128 time:69.8	 train loss:0.2345=0.0951+ test loss:0.2421
epoch:129 time:68.8	 train loss:0.2344=0.0949+ test loss:0.2419
epoch:130 time:67.9	 train loss:0.2344=0.0948+ test loss:0.2423
epoch:131 time:71.9	 train loss:0.2343=0.0947+ test loss:0.2423
epoch:132 time:69.8	 train loss:0.2343=0.0946+ test loss:0.2424
epoch:133 time:67.1	 train loss:0.2343=0.0945+ test loss:0.2421
epoch:134 time:69.7	 train loss:0.2343=0.0945+ test loss:0.2423
epoch:135 time:68.0	 train loss:0.2343=0.0945+ test loss:0.2423
epoch:136 time:68.2	 train loss:0.2341=0.0943+ test loss:0.2423
epoch:137 time:70.4	 train loss:0.2341=0.0941+ test loss:0.2423
epoch:138 time:66.2	 train loss:0.2341=0.0941+ test loss:0.2423
epoch:139 time:72.2	 train loss:0.2342=0.0941+ test loss:0.2426
epoch:140 time:67.5	 train loss:0.234=0.0939+ test loss:0.2424
epoch:141 time:68.3	 train loss:0.2341=0.0939+ test loss:0.2423
epoch:142 time:67.1	 train loss:0.2339=0.0937+ test loss:0.2425
epoch:143 time:68.1	 train loss:0.2339=0.0937+ test loss:0.2424
epoch:144 time:67.8	 train loss:0.234=0.0937+ test loss:0.2427
epoch:145 time:68.0	 train loss:0.2339=0.0936+ test loss:0.2425
epoch:146 time:67.4	 train loss:0.2338=0.0935+ test loss:0.2424
epoch:147 time:67.1	 train loss:0.2338=0.0935+ test loss:0.2426
epoch:148 time:68.8	 train loss:0.2338=0.0934+ test loss:0.2424
epoch:149 time:69.6	 train loss:0.2339=0.0934+ test loss:0.2427
epoch:150 time:71.3	 train loss:0.2337=0.0933+ test loss:0.2426
epoch:151 time:69.0	 train loss:0.2338=0.0932+ test loss:0.2425
epoch:152 time:67.6	 train loss:0.2337=0.0931+ test loss:0.2426
epoch:153 time:68.3	 train loss:0.2337=0.0931+ test loss:0.2426
epoch:154 time:70.6	 train loss:0.2337=0.093+ test loss:0.2425
epoch:155 time:69.7	 train loss:0.2336=0.0929+ test loss:0.2425
epoch:156 time:67.2	 train loss:0.2336=0.0929+ test loss:0.2426
epoch:157 time:71.2	 train loss:0.2336=0.0928+ test loss:0.2424
epoch:158 time:70.8	 train loss:0.2336=0.0928+ test loss:0.2424
epoch:159 time:67.8	 train loss:0.2335=0.0927+ test loss:0.2426
epoch:160 time:71.4	 train loss:0.2335=0.0926+ test loss:0.2425
epoch:161 time:68.0	 train loss:0.2335=0.0926+ test loss:0.2427
epoch:162 time:65.8	 train loss:0.2335=0.0926+ test loss:0.2427
epoch:163 time:67.6	 train loss:0.2335=0.0925+ test loss:0.2426
epoch:164 time:66.8	 train loss:0.2334=0.0924+ test loss:0.2426
epoch:165 time:71.8	 train loss:0.2334=0.0924+ test loss:0.2428
epoch:166 time:66.2	 train loss:0.2335=0.0924+ test loss:0.243
epoch:167 time:67.8	 train loss:0.2334=0.0923+ test loss:0.2427
epoch:168 time:66.0	 train loss:0.2333=0.0921+ test loss:0.2427
epoch:169 time:70.9	 train loss:0.2333=0.092+ test loss:0.2427
epoch:170 time:67.5	 train loss:0.2333=0.0921+ test loss:0.2429
epoch:171 time:66.8	 train loss:0.2334=0.0922+ test loss:0.2429
epoch:172 time:69.2	 train loss:0.2333=0.092+ test loss:0.2428
epoch:173 time:67.3	 train loss:0.2333=0.092+ test loss:0.2428
epoch:174 time:69.9	 train loss:0.2333=0.0919+ test loss:0.2426
epoch:175 time:68.1	 train loss:0.2332=0.0919+ test loss:0.2427
epoch:176 time:71.6	 train loss:0.2332=0.0918+ test loss:0.243
epoch:177 time:69.3	 train loss:0.2331=0.0917+ test loss:0.2432
epoch:178 time:69.0	 train loss:0.2331=0.0917+ test loss:0.2431
epoch:179 time:68.6	 train loss:0.2331=0.0916+ test loss:0.2433
epoch:180 time:67.5	 train loss:0.2332=0.0917+ test loss:0.243
epoch:181 time:66.9	 train loss:0.2332=0.0917+ test loss:0.243
epoch:182 time:67.2	 train loss:0.2331=0.0916+ test loss:0.243
epoch:183 time:71.1	 train loss:0.2331=0.0915+ test loss:0.2431
epoch:184 time:68.8	 train loss:0.2331=0.0915+ test loss:0.2428
epoch:185 time:67.2	 train loss:0.233=0.0914+ test loss:0.2432
epoch:186 time:70.4	 train loss:0.2331=0.0914+ test loss:0.2432
epoch:187 time:69.6	 train loss:0.233=0.0914+ test loss:0.2431
epoch:188 time:67.7	 train loss:0.233=0.0912+ test loss:0.2432
epoch:189 time:71.4	 train loss:0.2329=0.0912+ test loss:0.2429
epoch:190 time:66.4	 train loss:0.233=0.0913+ test loss:0.2433
epoch:191 time:69.1	 train loss:0.2331=0.0913+ test loss:0.2432
epoch:192 time:67.7	 train loss:0.2329=0.0912+ test loss:0.2432
epoch:193 time:66.6	 train loss:0.2329=0.0911+ test loss:0.243
epoch:194 time:66.3	 train loss:0.2329=0.0911+ test loss:0.2433
epoch:195 time:68.6	 train loss:0.2328=0.091+ test loss:0.2433
epoch:196 time:66.9	 train loss:0.2329=0.091+ test loss:0.2432
epoch:197 time:66.0	 train loss:0.2329=0.091+ test loss:0.2432
epoch:198 time:68.9	 train loss:0.2329=0.091+ test loss:0.2432
epoch:199 time:66.8	 train loss:0.2329=0.091+ test loss:0.2431
epoch:200 time:68.1	 train loss:0.2329=0.0909+ test loss:0.2433
epoch:201 time:67.8	 train loss:0.2328=0.0909+ test loss:0.2432
epoch:202 time:68.2	 train loss:0.2329=0.0909+ test loss:0.2433
epoch:203 time:68.0	 train loss:0.2328=0.0908+ test loss:0.2434
epoch:204 time:67.4	 train loss:0.2328=0.0908+ test loss:0.2432
epoch:205 time:70.4	 train loss:0.2328=0.0908+ test loss:0.2432
epoch:206 time:66.7	 train loss:0.2328=0.0908+ test loss:0.2432
epoch:207 time:66.8	 train loss:0.2328=0.0907+ test loss:0.2432
epoch:208 time:66.3	 train loss:0.2328=0.0907+ test loss:0.2433
epoch:209 time:68.6	 train loss:0.2328=0.0906+ test loss:0.2435
epoch:210 time:67.5	 train loss:0.2329=0.0907+ test loss:0.2434
epoch:211 time:66.1	 train loss:0.2328=0.0907+ test loss:0.2433
epoch:212 time:68.1	 train loss:0.2327=0.0906+ test loss:0.2435
epoch:213 time:65.1	 train loss:0.2327=0.0905+ test loss:0.2436
epoch:214 time:66.7	 train loss:0.2327=0.0905+ test loss:0.2434
epoch:215 time:69.1	 train loss:0.2327=0.0905+ test loss:0.2433
epoch:216 time:65.8	 train loss:0.2327=0.0904+ test loss:0.2435
epoch:217 time:68.7	 train loss:0.2327=0.0904+ test loss:0.2435
epoch:218 time:68.1	 train loss:0.2326=0.0903+ test loss:0.2434
epoch:219 time:66.2	 train loss:0.2327=0.0904+ test loss:0.2433
epoch:220 time:66.9	 train loss:0.2326=0.0903+ test loss:0.2436
epoch:221 time:69.5	 train loss:0.2327=0.0903+ test loss:0.2433
epoch:222 time:66.6	 train loss:0.2327=0.0903+ test loss:0.2437
epoch:223 time:65.5	 train loss:0.2327=0.0903+ test loss:0.2434
epoch:224 time:71.0	 train loss:0.2326=0.0903+ test loss:0.2436
epoch:225 time:67.9	 train loss:0.2327=0.0903+ test loss:0.2435
epoch:226 time:68.8	 train loss:0.2326=0.0901+ test loss:0.2434
epoch:227 time:68.5	 train loss:0.2327=0.0902+ test loss:0.2433
epoch:228 time:68.7	 train loss:0.2326=0.0901+ test loss:0.2436
epoch:229 time:69.1	 train loss:0.2326=0.0901+ test loss:0.2435
epoch:230 time:67.3	 train loss:0.2325=0.09+ test loss:0.2436
epoch:231 time:66.5	 train loss:0.2327=0.0902+ test loss:0.2435
epoch:232 time:66.7	 train loss:0.2326=0.09+ test loss:0.2433
epoch:233 time:66.2	 train loss:0.2326=0.0901+ test loss:0.2436
epoch:234 time:67.4	 train loss:0.2326=0.0901+ test loss:0.2436
epoch:235 time:68.3	 train loss:0.2326=0.09+ test loss:0.2436
epoch:236 time:67.9	 train loss:0.2325=0.0899+ test loss:0.2437
epoch:237 time:65.1	 train loss:0.2325=0.09+ test loss:0.2436
epoch:238 time:68.9	 train loss:0.2325=0.0899+ test loss:0.2438
epoch:239 time:66.9	 train loss:0.2324=0.0898+ test loss:0.2435
epoch:240 time:65.7	 train loss:0.2325=0.0899+ test loss:0.2437
epoch:241 time:68.1	 train loss:0.2325=0.0899+ test loss:0.2437
epoch:242 time:67.0	 train loss:0.2325=0.0899+ test loss:0.2439
epoch:243 time:70.3	 train loss:0.2325=0.0899+ test loss:0.2438
epoch:244 time:68.4	 train loss:0.2325=0.0898+ test loss:0.2438
epoch:245 time:68.4	 train loss:0.2323=0.0897+ test loss:0.2436
epoch:246 time:66.2	 train loss:0.2325=0.0898+ test loss:0.2437
epoch:247 time:70.0	 train loss:0.2325=0.0898+ test loss:0.2438
epoch:248 time:67.0	 train loss:0.2324=0.0897+ test loss:0.2439
epoch:249 time:65.5	 train loss:0.2325=0.0897+ test loss:0.2437
epoch:250 time:66.4	 train loss:0.2324=0.0896+ test loss:0.2439
epoch:251 time:65.9	 train loss:0.2324=0.0896+ test loss:0.2436
epoch:252 time:68.8	 train loss:0.2325=0.0897+ test loss:0.2437
epoch:253 time:67.1	 train loss:0.2325=0.0897+ test loss:0.2439
epoch:254 time:67.9	 train loss:0.2325=0.0897+ test loss:0.2438
epoch:255 time:66.8	 train loss:0.2325=0.0897+ test loss:0.2438
epoch:256 time:67.2	 train loss:0.2324=0.0896+ test loss:0.2438
epoch:257 time:69.1	 train loss:0.2325=0.0896+ test loss:0.2438
epoch:258 time:67.9	 train loss:0.2324=0.0895+ test loss:0.2439
epoch:259 time:66.8	 train loss:0.2324=0.0895+ test loss:0.244
epoch:260 time:67.8	 train loss:0.2324=0.0896+ test loss:0.2438
epoch:261 time:69.9	 train loss:0.2324=0.0895+ test loss:0.2439
epoch:262 time:67.4	 train loss:0.2324=0.0895+ test loss:0.2441
epoch:263 time:65.3	 train loss:0.2324=0.0895+ test loss:0.2438
epoch:264 time:69.1	 train loss:0.2323=0.0894+ test loss:0.244
epoch:265 time:66.9	 train loss:0.2323=0.0894+ test loss:0.2438
epoch:266 time:67.3	 train loss:0.2323=0.0894+ test loss:0.2438
epoch:267 time:68.6	 train loss:0.2324=0.0895+ test loss:0.2439
epoch:268 time:65.6	 train loss:0.2323=0.0894+ test loss:0.244
epoch:269 time:68.6	 train loss:0.2323=0.0894+ test loss:0.2438
epoch:270 time:68.1	 train loss:0.2323=0.0893+ test loss:0.244
epoch:271 time:67.9	 train loss:0.2323=0.0893+ test loss:0.244
epoch:272 time:66.9	 train loss:0.2324=0.0894+ test loss:0.2439
epoch:273 time:68.9	 train loss:0.2324=0.0894+ test loss:0.2439
epoch:274 time:66.4	 train loss:0.2323=0.0893+ test loss:0.2438
epoch:275 time:66.3	 train loss:0.2323=0.0893+ test loss:0.2442
epoch:276 time:69.8	 train loss:0.2324=0.0893+ test loss:0.2442
epoch:277 time:66.3	 train loss:0.2322=0.0892+ test loss:0.2439
epoch:278 time:68.0	 train loss:0.2323=0.0892+ test loss:0.2439
epoch:279 time:67.9	 train loss:0.2322=0.0892+ test loss:0.2441
epoch:280 time:72.4	 train loss:0.2322=0.0892+ test loss:0.2439
epoch:281 time:69.7	 train loss:0.2322=0.0892+ test loss:0.244
epoch:282 time:67.2	 train loss:0.2322=0.0891+ test loss:0.2441
epoch:283 time:69.5	 train loss:0.2323=0.0892+ test loss:0.2439
epoch:284 time:69.7	 train loss:0.2323=0.0892+ test loss:0.2439
epoch:285 time:70.5	 train loss:0.2323=0.0892+ test loss:0.2438
epoch:286 time:68.5	 train loss:0.2323=0.0892+ test loss:0.2441
epoch:287 time:71.4	 train loss:0.2322=0.0891+ test loss:0.244
epoch:288 time:68.5	 train loss:0.2322=0.0891+ test loss:0.244
epoch:289 time:66.8	 train loss:0.2323=0.0892+ test loss:0.244
epoch:290 time:71.2	 train loss:0.2323=0.0892+ test loss:0.2441
epoch:291 time:68.0	 train loss:0.2322=0.0891+ test loss:0.2442
epoch:292 time:66.8	 train loss:0.2322=0.0891+ test loss:0.2442
epoch:293 time:70.3	 train loss:0.2322=0.0891+ test loss:0.2441
epoch:294 time:67.5	 train loss:0.2323=0.0891+ test loss:0.2442
epoch:295 time:72.5	 train loss:0.2323=0.0891+ test loss:0.2442
epoch:296 time:67.1	 train loss:0.2323=0.0891+ test loss:0.2441
epoch:297 time:69.9	 train loss:0.2322=0.0891+ test loss:0.2442
epoch:298 time:66.7	 train loss:0.2321=0.0889+ test loss:0.2444
epoch:299 time:69.7	 train loss:0.2322=0.089+ test loss:0.2442
epoch:300 time:67.2	 train loss:0.2322=0.089+ test loss:0.2443
epoch:301 time:67.0	 train loss:0.2322=0.089+ test loss:0.244
epoch:302 time:69.5	 train loss:0.2322=0.089+ test loss:0.2443
epoch:303 time:66.7	 train loss:0.2322=0.089+ test loss:0.2442
epoch:304 time:70.9	 train loss:0.2322=0.0889+ test loss:0.2442
epoch:305 time:68.9	 train loss:0.2322=0.0889+ test loss:0.2443
epoch:306 time:70.4	 train loss:0.2322=0.089+ test loss:0.2443
epoch:307 time:69.3	 train loss:0.2322=0.0889+ test loss:0.2442
epoch:308 time:69.0	 train loss:0.2321=0.0889+ test loss:0.244
epoch:309 time:69.4	 train loss:0.2321=0.0888+ test loss:0.2442
epoch:310 time:70.6	 train loss:0.2322=0.0889+ test loss:0.2443
epoch:311 time:70.4	 train loss:0.2321=0.0888+ test loss:0.2444
epoch:312 time:67.3	 train loss:0.2321=0.0888+ test loss:0.2443
epoch:313 time:69.5	 train loss:0.2322=0.0889+ test loss:0.2442
epoch:314 time:69.8	 train loss:0.2322=0.0888+ test loss:0.2442
epoch:315 time:68.6	 train loss:0.2322=0.0888+ test loss:0.2442
epoch:316 time:69.0	 train loss:0.2321=0.0888+ test loss:0.2442
epoch:317 time:67.5	 train loss:0.2322=0.0888+ test loss:0.2444
epoch:318 time:68.3	 train loss:0.2321=0.0887+ test loss:0.2443
epoch:319 time:69.8	 train loss:0.2321=0.0888+ test loss:0.2442
epoch:320 time:69.2	 train loss:0.2321=0.0887+ test loss:0.2444
epoch:321 time:71.0	 train loss:0.2321=0.0888+ test loss:0.2444
epoch:322 time:67.3	 train loss:0.2321=0.0888+ test loss:0.2443
epoch:323 time:69.8	 train loss:0.2321=0.0887+ test loss:0.2443
epoch:324 time:67.5	 train loss:0.2321=0.0887+ test loss:0.2442
epoch:325 time:68.5	 train loss:0.2321=0.0887+ test loss:0.2441
epoch:326 time:68.2	 train loss:0.2322=0.0888+ test loss:0.2442
epoch:327 time:66.2	 train loss:0.2321=0.0887+ test loss:0.2442
epoch:328 time:67.5	 train loss:0.2321=0.0888+ test loss:0.2442
epoch:329 time:67.6	 train loss:0.2321=0.0887+ test loss:0.2444
epoch:330 time:69.5	 train loss:0.2321=0.0887+ test loss:0.2443
epoch:331 time:68.5	 train loss:0.232=0.0886+ test loss:0.2443
epoch:332 time:70.1	 train loss:0.2321=0.0887+ test loss:0.2441
epoch:333 time:71.2	 train loss:0.2321=0.0887+ test loss:0.244
epoch:334 time:67.6	 train loss:0.2321=0.0887+ test loss:0.2442
epoch:335 time:69.4	 train loss:0.2321=0.0887+ test loss:0.2445
epoch:336 time:69.6	 train loss:0.232=0.0886+ test loss:0.2442
epoch:337 time:69.5	 train loss:0.232=0.0886+ test loss:0.2446
epoch:338 time:68.8	 train loss:0.2321=0.0886+ test loss:0.2445
epoch:339 time:69.9	 train loss:0.2321=0.0887+ test loss:0.2441
epoch:340 time:69.0	 train loss:0.2321=0.0887+ test loss:0.2444
epoch:341 time:68.0	 train loss:0.2321=0.0886+ test loss:0.2444
epoch:342 time:69.6	 train loss:0.2321=0.0886+ test loss:0.2442
epoch:343 time:67.3	 train loss:0.232=0.0886+ test loss:0.2441
epoch:344 time:68.1	 train loss:0.232=0.0885+ test loss:0.2443
epoch:345 time:71.5	 train loss:0.2321=0.0886+ test loss:0.2444
epoch:346 time:68.1	 train loss:0.2321=0.0886+ test loss:0.2444
epoch:347 time:70.5	 train loss:0.2321=0.0886+ test loss:0.2446
epoch:348 time:66.6	 train loss:0.232=0.0885+ test loss:0.2445
epoch:349 time:70.5	 train loss:0.232=0.0885+ test loss:0.2445
```

#### test

```shell
s0
has results save path
has model save path
--------test processing-------
-0.001713 0.0531892 -0.0008817 0.0139672 -0.0014701 0.0079551 -0.0008823 0.0043282 -0.001433 0.0031781 -0.001276 0.0165237
0.0018608 0.0239208 -8.97e-05 0.0096702 -0.0001467 0.0042483 -0.000829 0.0028051 -0.0005696 0.0017311 4.52e-05 0.008476
epoch:150time:542.66     test hit:0.0342 ndcg:0.0259 recall:0.0328
-0.0017066 0.0534387 -0.0008945 0.0139768 -0.0014752 0.007943 -0.000896 0.0043173 -0.0014389 0.0031686 -0.0012822 0.016569
0.0018409 0.0240154 -9.47e-05 0.0096801 -0.000159 0.0042421 -0.0008343 0.0027979 -0.0005811 0.0017262 3.43e-05 0.0084932
epoch:155time:530.33     test hit:0.0342 ndcg:0.026 recall:0.0329
-0.0016873 0.0536723 -0.0008876 0.0139863 -0.0014671 0.0079325 -0.0008909 0.0043067 -0.0014316 0.0031598 -0.0012729 0.0166116
0.0018356 0.0241056 -9.12e-05 0.0096901 -0.0001569 0.0042363 -0.0008301 0.0027913 -0.0005783 0.0017213 3.58e-05 0.0085098
epoch:160time:513.58     test hit:0.0344 ndcg:0.0261 recall:0.0331
-0.001668 0.053886 -0.0008852 0.0139935 -0.0014608 0.0079201 -0.0008893 0.0042959 -0.0014261 0.0031503 -0.0012659 0.0166493
0.0018302 0.0241896 -8.83e-05 0.0096975 -0.0001567 0.00423 -0.0008266 0.0027842 -0.0005777 0.0017164 3.62e-05 0.0085244
epoch:165time:522.13     test hit:0.0345 ndcg:0.0262 recall:0.0332
-0.0016505 0.0540849 -0.0008827 0.0140011 -0.0014522 0.0079108 -0.0008881 0.0042876 -0.0014187 0.0031435 -0.0012585 0.0166857
0.0018224 0.0242669 -8.43e-05 0.0097055 -0.0001576 0.0042251 -0.0008223 0.0027787 -0.0005778 0.0017128 3.61e-05 0.0085387
epoch:170time:514.15     test hit:0.0347 ndcg:0.0263 recall:0.0334
-0.0016529 0.0542732 -0.0008936 0.014009 -0.001462 0.0079013 -0.0009003 0.0042791 -0.0014289 0.0031361 -0.0012675 0.0167198
0.0018028 0.0243417 -9.34e-05 0.0097132 -0.0001686 0.0042204 -0.0008308 0.0027732 -0.0005883 0.0017089 2.43e-05 0.0085523
epoch:175time:513.04     test hit:0.0347 ndcg:0.0263 recall:0.0333
-0.0016351 0.0544443 -0.000893 0.0140145 -0.0014553 0.0078924 -0.0009001 0.0042711 -0.001423 0.0031296 -0.0012613 0.0167505
0.0017939 0.024409 -9.11e-05 0.0097196 -0.00017 0.0042156 -0.0008275 0.0027681 -0.0005887 0.0017054 2.33e-05 0.0085644
epoch:180time:512.47     test hit:0.0346 ndcg:0.0263 recall:0.0333
-0.0016261 0.0546131 -0.0008949 0.0140207 -0.0014545 0.0078837 -0.000902 0.0042634 -0.0014226 0.0031233 -0.00126 0.0167809
0.001787 0.0244773 -9.19e-05 0.0097259 -0.0001728 0.004211 -0.0008283 0.0027632 -0.0005913 0.001702 2.06e-05 0.0085767
epoch:185time:515.94     test hit:0.0348 ndcg:0.0264 recall:0.0334
-0.001612 0.0547551 -0.0008923 0.0140236 -0.0014479 0.0078739 -0.0009001 0.0042555 -0.0014163 0.0031166 -0.0012537 0.016805
0.0017833 0.0245338 -8.73e-05 0.0097296 -0.000172 0.004206 -0.0008241 0.0027579 -0.0005902 0.0016985 2.19e-05 0.008586
epoch:190time:517.9      test hit:0.0348 ndcg:0.0265 recall:0.0334
-0.001601 0.054905 -0.000892 0.0140316 -0.0014434 0.0078685 -0.0009004 0.0042505 -0.0014124 0.0031122 -0.0012499 0.0168336
0.0017775 0.0245934 -8.62e-05 0.0097369 -0.0001735 0.0042034 -0.000822 0.0027545 -0.0005913 0.0016963 2.09e-05 0.0085977
epoch:195time:510.14     test hit:0.0349 ndcg:0.0265 recall:0.0335
```
