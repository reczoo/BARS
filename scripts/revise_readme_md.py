import regex
import glob

data_path = "../ranking/**/*.md"

file_list = glob.glob(data_path, recursive=True)
for f in file_list:
    if "taobaoad_x1" in f:
        print(f)
        with open(f, "r") as fd:
            res = fd.read()
            phrase = "Author: [XUEPAI](https://github.com/xue-pai)"
            correct = "Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)"
            res = res.replace(phrase, correct)
            phrase = "Please refer to the BARS dataset [TaobaoAd_x1](https://github.com/openbenchmark/BARS/blob/main/datasets/Taobao#TaobaoAd_x1) to get data ready."
            correct = "Please refer to [TaobaoAd_x1](https://github.com/reczoo/Datasets/tree/main/Taobao/TaobaoAd_x1) to get the dataset details."
            res = res.replace(phrase, correct)
            res = res.replace("xue-pai", "reczoo")
            # print(res[0:200])
        with open(f, "w") as fd:
            fd.write(res)
