import os
import numpy as np
import glob
import pandas as pd


input_dir = "../ranking/ctr/FinalNet/FinalNet_criteo_x4_001"

model_id = os.path.basename(input_dir)
model_name = input_dir.split("ctr/")[1].split("/")[0]
dataset_id = "_".join(model_id.split("_")[model_name.count("_") + 1:]).capitalize()
short_dataset_id = "_".join(dataset_id.split("_")[0:2]).capitalize()
dataset_name = dataset_id.split("_")[0]

# dataset_name = "MovieLens"
# dataset_id = "MovielensLatest_x1"
# short_dataset_id = "MovielensLatest_x1"

# dataset_name = "Amazon"
# dataset_id = "AmazonElectronics_x1"
# short_dataset_id = "AmazonElectronics_x1"

# dataset_name = "KuaiShou"
# dataset_id = "KuaiVideo_x1"
# short_dataset_id = "KuaiVideo_x1"

# dataset_name = "MicroVideo"
# dataset_id = "MicroVideo1.7M_x1"
# short_dataset_id = "MicroVideo1.7M_x1"

# dataset_name = "Taobao"
# dataset_id = "TaobaoAd_x1"
# short_dataset_id = "TaobaoAd_x1"

# dataset_name = "Frappe"
# dataset_id = "Frappe_x1"
# short_dataset_id = "Frappe_x1"

# dataset_name = "Avazu"
# dataset_id = "Avazu_x4"
# short_dataset_id = "Avazu_x4"

dataset_name = "Criteo"
dataset_id = "Criteo_x4"
short_dataset_id = "Criteo_x4"

dataset_url = f"https://github.com/reczoo/Datasets/tree/main/{dataset_name}/{short_dataset_id}"
print(dataset_url)

hardware_envs = ""
software_envs = ""
read_state = None
with open(os.path.join(input_dir, "environments.txt"), "r") as fd:
    for line in fd:
        if line.startswith("[Hardware]"):
            read_state = "hardware"
            continue
        elif line.startswith("[Software]"):
            read_state = "software"
            continue
        if read_state == "hardware" and line.strip() != "":
            hardware_envs += "  " + line
        elif read_state == "software" and line.strip() != "":
            software_envs += "  " + line

fuxictr_version = software_envs.strip().split("\n")[-1].split(":")[-1].strip()
model_url = f"https://github.com/reczoo/FuxiCTR/tree/v{fuxictr_version}/model_zoo/{model_name}"
fuxictr_url1 = f"https://github.com/reczoo/FuxiCTR/tree/v{fuxictr_version}"
fuxictr_url2 = f"https://github.com/reczoo/FuxiCTR/archive/refs/tags/v{fuxictr_version}.zip"

log_path = glob.glob(os.path.join(input_dir, "*.log"))[0]
print(log_path)
exp_id = os.path.basename(log_path).split(".")[0]
logs = open(log_path).read()

config_path = glob.glob(os.path.join(input_dir, "*_config*"))[0]
print(config_path)
if os.path.isdir(config_path):
    config_dir = os.path.basename(config_path)

df = pd.read_csv(os.path.join(input_dir, "results.csv"), header=None)
res = df.values[:, -1]
num_runs = len(res)
if num_runs == 1:
    result_table = ""
    rows_header = False
else:
    result_table = f"Total {num_runs} runs:\n\n"
    rows_header = True

res_array = []
for i, line in enumerate(res):
    metrics = line.split("]")[-1].split("-")
    if i == 0:
        result_table += "| Runs " * int(rows_header)
        for m in metrics:
            result_table += "| " + m.split(":")[0].strip() + " "
        result_table += " |\n"
        result_table += ("|:" + "-"*20 + ":") * int(rows_header)
        for m in metrics:
            result_table += "|:" + "-"*20 + ":"
        result_table += "|\n"
    result_table += "| {} ".format(i + 1) * int(rows_header)
    tmp = []
    for m in metrics:
        result_table += "| " + m.split(":")[1].strip() + " "
        tmp.append(float(m.split(":")[1].strip()))
    res_array.append(tmp)
    result_table += " |\n"
if num_runs > 1:
    avg_list = list(np.mean(np.array(res_array), axis=0))
    std_list = list(np.std(np.array(res_array), axis=0))
    result_table += "| Avg | " + " | ".join(["{:.6f}".format(x) for x in avg_list]) + " |\n"
    result_table += "| Std | &#177;" + " | &#177;".join(["{:.8f}".format(x) for x in std_list]) + " |\n"
# print(result_table)

markdown_template = \
f"## {model_id}\n\
\n\
A hands-on guide to run the {model_name} model on the {dataset_id} dataset.\n\
\n\
Author: [BARS Benchmark](https://github.com/reczoo/BARS/blob/main/CITATION)\n\
\n\
\n| [Environments](#Environments) | [Dataset](#Dataset) | [Code](#Code) | [Results](#Results) | [Logs](#Logs) |\
\n|:-----------------------------:|:-----------:|:--------:|:--------:|-------|\
\n\
### Environments\n\
+ Hardware\n\
\n\
  ```python\n\
{hardware_envs}\n\
  ```\n\
\n\
+ Software\n\
\n\
  ```python\n\
{software_envs}\n\
  ```\n\
\n\
### Dataset\n\
Please refer to [{dataset_id}]({dataset_url}) to get the dataset details.\n\
\n\
### Code\n\
\n\
We use the [{model_name}]({model_url}) model code from [FuxiCTR-v{fuxictr_version}]({fuxictr_url1}) for this experiment.\n\
\n\
Running steps:\n\
\n\
1. Download [FuxiCTR-v{fuxictr_version}]({fuxictr_url2}) and install all the dependencies listed in the [environments](#environments).\n\
    \n\
    ```bash\n\
    pip uninstall fuxictr\n\
    pip install fuxictr=={fuxictr_version}\n\
    ```\n\
\n\
2. Create a data directory and put the downloaded data files in `../data/{dataset_name}/{dataset_id}`.\n\
\n\
3. Both `dataset_config.yaml` and `model_config.yaml` files are available in [{config_dir}](./{config_dir}). Please make sure that the data paths in `dataset_config.yaml` are correctly set.\n\
\n\
4. Run the following script to start training and evaluation.\n\
\n\
    ```bash\n\
    cd FuxiCTR/model_zoo/{model_name}\n\
    nohup python run_expid.py --config YOUR_PATH/{model_name}/{config_dir} --expid {exp_id} --gpu 0 > run.log &\n\
    tail -f run.log\n\
    ```\n\
\n\
### Results\n\
\n\
{result_table}\n\
\n\
### Logs\n\
```python\n\
{logs}\n\
```\n\
"
# print(markdown_template)

with open(os.path.join(input_dir, "README.md"), "w") as fout:
    fout.write(markdown_template) 
