# This file is modified from https://github.com/WeiyuCheng/AFN-AAAI-20/blob/master/src/download_criteo_and_avazu.py
# to download the preprocessed data split Avazu_x1

import os
import zipfile
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

if __name__ == "__main__":
    print("Begin to download avazu data, the total size is 683MB...")
    download('https://worksheets.codalab.org/rest/bundles/0xf5ab597052744680b1a55986557472c7/contents/blob/', './avazu.zip')
    print("Unzipping avazu dataset...")
    with zipfile.ZipFile('./avazu.zip', 'r') as zip_ref:
        zip_ref.extractall('./avazu/')
    print("Done.")


