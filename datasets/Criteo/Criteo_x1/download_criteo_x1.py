# This file is modified from https://github.com/WeiyuCheng/AFN-AAAI-20/blob/master/src/download_criteo_and_avazu.py
# to download the preprocessed data split Criteo_x1

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
    print("Begin to download criteo data, the total size is 3GB...")
    download('https://worksheets.codalab.org/rest/bundles/0x8dca5e7bac42470aa445f9a205d177c6/contents/blob/', './criteo.zip')
    print("Unzipping criteo dataset...")
    with zipfile.ZipFile('./criteo.zip', 'r') as zip_ref:
        zip_ref.extractall('./criteo/')
    print("Done.")

