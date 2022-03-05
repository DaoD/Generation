import os
import glob
import time
import argparse
import moxing as mox
import sys

code_rootdir = "s3://obs-app-2020042019121301221/SEaaKM/z50021765/"
data_rootdir = "s3://obs-app-2020042019121301221/SEaaKM/m50017495/"
# s3_rootdir = "s3://bucket-852/m50017495/"

mox.file.shift('os', 'mox')
os.makedirs("/cache/code")
os.makedirs("/cache/data")
mox.file.copy_parallel(code_rootdir + '/KGDialogue', '/cache/code')
mox.file.copy_parallel(data_rootdir + '/code/session_search', '/cache/data')

os.system('pip install /cache/data/torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl')
os.system('pip install /cache/data/torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl')

os.system('pip install -r /cache/code/requirements.txt')
os.system('pip install dgl-cu101')

s3_model_path = code_rootdir + "/PretrainedLM/"
s3_output_path = code_rootdir + "/KGDialogue/output/"

def extract_data():
    os.makedirs('/cache/pretrained_model')
    mox.file.copy_parallel(s3_model_path, '/cache/pretrained_model')
    os.makedirs('/cache/output')
    os.makedirs('/cache/output/logs')
    os.makedirs('/cache/output/models')


def main():
    extract_data()

    os.system(f"cd /cache/code/with_generation/ && python runModel.py --score_file_path /cache/output/ --per_gpu_batch_size 32  --per_gpu_test_batch_size 32 --model_name_or_path /cache/pretrained_model/BARTBase/ --config_name /cache/pretrained_model/BARTBase/ --tokenizer_name /cache/pretrained_model/BARTBase/ --log_path /cache/output/logs/ --save_path /cache/output/models/ --epochs 3 --learning_rate 1e-5 --file_suffix self_original --task personachat")

    mox.file.copy_parallel('/cache/output', s3_output_path)
    print("write success")

if __name__ == '__main__':
    main()