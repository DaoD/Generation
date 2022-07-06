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
mox.file.copy_parallel(code_rootdir + '/DialogueGeneration', '/cache/code')
mox.file.copy_parallel(data_rootdir + '/code/session_search', '/cache/data')

os.system('pip install /cache/data/torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl')
os.system('pip install /cache/data/torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl')

os.system('pip install -r /cache/code/requirements.txt')
os.system('pip install dgl-cu101')

s3_model_path = code_rootdir + "/PretrainedLM/"
s3_output_path = code_rootdir + "/DialogueGeneration/output/"

def extract_data():
    os.makedirs('/cache/pretrained_model')
    mox.file.copy_parallel(s3_model_path, '/cache/pretrained_model')
    os.makedirs('/cache/output')
    os.makedirs('/cache/output/models')
    os.makedirs('/cache/output/models/ensemble/')
    os.makedirs('/cache/output/models/retonly/')
    os.makedirs('/cache/output/models/genonly/')
    os.makedirs('/cache/output/result')
    os.makedirs('/cache/output/result/reddit3/')


def main():
    extract_data()

    os.system(f"cd /cache/code/GenOnly/ && python runGenOnlyModel.py --dataset reddit3 --per_gpu_train_batch_size 48 --per_gpu_eval_batch_size 48 --output_dir /cache/output/models/genonly/ --result_dir /cache/output/result/reddit3/ --model_name_or_path /cache/pretrained_model/DialoGPT_small/ --config_name /cache/pretrained_model/DialoGPT_small/ --tokenizer_name /cache/pretrained_model/DialoGPT_small/ --learning_rate 5e-5 --num_train_epochs 3")

    mox.file.copy_parallel('/cache/output', s3_output_path)
    print("write success")

if __name__ == '__main__':
    main()