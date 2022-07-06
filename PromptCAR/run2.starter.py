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
mox.file.copy_parallel(code_rootdir + '/PromptCAR', '/cache/code')
mox.file.copy_parallel(data_rootdir + '/code/session_search', '/cache/data')

os.system('pip install /cache/data/torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl')
os.system('pip install /cache/data/torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl')

os.system('pip install -r /cache/data/requirements.txt')
os.system('pip install dgl-cu101')

s3_model_path = data_rootdir + "/data/yutao/"
s3_req_path = data_rootdir + "/data/requirement/"
s3_output_path = code_rootdir + "/PromptCAR/output/"


def install_package():
    os.makedirs('/cache/mypackages/')
    mox.file.copy_parallel(s3_req_path, '/cache/mypackages/')
    os.system("pip install sentencepiece==0.1.90")
    print("begin pytrec")
    os.system("cd /cache/mypackages/pytrec_eval-0.5 && python setup.py install")
    print("pytrec ok")


def extract_data():
    os.makedirs('/cache/pretrained_model')
    mox.file.copy_parallel(s3_model_path, '/cache/pretrained_model')

    os.makedirs('/cache/output')
    os.makedirs('/cache/output/logs')
    os.makedirs('/cache/output/models')

def parse_args():
    parser = argparse.ArgumentParser(description='Process Reader Data')
    # to ignore
    parser.add_argument('--data_url', default='s3://bucket-857/h00574873/test/model_save/',
                        help='data_url for yundao')
    parser.add_argument('--init_method', default='',
                        help='init_method for yundao')
    parser.add_argument('--train_url', default='s3://bucket-857/h00574873/test/model_save/',
                        help='train_url for yundao')
    parser.add_argument("--s3_path_dir", type=str,
                        default='s3://bucket-852/f00574594/data/HGN_data/train_data_with_tfidf30_bert_large_aug/path_data/',
                        help='define path directory')
    parser.add_argument("--s3_HGN_data_dir", type=str,
                        default='s3://bucket-852/f00574594/data/KFB_data/reader_data_no_sep/',
                        help='define output directory')
    parser.add_argument("--my_output_dir", type=str,
                        default='s3://bucket-852/m50017495/replearn/output_train/',
                        help='define output directory')
    return parser.parse_args()


def main():
    extract_data()
    args = parse_args()

    install_package()
    
    os.system(f"cd /cache/code/ && python runModel2.py --data_path /cache/data/data/ --score_file_path /cache/output/ --bert_model_path /cache/pretrained_model/BertModel/ --per_gpu_batch_size 256  --per_gpu_test_batch_size 512 --log_path /cache/output/logs/ --save_path /cache/output/models/BertPromptSS --epochs 6 --learning_rate 1e-4")

    mox.file.copy_parallel('/cache/output', s3_output_path)
    print("write success")


if __name__ == '__main__':
    main()