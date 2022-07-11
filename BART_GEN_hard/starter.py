import os
os.environ.pop('CREDENTIAL_PROFILES_FILE', None)
import argparse
import sys
import moxing as mox

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="数据集路径")
parser.add_argument("--OUTPUT_DIR",type=str,help="结果输出路径")
parser.add_argument("--LOG_DIR", type=str, help="log所在路径")
args = parser.parse_args()

code_rootdir = "s3://obs-app-2020042019121301221/SEaaKM/z50021765/"
mox.file.shift('os', 'mox')
mox.file.set_auth(ak='3RTVHMUQSMJIWPAXISYV', sk='MYPHiRJxK7wKDdA1BpKJhNcByEVCWR0HTC6bNwwV')

os.makedirs("/cache/code")
os.makedirs('/cache/output')
os.makedirs('/cache/output/models')
os.makedirs('/cache/output/models/bart/')
os.makedirs('/cache/output/result')
os.makedirs('/cache/output/result/reddit3/')
os.makedirs("/cache/pretrained_model")

mox.file.copy_parallel(code_rootdir + "/DialogueGeneration", "/cache/code")
ptm_model_path = code_rootdir + "/PretrainedLM/"
output_path = code_rootdir + "/DialogueGeneration/output/"
mox.file.copy_parallel(ptm_model_path, "/cache/pretrained_model")

os.system("pip install tqdm")
os.system("pip install transformers")

# os.system("cd /cache/code/BARTEnsemble/ && python runModelEnsHard.py --dataset reddit3 --per_gpu_train_batch_size 50 --per_gpu_eval_batch_size 50 --output_dir /cache/output/models/bart/ --result_dir /cache/output/result/reddit3/ --model_name_or_path /cache/pretrained_model/BARTBase/ --config_name /cache/pretrained_model/BARTBase/ --tokenizer_name /cache/pretrained_model/BARTBase/ --learning_rate 3e-5 --num_train_epochs 3")

os.system("python /opt/huawei/schedule-train/algorithm/runModelEnsHard.py --data_dir /cache/code/ --dataset reddit3 --per_gpu_train_batch_size 50 --per_gpu_eval_batch_size 50 --output_dir /cache/output/models/bart/ --result_dir /cache/output/result/reddit3/ --model_name_or_path /cache/pretrained_model/BARTBase/ --config_name /cache/pretrained_model/BARTBase/ --tokenizer_name /cache/pretrained_model/BARTBase/ --learning_rate 3e-5 --num_train_epochs 3")

mox.file.copy_parallel("/cache/output", output_path)
print("write success")

os.system("chmod -R 777 /opt/huawei/schedule-train/")