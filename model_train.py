import os
import shutil
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-path", help="Specify the absolute path to the compressed file")
args = parser.parse_args()

if not hasattr(args, 'path'):
    print("Please specify the absolute path with \"-path\"")
else:
    os.chdir(sys.path[0])

    if len(os.listdir('model_file/pretrain')) > 1:
        pass
    else:
        shutil.copy(args.path, 'model_file/pretrain')
    os.system('pip install -e slim')
    os.system('python object_detection/legacy/train.py --train_dir=model_file/train ' +
              '--pipeline_config_path=ssd_resnet50_v1_fpn.config')