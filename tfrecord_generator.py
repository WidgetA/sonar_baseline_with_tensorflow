import argparse
import os
import sys
import zipfile
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-path", help="Specify the absolute path to the compressed file")
args = parser.parse_args()

if not hasattr(args, 'path'):
    print("Please specify the absolute path with \"-path\"")
else:
    os.chdir(sys.path[0])
    f = zipfile.ZipFile(args.path)
    target_path = os.path.join(sys.path[0], 'dataset/raw_data')
    for file in f.namelist():
        filename = file.encode('cp437').decode('gbk')
        f.extract(file, r'./dataset/raw_data')
        os.rename(os.path.join(target_path, file), os.path.join(target_path, filename))

    trainset_path = os.path.join(target_path, 'train')
    for folder in os.listdir(trainset_path):
        if folder not in ['侧扫声呐图像', '负样本', '前视声呐图像']:
            if os.path.isdir(os.path.join(trainset_path, folder)):
                shutil.rmtree(os.path.join(trainset_path, folder))
            else:
                pass
        else:
            continue

    os.rename(os.path.join(trainset_path, '侧扫声呐图像'), os.path.join(trainset_path, 'cesao'))
    os.rename(os.path.join(trainset_path, '负样本'), os.path.join(trainset_path, 'fu'))
    os.rename(os.path.join(trainset_path, '前视声呐图像'), os.path.join(trainset_path, 'qianshi'))

    

