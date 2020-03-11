import argparse
import os
import sys
import zipfile
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument("-path", help="Specify the absolute path to the compressed file")
args = parser.parse_args()


def listdir_nohidden(path, tp):
    for f in os.listdir(path):
        if f[-4:] == '.' + tp:
            yield f


def validate_set(srcimg_path, src_label_path, tarimg_path, tarlabel_path, rate):
    file_list = os.listdir(srcimg_path)
    file_number = len(file_list)
    pick_number = int(file_number * rate)
    sample = random.sample(file_list, pick_number)

    for name in sample:
        if name[-3:] in ['jpg', 'xml']:
            name = name[:-4]
            shutil.copy(os.path.join(srcimg_path, name + '.jpg'), os.path.join(tarimg_path, name + '.jpg'))
            shutil.copy(os.path.join(src_label_path, name + '.xml'), os.path.join(tarlabel_path, name + '.xml'))
        else:
            continue


def gen_list(img_path, list_path, tp):
    img_list = list(listdir_nohidden(img_path, tp))
    with open(os.path.join(list_path, 'train.txt'), 'w+') as f:
        for i in img_list:
            f.write(i[:-4] + '\n')
        f.close()


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

    for folder in [os.path.join(trainset_path, 'cesao'), os.path.join(trainset_path, 'qianshi')]:
        for i, sub_folder in enumerate([os.path.join(folder, 'box'), os.path.join(folder, 'image')]):
            if i is 0:
                for file in list(listdir_nohidden(sub_folder, 'xml')):
                    shutil.copy(os.path.join(sub_folder, file), os.path.join(sys.path[0], 'dataset/train/xml'))
            else:
                for file in list(listdir_nohidden(sub_folder, 'jpg')):
                    shutil.copy(os.path.join(sub_folder, file), os.path.join(sys.path[0], 'dataset/train/jpg'))

    validate_set('./dataset/train/jpg', './dataset/train/xml', './dataset/validate/jpg', './dataset/validate/xml', 0.05)

    gen_list('./dataset/train/jpg', './dataset/train', 'jpg')
    gen_list('./dataset/validate/jpg', './dataset/validate', 'jpg')

    label_map_path = 'tf_datatools/pascal_label_map.pbtxt'
    train_img = 'dataset/train/jpg'
    train_xml = 'dataset/train/xml'
    train_set = 'dataset/train/train.txt'

    val_img = 'dataset/validate/jpg'
    val_xml = 'dataset/validate/xml'
    val_set = 'dataset/validate/train.txt'

    os.system(
        f'python tf_datatools/create_pascal_tf_record.py --label_map_path={label_map_path} \
        --annotations_dir={train_xml} \
        --data_dir={train_img} \
        --set={train_set} \
        --output_path=dataset/tfrecords/pascal_train.record')

    os.system(
        f'python tf_datatools/create_pascal_tf_record.py --label_map_path={label_map_path} \
            --annotations_dir={val_xml} \
            --data_dir={val_img} \
            --set={val_set} \
            --output_path=dataset/tfrecords/pascal_val.record')