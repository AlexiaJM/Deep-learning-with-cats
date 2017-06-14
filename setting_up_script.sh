## First time installs
#pip install tensorboard_logger
#pip install opencv-python

## Download CAT dataset
wget -nc https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/CAT_DATASET_01.zip
wget -nc https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/CAT_DATASET_02.zip
wget -nc https://web.archive.org/web/20150703060412/http://137.189.35.203/WebUI/CatDatabase/Data/00000003_015.jpg.cat

## Setting up folder
unzip CAT_DATASET_01.zip -d cat_dataset
unzip CAT_DATASET_02.zip -d cat_dataset
mv cat_dataset/CAT_00/* cat_dataset
rmdir cat_dataset/CAT_00
mv cat_dataset/CAT_01/* cat_dataset
rmdir cat_dataset/CAT_01
mv cat_dataset/CAT_02/* cat_dataset
rmdir cat_dataset/CAT_02
mv cat_dataset/CAT_03/* cat_dataset
rmdir cat_dataset/CAT_03
mv cat_dataset/CAT_04/* cat_dataset
rmdir cat_dataset/CAT_04
mv cat_dataset/CAT_05/* cat_dataset
rmdir cat_dataset/CAT_05
mv cat_dataset/CAT_06/* cat_dataset
rmdir cat_dataset/CAT_06

## Error correction
rm cat_dataset/00000003_019.jpg.cat
mv 00000003_015.jpg.cat cat_dataset/00000003_015.jpg.cat

## Preprocessing
mkdir cat_dataset_output
wget -nc https://raw.githubusercontent.com/AlexiaJM/Deep-learning-with-cats/master/preprocess_cat_dataset.py
python preprocess_cat_dataset.py
