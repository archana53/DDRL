# Download AFLW\ dataset.zip https://drive.google.com/uc?id=1REnIf-H9uZsIcn8MO9tb835pLD4_Ycb6
unzip AFLW\ dataset.zip
cd AFLW\ dataset
tar -xvzf aflw-images-0.tar.gz
tar -xvzf aflw-images-2.tar.gz
tar -xvzf aflw-images-3.tar.gz
mkdir -p images/0 images/2 images/3
mv aflw/data/flickr/0 images
mv aflw/data/flickr/2 images
mv aflw/data/flickr/3 images
mkdir AFLW_lists
cd ..
mv AFLW\ dataset aflw_dataset
python preprocess_aflw.py --root_dir "/content/DDRL/modules/data/aflw_dataset"