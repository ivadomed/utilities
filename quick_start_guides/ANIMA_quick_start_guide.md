# Compute ANIMA metrics

For MS and SCI lesion segmentation tasks, you can compute ANIMA metrics using the [compute_anima_metrics.py](https://github.com/ivadomed/model_seg_sci/blob/main/testing/compute_anima_metrics.py) script.

Installation

```
cd ~
mkdir anima/
cd anima/
wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.2/Anima-Ubuntu-4.2.zip.   # (change version to latest)
unzip Anima-Ubuntu-4.2.zip
rm Anima-Ubuntu-4.2.zip
git lfs install
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Public.git
git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git
```

Configure directories

```
cd ~
mkdir .anima/
touch .anima/config.txt

echo "[anima-scripts]" >> .anima/config.txt
echo "anima = ${HOME}/anima/Anima-Binaries-4.2/" >> .anima/config.txt
echo "anima-scripts-public-root = ${HOME}/anima/Anima-Scripts-Public/" >> .anima/config.txt
echo "extra-data-root = ${HOME}/anima/Anima-Scripts-Data-Public/" >> .anima/config.txt
```

Run the script

```
python compute_anima_metrics.py --pred_folder <path_to_predictions_folder>  --gt_folder <path_to_gt_folder> -dname <dataset_name>
```
