# Vertebral labeling with SPINEPS start guide

This file provides a start guide to use [SPINEPS](https://github.com/Hendrik-code/spineps) for vertebral labeling. 

SPINEPS is an automatic method for vertebral labeling of the spine on T1w and T2w MRI scans. Because SPINEPS is not doing labeling of the different vertebrae and discs, the label provided by the segmentations and the discs labels will not be accurate if the first vertebrae C2 does not appear in the image.

## Installation

### Discs labeling installation

1. First, create a conda environment with python 3.11.

```bash
conda create --name spineps python=3.11
conda activate spineps
conda install pip
```

2. Clone this repository
```bash
git clone git@github.com:ivadomed/utilities.git
```

3. Run this command or add it directly to your `.bashrc` or `.zshrc` (generally at the root of your home folder)
```bash
export IVADOMED_UTILITIES_REPO=<PATH-to-UTILITIES>
```

> You can also run `export IVADOMED_UTILITIES_REPO="$(pwd)/utilities"` after the last step


### SPINEPS installation

Then, you need to install [SPINEPS](https://github.com/Hendrik-code/spineps) in the same virtual environment.

> **Note**
> For this step you can also follow the official installation [here](https://github.com/Hendrik-code/spineps#installation-ubuntu)

1. Install the correct version of [pytorch](https://pytorch.org/get-started/locally/) in you environment.

2. Confirm that your pytorch package is working! Try calling these command:
```bash
nvidia-smi 
```
This should show the usage of your GPUs return `True`.
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
This should return `True`.

3. Clone spineps and install it:
> Note: You can also clone using the http adress: https://github.com/Hendrik-code/spineps.git
```bash
git clone git@github.com:Hendrik-code/spineps.git
cd spineps
pip install -e .
```

4. Dowload spineps' first weights [Inst_Vertebra_3.0.zip](https://syncandshare.lrz.de/dl/fi16bYYmqpwPQZRGd1M4G6/Inst_Vertebra_3.0.zip)

5. Then download for:
    - T2w labeling [T2w_Segmentor_2.0.zip](https://syncandshare.lrz.de/dl/fi16bYYmqpwPQZRGd1M4G6/T2w_Segmentor_2.0.zip)
    - T1w labeling [T1w_Segmentor.zip](https://syncandshare.lrz.de/dl/fi16bYYmqpwPQZRGd1M4G6/T1w_Segmentor.zip)

6. Make a directory and move all the weights in the folder `<PATH-to-SPINEPS>/spineps/models`:
```bash
mkdir spineps/models
cd ..
```

7. Run this command or add it directly to your `.bashrc` or `.zshrc` (generally at the root of your home folder)
```bash
export SPINEPS_SEGMENTOR_MODELS=<PATH-to-SPINEPS>/spineps/models
```

## Discs labeling function

After the installation, you should have 2 repositories in your current folder (`spineps` and `utilities`):
```bash
ls
```

Finally to compute discs labeling with SPINEPS, you need to add this function to any bash script:
```bash
label_with_spineps(){
    local img_path=$(realpath "$1")
    local out_path="$2"
    local contrast="$3"
    local img_name="$(basename "$img_path")"
    (
        # Create temporary directory
        tmpdir="$(mktemp -d)"
        echo "$tmpdir" was created

        # Copy image to temporary directory
        tmp_img_path="${tmpdir}/${img_name}"
        cp "$img_path" "$tmp_img_path"

        # Activate conda env
        eval "$(conda shell.bash hook)"
        conda activate spineps

        # Select semantic weights
        if [ "$contrast" = "t1" ];
            then semantic=t1w_segmentor;
            else semantic=t2w_segmentor_2.0;
        fi
        
        # Run SPINEPS on image with GPU
        spineps sample -i "$tmp_img_path" -model_semantic "$semantic" -model_instance inst_vertebra_3.0 -dn derivatives -iic
        # Run SPINEPS on image with CPU
        # spineps sample -i "$tmp_img_path" -model_semantic "$semantic" -model_instance inst_vertebra_3.0 -dn derivatives -cpu -iic
        
        # Run vertebral labeling with SPINEPS vertebrae prediction
        vert_path="$(echo ${tmpdir}/derivatives/*_seg-vert_msk.nii.gz)"
        python3 "${IVADOMED_UTILITIES_REPO}/training_scripts/generate_discs_labels_with_SPINEPS.py" --path-vert "$vert_path" --path-out "$out_path"

        # Remove temporary directory
        rm -r "$tmpdir"
        echo "$tmpdir" was removed

        # Deactivate conda environment
        conda deactivate
    )
}
```

## Usage

You can now call this function:
```bash
label_with_spineps "$IMG_PATH" "$OUT_PATH" "$CONTRAST"
```
With:
- `IMG_PATH` corresponding to the path to your input image
- `OUT_PATH` corresponding to the discs labels output path
- `CONTRAST` corresponding to the input contrast (`t1` or `t2`)



