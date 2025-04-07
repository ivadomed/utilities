#!/bin/bash
# Training nnUNet on a dataset

# define arguments for nnUNet
dataset_num="XXX"
dataset_name="Dataset${dataset_num}_<your-dataset-name>"
# nnunet_trainer="nnUNetTrainerDiceCELoss_noSmooth"         # default: nnUNetTrainer or nnUNetTrainer_2000epochs
nnunet_trainer="nnUNetTrainerDA5_DiceCELoss_noSmooth"       # custom trainer
# nnunet_trainer="nnUNetTrainer_5epochs"                    # options: nnUNetTrainer_1epoch, nnUNetTrainer_5epochs

# model_type="M"                                              # options: "M" or "L" or "XL"
# nnunet_planner="nnUNetPlannerResEnc${model_type}"           # default: nnUNetPlannerResEncM/L
nnunet_plans_file="nnUNetPlans"

# configurations=("3d_fullres" "2d")                        # for 2D training, use "2d"
configurations=("3d_fullres")                               # for 2D training, use "2d"
cuda_visible_devices=0
fold=1

# define final variables where the data will be copied
# NOTE: this assumes that following folders exist at the defined paths -- update to match your folder structure
final_prepro_dir="/home/$(whoami)/projects/def-jcohen/$(whoami)/datasets/nnUNet_preprocessed"
final_results_dir="/home/$(whoami)/code/nnunet-v2/nnUNet_results"

echo $SLURM_TMPDIR

# NOTE: Compute Canada recommends moving data to $SLURM_TMPDIR because this folder has fast read/write ability
# which makes dataloading faster. Hence, we copy the datasets to $SLURM_TMPDIR before starting the training
echo "-------------------------------------------"
echo "Moving the dataset to SLURM_TMPDIR: ${SLURM_TMPDIR}"
echo "-------------------------------------------"

# create folders in SLURM_TMPDIR
if [[ ! -d $SLURM_TMPDIR/nnUNet_raw ]]; then
    mkdir $SLURM_TMPDIR/nnUNet_raw

    # copy the dataset to SLURM_TMPDIR
    cp -r /home/$(whoami)/projects/rrg-bengioy-ad/$(whoami)/datasets/nnUNet_raw/${dataset_name} ${SLURM_TMPDIR}/nnUNet_raw
fi

# create folders in SLURM_TMPDIR
mkdir $SLURM_TMPDIR/nnUNet_preprocessed
mkdir $SLURM_TMPDIR/nnUNet_results

# temporarily export the nnUNet environment variables (to make nnUNet happy)
export nnUNet_raw=$SLURM_TMPDIR/nnUNet_raw
export nnUNet_preprocessed=$SLURM_TMPDIR/nnUNet_preprocessed
export nnUNet_results=$SLURM_TMPDIR/nnUNet_results


echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"
nnUNetv2_plan_and_preprocess -d ${dataset_num} -c ${configurations} --verify_dataset_integrity 


for configuration in ${configurations[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold, Configuration $configuration"
    echo "-------------------------------------------"
    
    # training
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${dataset_num} $configuration $fold -tr ${nnunet_trainer} -p ${nnunet_plans_file}
    
    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    # run inference on test set
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs -tr ${nnunet_trainer} -p ${nnunet_plans_file} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__${nnunet_plans_file}__${configuration}/fold_${fold}/test -d ${dataset_num} -f $fold -c ${configuration} # -step_size 0.9 --disable_tta
    
done

echo ""
echo "--------------------------------------------------------------------------------------------"
echo "Testing done, Moving the results/preprocessed data from ${SLURM_TMPDIR} to the home directory"
echo "-----------------------------------------------------------------------------------------"


# copy the preprocessed data back to the home directory
cp -r ${SLURM_TMPDIR}/nnUNet_preprocessed/${dataset_name} ${final_prepro_dir}

# copy the results back to the home directory
cp -r ${SLURM_TMPDIR}/nnUNet_results/${dataset_name} ${final_results_dir}

echo "-------------------"
echo "Job Done!"
echo "-------------------"