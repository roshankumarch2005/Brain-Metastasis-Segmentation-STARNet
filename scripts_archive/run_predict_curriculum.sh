#!/bin/bash
set -e
export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
export CUDA_VISIBLE_DEVICES=1

mkdir -p /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/predictions_test_curriculum_baa_final

/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/nnUNetv2_predict \
 -i /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw/Dataset105_UCSF_Saluja_Exact/imagesTs \
 -o /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/predictions_test_curriculum_baa_final \
 -d 105 -c 3d_fullres -f 0 -tr nnUNetTrainer_InterLevel_SIAM_CurriculumBAA

echo "Prediction complete! Evaluating metrics..."
/mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/python3 /mnt/DATA/EE23B009/EE23B009_BTP/evaluate_saluja_test30.py /mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/Dataset105_UCSF_Saluja_Exact/predictions_test_curriculum_baa_final > /mnt/DATA/EE23B009/EE23B009_BTP/curriculum_baa_test_metrics.txt

echo "ALL DONE!"
