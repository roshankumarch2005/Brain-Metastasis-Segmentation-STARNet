#!/usr/bin/env bash
# ==============================================================================
# safe_cleanup_dgx.sh
# ------------------------------------------------------------------------------
# Safe, zero-wildcard file-by-file cleanup script for DGX workstation.
# Designed with strict fail-safe guards to guarantee that:
# 1. It only executes inside /Data3/EE23B009/EE23B009_BTP.
# 2. Critical project sentinels (dataset, environment, source code) MUST exist.
# 3. Only explicitly enumerated temporary/redundant files are deleted.
# 4. Zero impact on any other user or system directory on DGX.
# ==============================================================================

set -euo pipefail

EXPECTED_DIR="/Data3/EE23B009/EE23B009_BTP"
CURRENT_DIR=$(pwd -P)

echo "=========================================================================="
echo "          DGX WORKSPACE SAFE FILE-BY-FILE CLEANUP VERIFICATION            "
echo "=========================================================================="

# ------------------------------------------------------------------------------
# GUARD 1: Exact Working Directory Enforcement
# ------------------------------------------------------------------------------
if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
    echo "❌ FATAL SECURITY ERROR: Incorrect directory!"
    echo "   Expected Directory : $EXPECTED_DIR"
    echo "   Current Directory  : $CURRENT_DIR"
    echo "   ABORTING IMMEDIATELY to prevent accidental deletion of other folders."
    exit 1
fi
echo "✅ Guard 1 Passed: Executing inside $EXPECTED_DIR"

# ------------------------------------------------------------------------------
# GUARD 2: Critical Sentinel Verification (These MUST exist or script aborts)
# ------------------------------------------------------------------------------
REQUIRED_SENTINELS=(
    "nnunet_data"
    "nnunet_data/nnUNet_raw"
    "nnunet_data/nnUNet_preprocessed"
    "src"
    "smp_env"
    "checkpoints"
    "train.py"
    "prepare_nnunet_data.py"
    "TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx"
)

for sentinel in "${REQUIRED_SENTINELS[@]}"; do
    if [ ! -e "$sentinel" ]; then
        echo "❌ FATAL SAFETY ERROR: Critical sentinel '$sentinel' not found in $(pwd)!"
        echo "   Workspace state is unexpected. ABORTING without modifying anything."
        exit 1
    fi
done
echo "✅ Guard 2 Passed: All critical sentinels verified (Dataset, Code, & Environment intact)."

# ------------------------------------------------------------------------------
# GUARD 3: Strict Non-Deletion Whitelist (Never delete these under any circumstances)
# ------------------------------------------------------------------------------
PROTECTED_ITEMS=(
    "nnunet_data"
    "src"
    "smp_env"
    "checkpoints"
    "training_logs"
    "results"
    "dgx_wheels"
    "train.py"
    "prepare_nnunet_data.py"
    "prepare_dataset100_fast.py"
    "prepare_saluja_dataset.py"
    "prepare_saluja_restored_dataset.py"
    "prepare_nnunet_test.py"
    "restore_mri.py"
    "batch_restore_mri.py"
    "TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx"
    "pull_results_from_local.sh"
    "push_to_local.sh"
    "start_dgx_binary_9_and_10.sh"
    "launch_dgx_binary_config9_depthwise_siam.sh"
    "launch_dgx_binary_config10_depthwise_baa.sh"
    "run_inference_and_eval_dgx_finished.sh"
    "safe_cleanup_dgx.sh"
)

# ------------------------------------------------------------------------------
# Explicit List of Files to Safely Delete (Zero Wildcards)
# ------------------------------------------------------------------------------
EXPLICIT_FILES_TO_DELETE=(
    # Category A: Redundant Documentation, Diagrams, & Presentations (Saved Locally)
    "3D_PlainConvUNet_Architecture_Diagram.md"
    "BAA_Loss_Comprehensive_Guide.md"
    "BAA_Loss_Implementation.md"
    "BTP_Architecture_Exact_6Stages.png"
    "BTP_Complete_Pipeline_and_Architecture_Master.png"
    "BTP_Final_Report.md"
    "BTP_Progress_Report.pdf"
    "BTP_SIAM_Integration_Report.md"
    "Literature_vs_Our_BTP_Results_Comparison.md"
    "MRI_Restoration_Methodology.pdf"
    "MRI_Restoration_N4_and_NLM_Denoising.md"
    "Methods_1_2_3_Architecture_and_Ablation_Details.md"
    "Stage3_Architecture.md"
    "UCSF_BMSR_Literature_Review.md"
    "UCSF_BMSR_Literature_Review_and_Benchmarks.md"
    "UCSF_BMSR_Paper_Rudie_Saluja_2024.pdf"
    "complete_siam_workflow.md"
    "model_comparison.md"
    "nnU-Net_Architecture_and_Training_Details.pdf"
    "nnU-Net_Architecture_and_Training_Details.pptx"
    "nnUNet_Revisited_Paper_Deep_Dive.md"
    "plainconvunet_3d_architecture.png"
    "siam_architecture.md"
    "siam_architecture.txt"
    "siam_pytorch_output.md"
    "siam_pytorch_output.txt"
    "standard_nnunet_architecture.md"
    "unet_architecture.md"

    # Category B: HTML Dashboards & Visualization Helpers (Saved Locally)
    "build_complete_dashboard.py"
    "compile_btp_report_pdf.py"
    "generate_100325a_comparison_htmls.py"
    "generate_acdc_3d_htmls.py"
    "generate_acdc_alpha08_3d.py"
    "generate_acdc_baa_3d.py"
    "generate_all_htmls.py"
    "generate_architecture_diagram.py"
    "generate_b0_snapshot_100362A.py"
    "generate_brain_dashboard_models.py"
    "generate_curriculum_worst_cases_proof.py"
    "generate_gt_only_100362A.py"
    "generate_method1_diagram.py"
    "generate_method1_head_to_head_case.py"
    "generate_method2_diagram.py"
    "generate_method2_method3_htmls.py"
    "generate_method3_diagram.py"
    "generate_multiclass_curriculum_htmls.py"
    "generate_multiclass_m10_m9_htmls.py"
    "generate_multiclass_method3_htmls.py"
    "generate_nnunet_architecture_deck.py"
    "generate_nnunet_figures.py"
    "generate_presentation.py"
    "generate_saluja_b0_htmls.py"
    "generate_saluja_b10_htmls.py"
    "generate_saluja_b11_htmls.py"
    "generate_saluja_b1_htmls.py"
    "generate_saluja_b3_htmls.py"
    "generate_saluja_b5_htmls.py"
    "generate_saluja_b6_htmls.py"
    "generate_saluja_curriculum_htmls.py"
    "make_complete_master_diagram.py"
    "make_exact_architecture_diagram.py"
    "plot_patient_lesion_distribution.py"
    "plot_segmentation_volume_histogram.py"
    "plot_simple_histograms.py"
    "rebuild_index.py"
    "update_100362a_b0.py"
    "visualize.py"
    "visualize_all_classes.py"
    "visualize_baa.py"
    "visualize_val.py"
    "index.html"

    # Category C: Redundant Test Metric CSVs (Evaluated and Saved Locally)
    "UCSF_BMSR_Benchmark_Table.csv"
    "brats_boundary_metrics_report.csv"
    "brats_boundary_metrics_report_Alpha08.csv"
    "brats_boundary_metrics_report_BAA.csv"
    "config10_depthwise_baa_test30_metrics.csv"
    "config13_depthwise_method1_siam_test30_metrics.csv"
    "config3_base_baa_test30_metrics.csv"
    "final_test_benchmark_comparison.csv"
    "method2_decoder1_halfres_multiclass_test30_metrics.csv"
    "method3_depthwise_separable_multiclass_test30_metrics.csv"
    "method3_siam_boundary_alpha08_binary_test30_metrics.csv"
    "method3_siam_curriculum_multiclass_test30_metrics.csv"

    # Category D: Old Standalone Evaluation & Helper Scripts (Tracked in Git/Local)
    "auto_run_b3_pipeline.py"
    "check_model_params.py"
    "convert_json_v2_to_v1.py"
    "evaluate.py"
    "evaluate_acdc_alpha08.py"
    "evaluate_acdc_baa.py"
    "evaluate_acdc_baseline.py"
    "evaluate_acdc_siam.py"
    "evaluate_all_metrics.py"
    "evaluate_boundaries_to_csv.py"
    "evaluate_boundary_baa.py"
    "evaluate_depthwise_multiclass_test30.py"
    "evaluate_ensemble.py"
    "evaluate_m5_raw.py"
    "evaluate_multiclass_curriculum.py"
    "evaluate_multiclass_test30_general.py"
    "evaluate_nnunet.py"
    "evaluate_saluja_test30.py"
    "evaluate_segresnet.py"
    "evaluate_sens_prec.py"
    "evaluate_siam.py"
    "evaluate_smp.py"
    "fix_geometry.py"
    "fix_json_for_v1.py"
    "model_test.py"
    "organize_splits.py"
    "postprocess_cca.py"
    "preprocess_2d.py"
    "run_b2_pipeline.py"
    "run_custom_ensemble.py"
    "run_saluja_ensemble_b1_b3.py"
    "train_segresnet.py"
    "train_smp.py"
    "verify_all_test_metrics.py"

    # Category E: Root-level Old Logs & Dumps
    "boundary_train_fixed_log.txt"
    "boundary_train_log.txt"
    "curriculum_baa_test_metrics.txt"
    "curriculum_binary_run.log"
    "eval_depthwise_multiclass.log"
    "eval_interlevel_siam_baa15.log"
    "interlevel_siam_baa15_test_metrics.txt"
    "interlevel_siam_boundary_alpha08_test_metrics.txt"
    "logout"
    "predict.log"
    "predict_siam.log"
    "saluja_baseline_train.log"
    "saluja_siam_baa15_train.log"
    "scratch_tmux_log.txt"
    "test_fix.pkl"
    "tmux_log.txt"
    "train_method1_all_multiclass.log"
    "train_method1_siam_boundary_binary.log"
    "train_method1_siam_curriculum_multiclass.log"
    "train_method1_siam_multiclass.log"
    "train_method2_decoder1.log"
    "train_method2_multiclass_local_gpu0.log"
    "train_method3_depthwise.log"
    "training_logs_interlevel_all.log"
    "training_logs_interlevel_all_multiclass.log"
    "training_logs_interlevel_bottleneck.log"
    "training_logs_method3_depthwise.log"
)

# Explicit List of Directories to Safely Remove
EXPLICIT_DIRS_TO_DELETE=(
    "baselines_backup"
    "scratch"
    "data_downloads"
    "__pycache__"
)

# ------------------------------------------------------------------------------
# Confirmation Prompt
# ------------------------------------------------------------------------------
echo ""
echo "Found ${#EXPLICIT_FILES_TO_DELETE[@]} explicit files and ${#EXPLICIT_DIRS_TO_DELETE[@]} directories queued for cleanup."
echo "Core datasets, models, code, and other user directories will NOT be touched."
echo ""
read -p "Type 'DELETE' to confirm safe cleanup of these specific files: " CONFIRM

if [ "$CONFIRM" != "DELETE" ]; then
    echo "Cleanup cancelled by user. Nothing was modified."
    exit 0
fi

echo ""
echo "Proceeding with safe removal..."

# ------------------------------------------------------------------------------
# Safe File Removal Loop
# ------------------------------------------------------------------------------
DELETED_FILES_COUNT=0
for target_file in "${EXPLICIT_FILES_TO_DELETE[@]}"; do
    # Cross-check against protected whitelist
    for protected in "${PROTECTED_ITEMS[@]}"; do
        if [ "$target_file" == "$protected" ]; then
            echo "⚠️ SKIPPING PROTECTED ITEM: $target_file"
            continue 2
        fi
    done

    # Remove only if it exists
    if [ -f "$target_file" ] || [ -L "$target_file" ]; then
        rm -f "$target_file"
        echo "  [REMOVED FILE] $target_file"
        DELETED_FILES_COUNT=$((DELETED_FILES_COUNT + 1))
    fi
done

# ------------------------------------------------------------------------------
# Safe Directory Removal Loop
# ------------------------------------------------------------------------------
DELETED_DIRS_COUNT=0
for target_dir in "${EXPLICIT_DIRS_TO_DELETE[@]}"; do
    # Cross-check against protected whitelist
    for protected in "${PROTECTED_ITEMS[@]}"; do
        if [ "$target_dir" == "$protected" ]; then
            echo "⚠️ SKIPPING PROTECTED DIRECTORY: $target_dir"
            continue 2
        fi
    done

    if [ -d "$target_dir" ]; then
        rm -rf "$target_dir"
        echo "  [REMOVED DIR]  $target_dir/"
        DELETED_DIRS_COUNT=$((DELETED_DIRS_COUNT + 1))
    fi
done

# ------------------------------------------------------------------------------
# Summary Verification
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo "✅ Safe cleanup complete!"
echo "   Files removed       : $DELETED_FILES_COUNT"
echo "   Directories removed : $DELETED_DIRS_COUNT"
echo "=========================================================================="
echo ""
echo "Verifying remaining workspace contents:"
ls -F
