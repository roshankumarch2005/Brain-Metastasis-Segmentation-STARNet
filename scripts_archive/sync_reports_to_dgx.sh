#!/bin/bash
# sync_reports_to_dgx.sh
# ------------------------
# Syncs updated BTP Final Report, Literature Comparison, and CSV benchmarks to DGX

DGX_HOST="user@10.23.20.28"
DGX_DIR="/Data3/EE23B009/EE23B009_BTP"

echo "=================================================================="
echo " 🔄 Syncing Reports, CSVs, and 3D Visualizers to DGX ($DGX_HOST)"
echo "=================================================================="

# 1. Sync Markdown reports, CSVs, and evaluation scripts
echo -e "\n[1/2] Syncing Markdown reports and CSV benchmark files..."
rsync -avP /mnt/DATA/EE23B009/EE23B009_BTP/*.md \
            /mnt/DATA/EE23B009/EE23B009_BTP/*.csv \
            /mnt/DATA/EE23B009/EE23B009_BTP/evaluate_saluja_test30.py \
            /mnt/DATA/EE23B009/EE23B009_BTP/generate_saluja_method3_b12_htmls.py \
            $DGX_HOST:$DGX_DIR/

# 2. Sync master dashboard and B12 3D HTML visualizations
echo -e "\n[2/2] Syncing results/index.html and Model B12 3D HTMLs..."
mkdir -p /mnt/DATA/EE23B009/EE23B009_BTP/results/figures_saluja_method3_siam_boundary_alpha08_html
rsync -avP /mnt/DATA/EE23B009/EE23B009_BTP/results/index.html \
            $DGX_HOST:$DGX_DIR/results/

rsync -avP /mnt/DATA/EE23B009/EE23B009_BTP/results/figures_saluja_method3_siam_boundary_alpha08_html/ \
            $DGX_HOST:$DGX_DIR/results/figures_saluja_method3_siam_boundary_alpha08_html/

echo ""
echo "=================================================================="
echo " [✓] Reports, CSVs, and 3D visualizers successfully synced to DGX!"
echo "=================================================================="

