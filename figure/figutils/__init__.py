"""
figutils - Composite Delphi figure utilities
=============================================
Visualization modules (renamed from 'utils' to avoid collision
with the project-root utils.py used by the model).

Modules:
  common       - Model/data loading, DataLoader creation
  performance  - Shift/Total prediction analysis (classification + regression)
  calibration  - ROC-AUC, AP, calibration curves (Delphi Fig 2 style)
  trajectory   - Rate-vs-age plots, synthetic trajectory generation
  event        - Waiting time prediction vs observation
  boxplot      - AUC boxplot/scatter by ICD-10 chapter
  umap_viz     - UMAP embedding visualization
  shap_viz     - SHAP interaction heatmaps
  waterfall    - Custom SHAP waterfall plot (from Delphi)
  xai          - Attention patterns, feature importance
"""
