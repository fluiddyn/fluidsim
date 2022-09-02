#!/usr/bin/env bash

# Note: execute compress.py before this script
# see https://fluiddyn.readthedocs.io/en/latest/workflow-data-figs.html

#!/usr/bin/env bash

PROJECT_DIR=$PROJECT_DIR
MyCore_DIR=...

rsync -aP --progress --update \
  $PROJECT_DIR/results_papermill/* \
  $PROJECT_DIR/from_jeanzay/aniso/results_papermill/* \
  $MyCore_DIR/2022strat-turb-polo/notebooks

rsync -aP --progress --update \
  $PROJECT_DIR/aniso/ns3d* \
  $PROJECT_DIR/from_jeanzay/aniso/ns3d* \
  $MyCore_DIR/2022strat-turb-polo/simul_folders \
  --exclude "**/spatiotemporal/rank*_tmin*.h5" \
  --exclude "**/state_phys_t*.h5" \
  --exclude "end_states/*" \
  --exclude "results_papermill/*" \
  --exclude "**/*_uncompressed.h5" \
  --exclude "*/State_phys_*"
