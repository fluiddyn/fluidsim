#!/usr/bin/env bash

rsync -rv \
  /fsnet/project/meige/2022/22STRATURBANIS/results_papermill/* \
  /fsnet/project/meige/2022/22STRATURBANIS/from_occigen/aniso/results_papermill/* \
  /fsnet/project/meige/2013/13MUST/MyCore/2022strat-turb-toro/notebooks

rsync -rv \
  /fsnet/project/meige/2022/22STRATURBANIS/aniso/ns3d* \
  /fsnet/project/meige/2022/22STRATURBANIS/from_occigen/aniso/ns3d* \
  /fsnet/project/meige/2013/13MUST/MyCore/2022strat-turb-toro/simul_folders \
  --exclude "**/spatiotemporal/rank*_tmin*.h5" \
  --exclude "**/state_phys_t*.h5" \
  --exclude "end_states/*" \
  --exclude "results_papermill/*" \
  --exclude "**/*_uncompressed.h5" \
  --exclude "*/State_phys_*"

