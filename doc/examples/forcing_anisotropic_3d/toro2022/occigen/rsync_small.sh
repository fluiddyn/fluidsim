#!/usr/bin/env bash

rsync -rvz \
  augier@occigen.cines.fr:/scratch/cnt0022/egi2153/augier/aniso \
  /fsnet/project/meige/2022/22STRATURBANIS/from_occigen \
  --exclude "aniso/*/spatiotemporal/rank*_tmin*.h5" \
  --exclude "aniso/*/state_phys_t*.h5" \
  --exclude "aniso/*/State_phys_*/state_phys_t*.h5" \
  --exclude "aniso/*/State_phys_*" \
  --exclude "aniso/end_states/*"
