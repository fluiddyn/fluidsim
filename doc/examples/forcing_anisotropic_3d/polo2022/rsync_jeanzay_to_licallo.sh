#!/usr/bin/env bash

rsync -rvz -L --update --progress \
  uey73qw@jean-zay.idris.fr:/gpfswork/rech/uzc/uey73qw/aniso/ns3d.strat_polo*1280x1280* \
  /scratch/vlabarre/aniso/from_jeanzay/ \
  --exclude "*rank*_tmin*.h5" \
  --exclude "*state_phys_t*.h5" \
  --exclude "aniso/*/State_phys_*" \
  --exclude "aniso/end_states/*"

rsync -rvz -L --update --progress \
  uey73qw@jean-zay.idris.fr:/gpfswork/rech/uzc/uey73qw/aniso/ns3d.strat_polo*1920x1920* \
  /scratch/vlabarre/aniso/from_jeanzay/ \
  --exclude "*rank*_tmin*.h5" \
  --exclude "*state_phys_t*.h5" \
  --exclude "aniso/*/State_phys_*" \
  --exclude "aniso/end_states/*"

rsync -rvz -L --update --progress \
  uey73qw@jean-zay.idris.fr:/gpfswork/rech/uzc/uey73qw/aniso/ns3d.strat_polo*2560x2560* \
  /scratch/vlabarre/aniso/from_jeanzay/ \
  --exclude "*rank*_tmin*.h5" \
  --exclude "*state_phys_t*.h5" \
  --exclude "aniso/*/State_phys_*" \
  --exclude "aniso/end_states/*"

rsync -rvz -L --progress --update \
  uey73qw@jean-zay.idris.fr:/gpfswork/rech/uzc/uey73qw/aniso/end_states \
  uey73qw@jean-zay.idris.fr:/gpfswork/rech/uzc/uey73qw/aniso/results_papermill \
  /scratch/vlabarre/aniso/from_jeanzay/
