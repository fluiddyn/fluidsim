#!/usr/bin/env bash

rsync -rvz -L \
  augier@occigen.cines.fr:/scratch/cnt0022/egi2153/augier/aniso/end_states \
  /fsnet/project/meige/2022/22STRATURBANIS/from_occigen

du -h /fsnet/project/meige/2022/22STRATURBANIS/from_occigen/end_states
