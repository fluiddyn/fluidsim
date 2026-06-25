
PATH_SIMUL ?= /fsnet/project/meige/2026/26KOLMO/from_legi/strat/ns3d.strat_N10_kmax_eta0_640x640x640_V2pix2pix2pi_fNone_N10_2026-06-10_18-25-13
OUTPUT ?= figures.pdf

figures.pdf:
	PATH_SIMUL_DIR="$(PATH_SIMUL)" myst build --execute --typst notebook.md -o $(OUTPUT)

cleanall:
	rm -f *.pdf
