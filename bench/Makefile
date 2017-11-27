STDOUT := $(wildcard *.stdout)
STDERR := $(wildcard *.stderr)
JOBLOG := $(wildcard *JOB.md)
DATE=$(shell date +%Y%m%d)
SLURM_JOBS := $(shell squeue -h -u $(USER) | awk '{print $$1}') 


archive:
	mkdir -p archive_$(DATE)
	@mv $(JOBLOG) $(STDOUT) $(STDERR) archive_$(DATE)

checkout:
	vim -p $(STDOUT)

checkerr:
	vim -p $(STDERR)

clean:
	@- $(RM) $(STDOUT)
	@- $(RM) $(STDERR)

cleanall: clean
	@- $(RM) $(JOBLOG)

squeue:
	@echo $(SLURM_JOBS)

scancel:
	scancel $(SLURM_JOBS)
