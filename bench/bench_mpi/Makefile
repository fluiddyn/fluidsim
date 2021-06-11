
bench_py:
	mpirun -np 2 python bench_point2point.py

bench_cpp: a.out
	mpirun -np 2 ./a.out

a.out: bench_point2point.c
	mpicxx bench_point2point.c -O3 -lrt

clean:
	rm -f *.out

bench_ib.out: bench_point2point.c
	module purge
	module load openmpi/4.0.5-ib
	mpicxx bench_point2point.c -O3 -lrt -o bench_ib.out

bench_no_ib.out: bench_point2point.c
	module purge
	mpicxx bench_point2point.c -O3 -lrt -o bench_no_ib.out
