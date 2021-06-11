
bench_py:
	mpirun -np 2 python bench_point2point.py

bench_cpp: a.out
	mpirun -np 2 ./a.out

a.out: bench_point2point.c
	mpicc bench_point2point.c -O3

clean:
	rm -f *.out