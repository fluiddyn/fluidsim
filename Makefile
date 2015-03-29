
clean_so:
	find fluidsim -name "*.so" -delete

tests:
	python -m unittest discover
