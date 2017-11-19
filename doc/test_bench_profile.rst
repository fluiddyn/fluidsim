Testing, benchmarks and profiling
=================================

Fluidsim comes with command-line tools for testing, benchmarking and
profiling. Here are useful commands:

.. code-block:: bash

   fluidsim -h

Testing
-------

.. code-block:: bash

   fluidsim-test -h
   fluidsim-test -v

Benchmarks
----------

.. code-block:: bash

   fluidsim-bench -h
   fluidsim-bench -s ns2d

.. code-block:: bash

   fluidsim-bench-analysis -h


Profiling
---------

.. code-block:: bash

   fluidsim-profile -h
   fluidsim-profile -s ns2d
   fluidsim-profile -p -sf /tmp/fluidsim_profile/result_bench_ns2d_512x512_2017-11-19_22-19-26_8772.json
