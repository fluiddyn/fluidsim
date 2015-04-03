Tutorial: how to develop a new solver 
=====================================

A solver is defined by the equations that it solves, its numerical
methods, the possible initialization methods and the treatments
performed on the raw data produced.

Moreover, a solver can be decomposed in different parts that do
different tasks. In FluidSim, this principle is followed as much as
possible and a solver is organized in few big parts using Python
classes. Thus, a solver is defined by the classes used to performed to
different tasks. We here present the classes used for the simulations
in FluidSim.

The main class representing a simulation: Simul
-----------------------------------------------

The main classes representing a simulation are usually just called
"Simul" and thus are uniquely defined by the full path of the module
where they are defined as for example
:class:`fluiddyn.simul.solvers.ns2d.solver.Simul`. A Simul class
should inherit from
:class:`fluiddyn.simul.base.solvers.base.SimulBase` (or from a child
of this base class).

In :ref:`tutosimuluser`, we have instantiate the Simul class like
this::

  sim = solver.Simul(params)

The Simul class is the main class from which the solvers are
organized. In particular, the other big parts of a solver are
attributes of the object Simul. The class takes care of importing the
module where the other main classes are defined and it instantiates
them. This class also owns the important functions that implement the
equations and that are used by the other main classes (as
tendencies_nonlin, compute_freq_diss, compute_freq_complex,
operator_linear...).

The other classes responsible for the main tasks
------------------------------------------------

- Operators class (need only the parameters);

  The class Operators contains the information on the grid and
  functions to compute operators like the gradient, the curl, the
  divergence...

- State (need the parameters, the operators and the time stepping
  instances, so its ``__init__`` function takes as argument the
  simulation object);

  The class State contains the data (i.e. big arrays of numbers, more
  precisely
  :class:`fluiddyn.simul.operators.setofvariables.SetOfVariables`)
  representing the state of the simulation (for example the
  out-of-plane vorticity in spectral space for the pseudo-spectral
  ns2d solver). This class is also able to compute different
  quantities from the state, meaning that it knows about the
  relationships between the variables. The State classes inherit from
  the :class:`fluiddyn.simul.base.state.StateBase` class.

- InitState (the ``__init__`` function takes as argument the
  simulation object);

  The task of this class is to initialize the state before the
  simulation.

- TimeStepping (the ``__init__`` function takes as argument the
  simulation object);

  The TimeStepping class contains functions for the time advancement:
  time looping, Runge-Kutta scheme, CFL condition...

- Forcing

  ...
  
- Output

  * stdout + io
  * post-processing
  * plotting

Other very important classes: ContainerXML, Parameters, InfoSolver
------------------------------------------------------------------

- ContainerXML

- Parameters

- InfoSolver

  Every Simul class has to have an attribute InfoSolver, which is a
  InfoSolver class, i.e.\ a child of ContainerXML. This class is used
  to describe all other classes that are used for the solver.
