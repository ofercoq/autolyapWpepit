Code for the paper "Defining Lyapunov functions as the solution of a performance estimation saddle point problem" by Olivier Fercoq
available on https://arxiv.org/abs/2411.12317

It consists on a set of functions that allow the user to automatically find a Lyapunov function for an optimization algorithm.
It is based on PEPit to define the relations between the iterates of the algorithm and on dsp-cvxpy for the numerical resolution.

Requires:
  - PEPit: https://pepit.readthedocs.io/en/0.3.2/
  - DSP-CVXPY: https://pypi.org/project/dsp-cvxpy/
To use the software:
  - import PEPit, cvxpy, dsp
  - import autolyapWpepit
  - define your algorithm using PEPit
  - use the function of autolyapWpepit as in the examples
  - solve the saddle point problem using dsp-cvxpy
