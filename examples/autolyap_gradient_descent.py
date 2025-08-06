import numpy as np
import PEPit
import PEPit.functions
import cvxpy as cp
import dsp
import autolyapWpepit


# Example on gradient descent
problem = PEPit.PEP()
L = 1.
f = problem.declare_function(PEPit.functions.SmoothConvexFunction, L=L, name='f')
xs = PEPit.Point(name='xs')
zero = PEPit.Point(is_leaf=False, decomposition_dict=dict(), name='zero')
fs = PEPit.Expression(name='fs')
f.add_point((xs, zero, fs))
x0 = problem.set_initial_point('x0')

g0 = f.gradient(x0)
g0.name = 'g0'
x1 = x0 - 1.5/L * g0
g1 = f.gradient(x1)
g1.name = 'g1'

G, F, C = autolyapWpepit.extract_cvxpy_variables_and_constraints(problem)

Sigma = autolyapWpepit.transition_matrix_for_points([[xs, xs], [x0, x1], [g0, g1]])
sigma = autolyapWpepit.transition_matrix_for_expressions([[fs, fs], [f(x0), f(x1)]])
rows_Sigma = (np.sum(Sigma != 0, axis=1) > 0)
rows_sigma = (np.sum(sigma != 0, axis=1) > 0)
NN = autolyapWpepit.find_nonnegative_quantities(f, rows_Sigma, rows_sigma)

size_Q = sum(rows_Sigma)
Q = cp.Variable((size_Q, size_Q), 'Q', PSD=True)

size_q = len(NN)
q = cp.Variable(size_q, 'q')
C += [q >= 0]

V1mV = autolyapWpepit.build_lyapunov_function_difference(Sigma, sigma, rows_Sigma, G, F, Q, q, NN)
R_coeff = autolyapWpepit.exp2mat(f(x1) - fs)
R = cp.trace(R_coeff[0] @ G) + R_coeff[1] @ F + R_coeff[2]

obj = dsp.MinimizeMaximize(V1mV + R + 0.0001 * (cp.sum(Q) + cp.sum(q)))
dsp_problem = dsp.SaddlePointProblem(obj, C, minimization_vars=[Q, q], maximization_vars=[G, F])
dsp_problem.solve(eps=1e-4, verbose=True, solver=cp.SCS)
problem._eval_points_and_function_values(F.value, G.value)
problem.wrapper.assign_dual_values()

print("Saddle point problem value is:", (V1mV+R).value, 'and should be 0')

print("The Lyapunov function found is", Q.value[0,0], "xs**2+", Q.value[0,1]+Q.value[1,0], "xs x0+",
      Q.value[1,1], "x0**2+", Q.value[2,2], "g0**2+", Q.value[0,2], "g0 xs+", Q.value[1, 2], "g0 x0+",
      q.value[0], "(fs - f0 - g0(x0-xs) - 1/(2L)g0**2)+", q.value[1], "(f0 - fs - 1/(2L)g0**2)")
