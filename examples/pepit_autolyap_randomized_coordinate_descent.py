import numpy as np
import PEPit
from PEPit import functions
import cvxpy as cp
import dsp
import autolyapWpepit

d = 2
L = 0*np.arange(d, dtype=float)+1
gamma = 1 / np.array(L)
wrapper = "cvxpy"
solver = cp.SCS
verbose = 1


# Instantiate PEP
problem = PEPit.PEP()

# Declare a partition of the ambient space in d blocks of variables
partition = problem.declare_block_partition(d=d)

# Declare a smooth convex function
func = problem.declare_function(PEPit.functions.SmoothConvexFunction, L=d*max(L))

# Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
xs = func.stationary_point(name='stationary point')
fs = func(xs)
fs.name = "fs"

# Then define the point x_{t-1} of the algorithm
x0 = problem.set_initial_point(name='initial point')
# x0s = [partition.get_block(x0 - xs, i) for i in range(d)]

print(PEPit.Point.counter)
# Set the initial condition
# problem.set_initial_condition(sum([x0s[i]**2 * L[i] for i in range(d)]) <= 1, name='initial condition')
problem.set_initial_condition((x0-xs)**2 <= 1, name='initial_condition')
print(PEPit.Point.counter)

# Compute all the possible outcomes of the randomized coordinate descent step
f0 = func(x0)
f0.name = "f0"
g0 = func.gradient(x0)
g0.name = 'gradient 0'

for i in range(d):
    partition.get_block(g0, i).name = 'gradient 0 - block ' + str(i)
x1_list = [x0 - gamma[i] * partition.get_block(g0, i) for i in range(d)]

# add refined class constraints (the result is larger than 1 without them)

# Create additional constraints when two vectors have only one element that differ
for k in range(partition.get_nb_blocks()):
    constraint = (func(x1_list[k]) <= f0 + g0 * (x1_list[k] - x0) + L[k]/2 * (x1_list[k] - x0)**2)
    func(x1_list[k]).name = "f(x1_"+str(k)+")"
    problem.add_constraint(constraint, "coordinate-wise smoothness for block "+str(k))

for i in range(d):
    func.gradient(x1_list[i]).name = 'gradient 1 - case ' + str(i)

G, F, constraints = autolyapWpepit.extract_cvxpy_variables_and_constraints(problem)

# Construct the transition matrix
Sigma = []
for i in range(d):
    Sigma.append(autolyapWpepit.transition_matrix_for_points([[xs, xs], [x0, x1_list[i]], [g0, func.gradient(x1_list[i])]]))
    
# Declare Lyapunov function and residual function
rows_Sigma = (np.sum(Sigma[0] != 0, axis=1) > 0)
size_Q = sum(rows_Sigma)
Q = cp.Variable((size_Q, size_Q), 'Q', PSD=True)
q = cp.Variable(1, 'q')
constraints += [100*d >= q, q >= 0]
constraints += [Q[2,2] == 0]

R_coefs = autolyapWpepit.exp2mat(func(x0) - func(xs))
R = cp.trace(R_coefs[0] @ G) + R_coefs[1] @ F + R_coefs[2]

F0_coefs = autolyapWpepit.exp2mat(func(x0) - func(xs))[1]
F1_coefs = [autolyapWpepit.exp2mat(func(x1_list[i]) - func(xs))[1] for i in range(d)]

V1mV = 1/d * sum([dsp.inner(Q.flatten(), (Sigma[i] @ G @ Sigma[i].T - G)[rows_Sigma][:, rows_Sigma].flatten()) for i in range(d)]) \
    + dsp.inner(q, (1/d*sum(F1_coefs)- F0_coefs) @ F)

# Declare saddle point performance estimation problem
obj = dsp.MinimizeMaximize(V1mV + R + 0.001*q + 0.001 * Q[0,0])

dsp_problem = dsp.SaddlePointProblem(obj, constraints, minimization_vars=[Q, q], maximization_vars=[F,G])

# Solve the problem
eps = 1e-4
dsp_problem.solve(eps=eps, verbose=True, solver=cp.SCS)
problem._eval_points_and_function_values(F.value,
                                         G.value)
problem.wrapper.assign_dual_values()

print(str((V1mV+R).value)+ ' should be 0')

print('Theoretical Lyapunov inequality: E['+str(d)+' f(x1) + '+str(d/2)+' ||x1 - xs||**2_L] \leq (d-1) f(x0) + d/2 ||x0 - xs||**2_L')
qv = np.round(q.value[0], 3)
Qv = np.round(Q.value, 3)
Q11 = Qv[1,1]

co = size_Q
print('Numerical Lyapunov inequality for d='+str(d)+': E['+str(qv)+'f(x1) +'+str(Q11)+'||x1-xs||**2+'+str(Qv[0, co-1])+'g(x1)(x1-xs)+'+str(Qv[co-1, co-1])+'g(x1)**2] <= '+str(qv-1)+'f(x0) +'+str(Q11)+'||x0-xs||**2+'+str(round(Q.value[0, co-1],2))+'g(x0)(x0-xs)+'+str(round(Q.value[co-1, co-1],2))+'g(x0)**2')
