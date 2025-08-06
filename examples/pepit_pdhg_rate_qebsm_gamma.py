# This code numerically checks the convergence of the PDHG method
# by constructing a Lyapunov function defined as the solution
# of a performance estimation saddle point problem.
#
# We then use this technique to determine the worst case convergence rate
# of PDHG under quadratic error bound of the smoothed gap and for
# various values of the step size product.

import numpy as np
import matplotlib.pyplot as plt
import sys
import PEPit
from PEPit import operators
from PEPit import functions
from PEPit import primitive_steps
import cvxpy as cp
import dsp

def list_of_leaf_point_names():
    return [PEPit.Point.list_of_leaf_points[i].name for i in range(PEPit.Point.counter)]

solver = None
verbose = 1

GAMMA = np.arange(1-0.08*5, 4/3, 0.08)
ETA = 10 ** np.linspace(1, -4, 9)

try:
    npzfile = np.load('results_pepit_pdhg_rate_gebsm_gamma_full_Qq.npz')
    RHO = npzfile['RHO']
except:
    RHO = np.inf * np.ones((len(ETA), len(GAMMA)))

for i_eta, eta in enumerate(ETA):
    for i_gamma, gamma in enumerate(GAMMA):
        if RHO[i_eta, i_gamma] < np.inf:
            print('eta = ' + str(eta) + ' and gamma = ' + str(gamma) + ' already done.')
        else:
            print('---------------')
            print('eta = ' + str(eta) + ' and gamma = ' + str(gamma))
            L = 1.
            theta = 1.
            sigma_ = np.sqrt(gamma / L)
            tau_ = gamma / L / sigma_

            # min_x max_y f(x) + y A x - g(y), résolu par chambolle-pock

            # Instantiate PEP (and erase previously defined one)
            problem = PEPit.PEP()

            # Declare a smooth convex function
            f = problem.declare_function(PEPit.functions.ConvexFunction, name='f')
            g = problem.declare_function(PEPit.functions.ConvexFunction, name='g')
            M = problem.declare_function(PEPit.operators.LinearOperator, L=L, name='A')
            A = M.gradient
            AT = M.T.gradient

            # partition the space into primal and dual spaces
            partition = problem.declare_block_partition(d=2)

            # Define a saddle point zs and corresponding Lagrangian value Ls = f(xs) + ys A xs - g(ys)
            zs = PEPit.Point(is_leaf=False, decomposition_dict=dict(), name='zs')  # without loss of generality, we can choose it to be 0.
            xs = partition.get_block(zs, 0)
            ys = partition.get_block(zs, 1)
            xs.name = 'xs'
            ys.name = 'ys'
            fs = PEPit.Expression(name="fs")
            f.add_point((xs, -AT(ys), fs))
            gs = PEPit.Expression(name="gs")
            g.add_point((ys, A(xs), gs))
            A(xs).name = 'Axs'
            Ls = fs + ys * A(xs) - g(ys)

            counter_for_stationary_point = PEPit.Point.counter

            # Then define the point z_{0} of the algorithm
            z = problem.set_initial_point('initial point')
            x = partition.get_block(z, 0)
            y = partition.get_block(z, 1)
            x.name = "x"
            y.name = "y"
            Ax = A(x)
            Ax.name = "Ax"

            counter_for_initial_point = PEPit.Point.counter

            # Define one iteration of the primal-dual hybrid gradient
            prox = PEPit.primitive_steps.proximal_step
            yb1, _, _ = prox(y + sigma_ * A(x), g, sigma_)
            g.list_of_points[-1][1].name = 'diff_yb1'
            yb1.name = 'yb1'
            ATyb1 = AT(yb1)
            ATyb1.name = 'ATyb1'
            xb1, _, _ = prox(x - tau_ * ATyb1, f, tau_)
            f.list_of_points[-1][1].name = 'diff_xb1'
            xb1.name = 'xb1'
            x1 = xb1
            Ax1 = A(x1)
            Ax1.name = "Ax1"
            y1 = yb1 + theta * sigma_ * (Ax1 - Ax)
            y1.name = 'y1'
            zb1 = xb1 + yb1
            z1 = x1 + y1
            z1.name = 'z1'

            # define the smoothed gap at zb
            # G(zb) = max_(x,y) f(xb) + y A xb - g(y) - f(x) - yb A x - g(yb) - beta/2 ||zb - z||**2
            #       = f(xb) + pyb A xb - g(pyb) - f(pxb) - yb A pxb - g(yb) - beta/2 ||zb - pzb||^2
            # où pxb = prox_{f/beta}(xb - 1/beta AT yb) et pyb = prox_{g/beta}(yb + 1/beta A xb)
            beta = 1
            pxb1, _, _ = prox(xb1 - 1/beta * ATyb1, f, 1/beta)
            f.list_of_points[-1][1].name = 'diff_pxb1'
            pxb1.name = 'pxb1'
            pyb1, _, _  = prox(yb1 + 1/beta * A(xb1), g, 1/beta)
            g.list_of_points[-1][1].name = 'diff_pyb1'
            pyb1.name = 'pyb1'
            Gzb1 = f(xb1) + pyb1 * A(xb1) - g(pyb1) - f(pxb1) - yb1 * A(pxb1) + g(yb1) \
                - beta/2 * ((xb1 - pxb1)**2 / tau_ + (yb1 - pyb1)**2 / sigma_)
            # Define the difference of Lagrangian values
            Dzb1 = f(xb1) + ys * A(xb1) - g(ys) - f(xs) - yb1 * A(xs) + g(yb1)

            counter_for_z1 = PEPit.Point.counter

            yb2, _, _ = prox(y1 + sigma_ * A(x1), g, sigma_)
            g.list_of_points[-1][1].name = 'diff_yb2'
            yb2.name = 'yb2'
            ATyb2 = AT(yb2)
            ATyb2.name = 'ATyb2'
            xb2, _, _ = prox(x1 - tau_ * ATyb2, f, tau_)
            f.list_of_points[-1][1].name = 'diff_xb2'
            xb2.name = 'xb2'
            x2 = xb2
            Ax2 = A(x2)
            Ax2.name = "Ax2"
            y2 = yb2 + theta * sigma_ * (Ax2 - Ax1)
            y2.name = 'y2'
            zb2 = xb2 + yb2
            z2 = x2 + y2
            z2.name = 'z2'

            pxb2, _, _ = prox(xb2 - 1/beta * ATyb2, f, 1/beta)
            pxb2.name = 'pxb2'
            f.list_of_points[-1][1].name = 'diff_pxb2'
            pyb2, _, _  = prox(yb2 + 1/beta * A(xb2), g, 1/beta)
            pyb2.name = 'pyb2'
            g.list_of_points[-1][1].name = 'diff_pyb2'
            Gzb2 = f(xb2) + pyb2 * A(xb2) - g(pyb2) - f(pxb2) - yb2 * A(pxb2) + g(yb2) \
                - beta/2 * ((xb2 - pxb2)**2 / tau_ + (yb2 - pyb2)**2 / sigma_)
            Dzb2 = f(xb2) + ys * A(xb2) - g(ys) - f(xs) - yb2 * A(xs) + g(yb2)

            counter_for_z2 = PEPit.Point.counter

            # xb, yb, x_next, y_next, etc should remain in their respective primal or dual spaces
            for x_ in [xs, x, xb1, ATyb1, x1, pxb1, xb2, ATyb2, x2, pxb2]:
                for y_ in [ys, y, Ax, yb1, y1, Ax1, pyb1, yb2, y2, pyb2]:
                    problem.add_constraint(x_ * y_ == 0, 'partition')

            def norm2_V(x,y):
                return x**2 / tau_ + y**2 / sigma_

            problem.set_initial_condition(norm2_V(x-xs, y-ys) <= 1, 'initial condition')

            # Set an arbitrary performance metric
            problem.set_performance_metric(0*Ls)

            problem.add_constraint(Gzb1 >= eta/2 * norm2_V(xb1 - xs, yb1 - ys), 'qeb sm')
            problem.add_constraint(Gzb2 >= eta/2 * norm2_V(xb2 - xs, yb2 - ys), 'qeb sm')

            # Solve the silly PEP to recover a cvxpy object coding the constraints
            print("Constructing CVXPY objects from PEPit.")
            pepit_verbose = 0
            problem.solve(wrapper="cvxpy", solver=solver, verbose=pepit_verbose)
            cvxpy_prob = problem.wrapper.prob

            # Constructing next iterate matrix Sigma
            exp2mat = PEPit.tools.expressions_to_matrices.expression_to_matrices
            Sigma = exp2mat(z*z1+x*x1+Ax*Ax1)[0]
            Sigma += exp2mat(
                sum([PEPit.Point.list_of_leaf_points[counter_for_initial_point + a] * PEPit.Point.list_of_leaf_points[counter_for_z1 + a]
                    for a in range(counter_for_z1 - counter_for_initial_point)]))[0]
            Sigma = np.triu(Sigma) * 2 - np.diag(np.diag(Sigma))
            Sigma += exp2mat(
                sum([PEPit.Point.list_of_leaf_points[a] ** 2 for a in range(counter_for_stationary_point)]))[0]

            # Declaring cvxpy variables for the Lyapunov function and the residual function

            co = 10
            Q = cp.Variable((co, co), 'Q')
            q = cp.Variable(2, 'q')

            S = cp.Variable((co, co), 'S')
            s = cp.Variable(2, 's')
            Dzb1_G, Dzb1_F, Dzb1_C = exp2mat(Dzb1)
            Gzb1_G, Gzb1_F, Gzb1_C = exp2mat(Gzb1)

            Fzb1_G, Fzb1_F, Fzb1_C = exp2mat(f(xb1) + ys * A(xb1) - g(ys))
            Fzb2_G, Fzb2_F, Fzb2_C = exp2mat(f(xb2) + ys * A(xb2) - g(ys))
            gzb1_G, gzb1_F, gzb1_C = exp2mat(-f(xs) - yb1 * A(xs) + g(yb1))
            gzb2_G, gzb2_F, gzb2_C = exp2mat(-f(xs) - yb2 * A(xs) + g(yb2))

            # Set constraints on Q and q
            Px0s = exp2mat((x-xs)**2)[0]
            Py0s = exp2mat((y-ys)**2)[0]
            Px01 = exp2mat((x-x1)**2)[0]
            Py01 = exp2mat((y-y1)**2)[0]
            Py0b = exp2mat((y-yb1)**2)[0]
            Px1s = exp2mat((x1-xs)**2)[0]
            Py1s = exp2mat((y1-ys)**2)[0]
            P0s = (1/tau_ * Px0s + 1/sigma_ * Py0s)
            P01 = (1/tau_ * Px01 + 1/sigma_ * Py01)

            constraints = [Q >> P0s[:co, :co]]
            constraints += [q >= 0]

            # Simplification constraints
            # a = cp.Variable(7)
            # constraints += [Q == (a[0] * Px0s + a[1] * Py0s + a[2] * Px01 + a[3] * Py01 + a[4] * Py0b
            #                      + a[5] * Px1s + a[6] * Py1s)[:co,:co]]
            # constraints += [q == 0, a[4] == 0]

            # constraints += [Q == a[0] * P0s[:co, :co]]  # this constraint works only for gamma < 1

            F = cvxpy_prob.variables()[0]
            G = cvxpy_prob.variables()[1]

            eps = 1e-4
            if i_eta == 0:
                if i_gamma == 0:
                    rho_min = 0.
                    rho_max = 1.
                else:
                    rho_min = 0.5 * RHO[i_eta, i_gamma-1]
                    rho_max = 1.
            else:
                rho_min = RHO[i_eta-1, i_gamma]
                rho_max = 1.
            rho = (rho_max + rho_min) / 2
            while rho_max - rho_min > max(eps, 0.1*(1-rho_min)):
                print('Dichotomy in progress: rho_max - rho_min = ', str(rho_max - rho_min))
                V1mV = dsp.inner(Q.flatten(), (Sigma @ G @ Sigma.T - rho * G)[:co, :co].flatten()) \
                    + dsp.inner(q[0], (Fzb2_F @ F + cp.trace(Fzb2_G @ G) + Fzb2_C) - rho * (Fzb1_F @ F + cp.trace(Fzb1_G @ G) + Fzb1_C)) \
                    + dsp.inner(q[1], (gzb2_F @ F + cp.trace(gzb2_G @ G) + gzb2_C) - rho * (gzb1_F @ F + cp.trace(gzb1_G @ G) + gzb1_C))

                obj = dsp.MinimizeMaximize(V1mV)

                dsp_problem = dsp.SaddlePointProblem(obj, constraints + cvxpy_prob.constraints, minimization_vars=[Q, q], maximization_vars=cvxpy_prob.variables())

                # solve the problem
                print("Solving saddle point problem using DSP-CVXPY.")

                try:
                    dsp_problem.solve(eps=eps, verbose=False)
                except:
                    print("- problem is dsp solver, but we continue -")

                problem._eval_points_and_function_values(cvxpy_prob.variables()[0].value,
                                                         cvxpy_prob.variables()[1].value)

                print("Saddle point problem value for rho = " + str(rho) + " is " + str(V1mV.value) + " and should be 0.")
                if V1mV.value < eps:
                    # rho is valid, so the true one is smaller
                    rho_max = rho
                    rho = (rho_max + rho_min) / 2
                else:
                    rho_min = rho
                    rho = (rho_max + rho_min) / 2

            RHO[i_eta, i_gamma] = rho_max
            np.savez('results_pepit_pdhg_rate_gebsm_gamma_full_Qq',ETA=ETA, GAMMA=GAMMA, RHO=RHO)

markers = ['$a$', '$b$', '$c$', '$d$', '$e$', '$f$', '$g$', '$h$', '$i$']
for i in range(RHO.shape[0]):
    plt.semilogy(GAMMA, 1-RHO[i], marker=markers[i]);
plt.xlabel('γ'); plt.ylabel('1-ρ'); plt.legend(np.round(ETA,4)); plt.show()
