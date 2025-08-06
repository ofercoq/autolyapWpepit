# This code numerically checks the convergence of the PURE-CD method
# in the dense matrix case by constructing a Lyapunov function defined
# as the solution of a performance estimation saddle point problem.
#
# We then use this technique to check whether PURE-CD may converge for
# larger step sizes than currently proved.

import numpy as np
import PEPit
from PEPit import operators
from PEPit import functions
from PEPit import primitive_steps
import cvxpy as cp
import dsp
import autolyapWpepit

prox = PEPit.primitive_steps.proximal_step
exp2mat = PEPit.tools.expressions_to_matrices.expression_to_matrices

gamma = 1.3

# min_x max_y sum_i g_i(x_i) + y A_i x_i - h(y), résolu par pure-cd

# Instantiate PEP
problem = PEPit.PEP()

# Declare a smooth convex function
n = 2
L = [1.] * n
h = problem.declare_function(PEPit.functions.ConvexFunction, name='h')
M = [[]] * n
A = [[]] * n
AT = [[]] * n
g = [[]] * n
for i in range(n):
    M[i] = problem.declare_function(PEPit.operators.LinearOperator, L=L[i],
                                    name='A'+str(i))
    A[i] = M[i].gradient
    AT[i] = M[i].T.gradient
    g[i] = problem.declare_function(PEPit.functions.ConvexFunction, name='g'+str(i))

# Declare a partition of the primal-dual space,
#   where the dual space is split into n blocks
partition = problem.declare_block_partition(d=1+n)
# We will have A[i] acting from block i to block n:
#   these constraints will be defined later.

theta = n
tau_ = np.sqrt(gamma * n / theta / np.array(L)**2)
sigma_ = np.max(gamma / np.array(L)**2 / tau_ / n)

# Define a saddle point zs
#   and corresponding Lagrangian value Ls = sum_i g_i(xs[i]) + sum_i ys A[i] xs[i] - f(ys)
recenter_saddle_point = False
if recenter_saddle_point:
    zs = PEPit.Point(name='zs', is_leaf=False, decomposition_dict=dict())
    xs = [[]] * n
    for i in range(n):
        xs[i] = PEPit.Point(is_leaf=False, decomposition_dict=dict())
    ys = PEPit.Point(is_leaf=False, decomposition_dict=dict())
else:
    zs = PEPit.Point(name='zs')
    xs = [partition.get_block(zs, i) for i in range(n)]
    ys = partition.get_block(zs, n); ys.name = 'ys'
for i in range(n):
    xs[i].name = 'xs'+str(i)

Axs_dev = [[]] * n
for i in range(n):
    Axs_dev[i] = A[i](xs[i], name='A'+str(i)+'(xs'+str(i)+')')
Axs = np.sum([Axs_dev[i] for i in range(n)])
hs = PEPit.Expression(name="hs")
diff_ys = Axs
h.add_point((ys, diff_ys, hs))

ATys = [[]] * n
gs = [[]] * n
diff_xs = [[]] * n
for i in range(n):
    gs[i] = PEPit.Expression(name="gs"+str(i))
    ATys[i] = AT[i](ys, name='AT'+str(i)+'(ys)')
    diff_xs[i] = -ATys[i]
    g[i].add_point((xs[i], diff_xs[i], gs[i]))

Ls = -hs + np.sum([xs[i] * ATys[i] + g[i](xs[i]) for i in range(n)])

counter_for_stationary_point = PEPit.Point.counter

# Then define the point z_{0} of the algorithm
z = problem.set_initial_point('initial point')
x = [partition.get_block(z, i) for i in range(n)]
y = partition.get_block(z, n); y.name="y"
Ax_dev = [[]] * n
for i in range(n):
    x[i].name = "x"+str(i)
    Ax_dev[i] = A[i](x[i], name='A'+str(i)+'(x'+str(i)+')')
Ax = np.sum([Ax_dev[i] for i in range(n)])

counter_for_initial_point = PEPit.Point.counter

# Define one iteration of pure-cd
yb1, diff_yb1, h_yb1 = prox(y + sigma_ * Ax, h, sigma_, name='yb1')

ATyb1 = [[]] * n
xb1 = [[]] * n
diff_xb1 = [[]] * n
g_xb1 = [[]] * n
for i in range(n):
    ATyb1[i] = AT[i](yb1, name='ATyb1_'+str(i))
    xb1[i], diff_xb1[i], g_xb1[i] = prox(x[i] - tau_[i] * ATyb1[i], g[i], tau_[i])
                                         # , name='xb1_'+str(i))

Axb1_dev = [[]] * n
for i in range(n):
    Axb1_dev[i] = A[i](xb1[i], name='A'+str(i)+'(xb1_'+str(i)+')')
    
# We now draw a random variable j in {0, ..., n-1} and list
#   all the possible outcomes of the algorithm
y1 = [[]] * n
x1 = [[]] * n
for j in range(n):
    x1[j] = [[]] * n
    for i in range(n):
        if i == j:
            x1[j][j] = xb1[j]
        else:
            x1[j][i] = x[i]
    y1[j] = yb1 + sigma_ * theta * (Axb1_dev[j] - Ax_dev[j])

z1 = [[]] * n
for j in range(n):
    z1[j] = y1[j] + np.sum([x1[j][i] for i in range(n)])
    z1[j].name = 'z1_' + str(j)
counter_for_z1 = PEPit.Point.counter

# Additional iteration for the Lyapunov function

Ax1_dev = [[]] * n
Ax1 = [[]] * n
for j in range(n):
    Ax1_dev[j] = [[]] * n
    for i in range(n):
        Ax1_dev[j][i] = A[i](x1[j][i])
    Ax1[j] = np.sum([Ax1_dev[j][i] for i in range(n)])

yb2 = [[]] * n
diff_yb2 = [[]] * n
h_yb2 = [[]] * n
for j in range(n):
    yb2[j], diff_yb2[j], h_yb2[j] = prox(y1[j] + sigma_ * Ax1[j], h, sigma_)
                                        #, name='yb2_'+str(j))

ATyb2 = [[]] * n
xb2 = [[]] * n
diff_xb2 = [[]] * n
g_xb2 = [[]] * n
for j in range(n):
    ATyb2[j] = [[]] * n
    xb2[j] = [[]] * n
    diff_xb2[j] = [[]] * n
    g_xb2[j] = [[]] * n
    for i in range(n):
        ATyb2[j][i] = AT[i](yb2[j], name='ATyb2_'+str(j)+'_'+str(i))
        xb2[j][i], diff_xb2[j][i], g_xb2[j][i] = prox(x1[j][i] - tau_[i] * ATyb2[j][i], g[i], tau_[i]) #, name='xb2_'+str(j)+'_'+str(i))

Axb2_dev = [[]] * n
for j in range(n):
    Axb2_dev[j] = [[]] * n
    for i in range(n):
        Axb2_dev[j][i] = A[i](xb2[j][i], name='A'+str(i)+'(xb2_'+str(j)+'_'+str(i)+')')

# We now draw another random variable j2 in {0, ..., n-1} and list
#   all the possible outcomes of the algorithm
y2 = [[]] * n
x2 = [[]] * n
Ax2_dev = [[]] * n
Ax2 = [[]] * n
for j in range(n):
    x2[j] = [[]] * n
    y2[j] = [[]] * n
    Ax2_dev[j] = [[]] * n
    Ax2[j] = [[]] * n
    for j2 in range(n):
        x2[j][j2] = [[]] * n
        Ax2_dev[j][j2] = [[]] * n
        for i in range(n):
            if i == j2:
                x2[j][j2][j2] = xb2[j][j2]
                Ax2_dev[j][j2][j2] = Axb2_dev[j][j2]
            else:
                x2[j][j2][i] = x1[j][i]
                Ax2_dev[j][j2][i] = Ax1_dev[j][i]
        Ax2[j][j2] = np.sum([Ax2_dev[j][j2][i] for i in range(n)])
        y2[j][j2] = yb2[j] + sigma_ * theta * (Axb2_dev[j][j2] - Ax1_dev[j][j2])

z2 = [[]] * n
for j in range(n):
    z2[j] = [[]] * n
    for j2 in range(n):
        z2[j][j2] = y2[j][j2] + np.sum([x2[j][j2][i] for i in range(n)])
        z2[j][j2].name = 'z2_' + str(j) + '_' + str(j2)
counter_for_z2 = PEPit.Point.counter

yb3 = [[]] * n
diff_yb3 = [[]] * n
h_yb3 = [[]] * n
for j in range(n):
    yb3[j] = [[]] * n
    diff_yb3[j] = [[]] * n
    h_yb3[j] = [[]] * n
    for j2 in range(n):
        yb3[j][j2], diff_yb3[j][j2], h_yb3[j][j2] = prox(y2[j][j2] + sigma_ * Ax2[j][j2], h, sigma_)
        #, name='yb3_'+str(j)+str(j2))

ATyb3 = [[]] * n
xb3 = [[]] * n
diff_xb3 = [[]] * n
g_xb3 = [[]] * n
for j in range(n):
    ATyb3[j] = [[]] * n
    xb3[j] = [[]] * n
    diff_xb3[j] = [[]] * n
    g_xb3[j] = [[]] * n
    for j2 in range(n):
        ATyb3[j][j2] = [[]] * n
        xb3[j][j2] = [[]] * n
        diff_xb3[j][j2] = [[]] * n
        g_xb3[j][j2] = [[]] * n
        for i in range(n):
            ATyb3[j][j2][i] = AT[i](yb3[j][j2], name='ATyb3_'+str(j)+str(j2)+'_'+str(i))
            xb3[j][j2][i], diff_xb3[j][j2][i], g_xb3[j][j2][i] = prox(x2[j][j2][i] - tau_[i] * ATyb3[j][j2][i], g[i], tau_[i]) #, name='xb3_'+str(j)+str(j2)+'_'+str(i))

Axb3_dev = [[]] * n
for j in range(n):
    Axb3_dev[j] = [[]] * n
    for j2 in range(n):
        Axb3_dev[j][j2] = [[]] * n
        for i in range(n):
            Axb3_dev[j][j2][i] = A[i](xb3[j][j2][i], name='A'+str(i)+'(xb3_'+str(j)+str(j2)+'_'+str(i)+')')

g_x = [g[i](x[i]) for i in range(n)]
diff_x = [g[i].gradient(x[i], name='diff_x_'+str(i)) for i in range(n)]
for i in range(n):
    g_x[i].name = 'g(x'+str(i)+')'

g_x1 = [[g[i](x1[j][i]) for i in range(n)] for j in range(n)]
for i in range(n):
    for j in range(n):
        if g_x1[j][i].name is None:
            g_x1[j][i].name = 'g(x1_'+str(j)+'_'+str(i)+')'
            
# xb, yb, x_next, y_next, etc should remain in their respective primal or dual spaces

list_of_primal_points = [xs, ATys, x, ATyb1, diff_xb1, diff_x] + ATyb2 + diff_xb2
for j in range(n):
    list_of_primal_points += ATyb3[j] + diff_xb3[j]
list_of_dual_points = [ys, y, diff_yb1] + Axs_dev + Ax_dev + Axb1_dev + diff_yb2
for j in range(n):
    list_of_dual_points += Axb2_dev[j]
    for j2 in range(n):
        list_of_dual_points += Axb3_dev[j][j2]

for y_ in list_of_dual_points:
    for x_ in list_of_primal_points:
        for i in range(n):
            problem.add_constraint(x_[i] * y_ == 0, 'partition')
for x1_ in list_of_primal_points:
    for x2_ in list_of_primal_points:
        for j1 in range(n):
            for j2 in range(j1):
                problem.add_constraint(x1_[j1] * x2_[j2] == 0, 'partition')

Ddyb1 = h_yb1 - hs - diff_ys * (yb1 - ys)
EDdyb2 = 1/n * sum([h_yb2[j] - hs - diff_ys * (yb2[j] - ys) for j in range(n)])
Dpx = sum([g_x[i] - gs[i] - diff_xs[i] * (x[i] - xs[i]) for i in range(n)])
Dpxb1 = sum([g_xb1[i] - gs[i] - diff_xs[i] * (xb1[i] - xs[i]) for i in range(n)])
EDpx1 = 1/n * sum([
    sum([g_x1[j][i] - gs[i] - diff_xs[i] * (x1[j][i] - xs[i])
         for i in range(n)]) for j in range(n)])
EDpxb2 = 1/n * sum([
    sum([g_xb2[j][i] - gs[i] - diff_xs[i] * (xb2[j][i] - xs[i])
         for i in range(n)]) for j in range(n)])

final_counter = PEPit.Point.counter

def norm_V(x1, y1, x2, y2, tau, sigma):
    return 0.5 * sum([(x1[i]-x2[i])**2/tau[i] for i in range(n)]) \
        + 0.5 / n * (y1 - y2)**2 / sigma

if n == 1:
    Lyap0 = Dpxb1 + 0.5/tau_[0] * (x1[0][0]-xs[0])**2 + 0.5/sigma_*(y1[0]-ys)**2+theta/2/tau_[0]*(x1[0][0]-x[0])**2 - sigma_*(4*theta**2+1)/16 * ((y1[0]-yb1)/sigma_/theta)**2
    Lyap1 = EDpxb2 + 0.5/tau_[0] * (x2[0][0][0]-xs[0])**2 + 0.5/sigma_*(y2[0][0]-ys)**2+theta/2/tau_[0]*(x2[0][0][0]-x1[0][0])**2 - sigma_*(4*theta**2+1)/16 * ((y2[0][0]-yb2[0])/sigma_/theta)**2

                                
check_papers_result = 1
if check_papers_result == 1:
    problem.set_initial_condition((1-1/n)*Dpx + norm_V(x, y, xs, ys, tau_, sigma_) <= 1, 'initial condition')
    VV1 = (1-1/n)*EDpx1 + 1/n*sum([norm_V(x1[j], y1[j], xs, ys, tau_, sigma_) for j in range(n)])
    problem.set_performance_metric(VV1)
    problem.solve(verbose=0, solver=cp.SCS)
    print(VV1.eval(), ' should be <= 1')
if check_papers_result == 2:
    constant = max((4*theta**2-1)*(4-tau_[0]*sigma_*L[0]*(2*theta+1))*(4-tau_[0]*sigma_*L[0]*(2*theta-1))/(16*tau_[0]*(8*theta-tau_[0]*sigma_*L[0]*(4*theta**2+1))), 0)
    problem.set_initial_condition(Lyap0 <= 1, 'initial condition')
    problem.set_performance_metric(Lyap1 + constant * (x1[0][0] - x[0])**2)
    problem.solve(verbose=3, solver=cp.MOSEK)

# Build a cvxpy instance corresponding to the PEP problem and extract it.
G, F, constraints = autolyapWpepit.extract_cvxpy_variables_and_constraints(problem)

# Constructing next iterate matrix Sigma
exp2mat = PEPit.tools.expressions_to_matrices.expression_to_matrices
Sigma = [[]] * n
sigma = [[]] * n
for j in range(n):
    Sigma[j] = [[]] * n
    sigma[j] = [[]] * n
    for j2 in range(n):
        Sigma[j][j2] = autolyapWpepit.transition_matrix_for_points(
            [[zs, zs]] + [[xs[i], xs[i]] for i in range(n)]
            + [[ATys[i], ATys[i]] for i in range(n)]
            + [[Axs_dev[i], Axs_dev[i]] for i in range(n)]
            + [[z, z1[j]]]+[[x[i], x1[j][i]] for i in range(n)]
            + [[ATyb1[i], ATyb2[j][i]] for i in range(n)]
            + [[Ax_dev[i], Ax1_dev[j][i]] for i in range(n)]
            + [[diff_yb1, diff_yb2[j]]]
            + [[diff_xb1[i], diff_xb2[j][i]] for i in range(n)]
            + [[Ax1_dev[j][i], Ax2_dev[j][j2][i]] for i in range(n)]
            + [[diff_yb2[j], diff_yb3[j][j2]]]
            + [[diff_xb2[j][i], diff_xb3[j][j2][i]] for i in range(n)]
        )
        sigma[j][j2] = autolyapWpepit.transition_matrix_for_expressions(
            [[gs[i], gs[i]] for i in range(n)]
            + [[hs, hs]]
            + [[g_x[i], g_x1[j][i]] for i in range(n)]
            + [[g_xb1[i], g_xb2[j][i]] for i in range(n)]
            + [[h_yb1, h_yb2[j]]]
            + [[g_xb2[j][i], g_xb3[j][j2][i]] for i in range(n)]
            + [[h_yb2[j], h_yb3[j][j2]]]
        )

rows_Sigma = (np.sum(sum([np.abs(Sigma[j][j2]) for j in range(n) for j2 in range(n)]) != 0, axis=1) > 0)
rows_sigma = (np.sum(sum([np.abs(sigma[j][j2]) for i in range(n) for j2 in range(n)]) != 0, axis=1) > 0)
NN = []
for i in range(n):
    NN += autolyapWpepit.find_nonnegative_quantities(g[i], rows_Sigma, rows_sigma)
# for i in range(n):
#    NN += autolyapWpepit.find_nonnegative_quantities(M[i], rows_Sigma, rows_sigma)
NN += autolyapWpepit.find_nonnegative_quantities(h, rows_Sigma, rows_sigma)

# Declaring cvxpy variables for the Lyapunov function and the residual function
size_Q = sum(rows_Sigma)
Q = cp.Variable((size_Q, size_Q), 'Q', PSD=True)
size_q = len(NN)
if size_q > 0:
    q = cp.Variable(size_q, 'q')
    constraints += [q >= 0]
else:
    q = cp.Variable(1)
    constraints += [q == 0]
    NN = [exp2mat(0*hs)]

# Declaring cvxpy variables for the Lyapunov function and the residual function

V1mV = 1./(n*2) * sum([autolyapWpepit.build_lyapunov_function_difference(
    Sigma[j][j2], sigma[j][j2], rows_Sigma, G, F, Q, q, NN)
                   for j2 in range(n) for j in range(n)])

# Set constraints if necessary on Q and G for numerical stability
constraints += [Q << 2000 * np.eye(Q.shape[0])]  #, q <= 1000]
# constraints += [G << 2000 * np.eye(G.shape[0])]  #, F >= - 1000, F <= 1000]

R_coeffs = exp2mat(1./n*sum([(z-z1[j])**2 for j in range(n)]))
R = cp.trace(G @ R_coeffs[0]) + F @ R_coeffs[1] + R_coeffs[2]

obj = dsp.MinimizeMaximize(V1mV + R) # + 0.001 * (cp.sum(cp.abs(Q)) + cp.sum(q)))

dsp_problem = dsp.SaddlePointProblem(obj, constraints, minimization_vars=[Q, q], maximization_vars=[F,G])

# solve the problem
print("Solving saddle point problem using DSP-CVXPY.")

try:
    dsp_problem.solve(eps=1e-4, verbose=3, accept_unknown=True, solver=cp.MOSEK)
    #dsp_problem.solve(eps=1e-4, verbose=True, max_iters=1000000, solver=cp.SCS)
except Exception as err:
    print("---------- error in DSP -----------")
    print(err)
    print("-----------------------------------")

problem._eval_points_and_function_values(F.value,
                                         G.value)
problem.wrapper.assign_dual_values()

print("Saddle point problem value is " + str((V1mV + R).value) + " and should be 0.")

# np.set_printoptions(linewidth=200)


def simple_example(gamma=0.99, K=100, n=2):
    # Pour n > 1, on peut aller plus loin que gamma=1, et même jusqu'à gamma < 2 pour n grand...
    
    # the case min_{x1, x2} max_y y(x1+x2)
    x = np.ones(n)
    y = 1
    A = np.ones(n)
    sigma = np.sqrt(gamma / n)
    tau = sigma
    Z = np.zeros(K)
    for k in range(K):
        x_ = x.copy()
        yb = y + sigma * A @ x_
        xb = x_ - tau * A.T * yb
        i = np.random.randint(n)
        x[i] = xb[i]
        y = yb + n * sigma * A @ (x - x_)
        Z[k] = abs(sum(x)) + abs(y)
    return x, y, Z


def separate_blocks(points_grouped_by_block):
    eps = 1e-5
    d = len(points_grouped_by_block)
    points = points_grouped_by_block
    new_points = [[]] * d
    for j in range(d):
        M = np.vstack([points[j][i].eval() for i in range(len(points[j]))])
        U, S, V = np.linalg.svd(M)
        rank = np.sum(S > eps)
        new_points[j] = U @ np.diag(S)[:, :rank]

    return new_points

list_of_primal_points_transposed = [ [list_of_primal_points[i][j] for i in range(len(list_of_primal_points))] for j in range(n)]
new_points = separate_blocks([list_of_dual_points] + list_of_primal_points_transposed)
