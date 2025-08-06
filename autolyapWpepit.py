import numpy as np
import PEPit
import PEPit.functions
import cvxpy as cp
import dsp


def list_of_leaf_point_names():
    return [PEPit.Point.list_of_leaf_points[i].name for i in range(PEPit.Point.counter)]

def list_of_leaf_expression_names():
    return [PEPit.Expression.list_of_leaf_expressions[i].name for i in range(PEPit.Expression.counter)]

# Constructing next iterate matrix Sigma
exp2mat = PEPit.tools.expressions_to_matrices.expression_to_matrices
def transition_matrix_for_points(list_of_couples_of_points):
    # list_of_couples_of_expressions = [[x0, x1], [xs, xs]]
    n_leaf_P = PEPit.Point.counter
    Sigma = np.zeros((n_leaf_P, n_leaf_P))
    for couple in list_of_couples_of_points:
        row = (exp2mat(couple[0]**2)[0]==1)
        if np.sum(row) != 1:
            print("error when building transition matrix: one of the points is not a leaf point", couple[0], couple[0].name)
        Sigma += exp2mat(couple[0]*couple[1])[0]
    Sigma = np.triu(Sigma) * 2 - np.diag(np.diag(Sigma))
    return Sigma

def transition_matrix_for_expressions(list_of_couples_of_expressions):
    # list_of_couples_of_expressions = [[f(x0), f(x1)], [f(xs), f(xs)]]
    n_leaf_F = PEPit.Expression.counter
    sigma = np.zeros((n_leaf_F, n_leaf_F))

    for couple in list_of_couples_of_expressions:
        exp1 = couple[0]
        row = (exp2mat(exp1)[1]==1)
        if sum(row) != 1:
            print("error when building transition matrix: one of the expressions is not a leaf expression", exp1, exp1.name)
        sigma[row] = exp2mat(couple[1])[1]
    return sigma

def find_nonnegative_quantities(f, rows_Sigma, rows_sigma):
    NN = []
    for const in f.list_of_class_constraints:
        G_coef, F_coef, c_coef = exp2mat(const.expression)
        rows_G_coef = (np.sum(G_coef != 0, axis=1) > 0)
        if np.all(rows_G_coef * rows_Sigma == rows_G_coef) \
           and np.all((F_coef != 0) * rows_sigma == (F_coef != 0)): 
            # add nonegative quantity
            NN.append([-G_coef, - F_coef, -c_coef])
        
    return NN

def extract_cvxpy_variables_and_constraints(problem):
    # We solve a silly PEP to recover a cvxpy object coding the constraints
    problem.set_performance_metric(0*PEPit.Expression())
    problem.solve(wrapper="cvxpy", verbose=0, solver=cp.SCS)
    cvxpy_prob = problem.wrapper.prob
    F = cvxpy_prob.variables()[0]
    G = cvxpy_prob.variables()[1]
    C = cvxpy_prob.constraints
    return G, F, C

def build_lyapunov_function_difference(Sigma, sigma, rows_Sigma, G, F, Q, q, NN):
    V1mV = dsp.inner(Q.flatten(), (Sigma @ G @ Sigma.T - G)[rows_Sigma][:, rows_Sigma].flatten()) \
                       + np.sum([dsp.inner(q[i], NN[i][1] @ (sigma @ F - F) + \
                                 cp.trace(NN[i][0] @ (Sigma @ G @ Sigma.T - G))) for i in range(q.size)])
    return V1mV
