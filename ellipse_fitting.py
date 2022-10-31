from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cp

def fit_ellipse(X, Y):
    n = X.shape[0]
    U = np.vstack((X, Y))
    P = cp.Variable((2, 2), symmetric=True)
    b = cp.Variable(2)
    c = cp.Variable(1)
    constraints = [P >> np.identity(2)]
    # constraints += [cp.quad_form(U[:, i], P) + b.T@U[:, i] + c >= -1 for i in range(n)]
    constraints += [cp.quad_form(U[:, i], P) + b.T@U[:, i] + c <= 1 for i in range(n)]
    prob = cp.Problem(
        cp.Minimize(0),
        constraints)
    prob.solve()

    print(f"P: {P.value} \nb: {b.value} \nc: {c.value}")
    return P.value, b.value, c.value

def fit_ellipse3(X, Y):
    n = X.shape[0]
    U = np.vstack((X, Y))
    A = cp.Variable((2, 2), symmetric=True)
    b = cp.Variable(2)
    constraints = [A >> 0]
    constraints += [cp.norm(A@U[:, i] + b) <= 1 for i in range(n)]
    prob = cp.Problem(
        cp.Minimize(-cp.log_det(A)),
        constraints)
    prob.solve()

    print(f"A: {A.value} \nb: {b.value} \n")
    P = A.value.T@A.value
    bb = 2*A.value.T@b.value
    cc = b.value.T@b.value

    return P, bb, cc

def fit_ellipse2(X, Y):
    n = X.shape[0]
    U = np.vstack((X, Y))
    A = cp.Variable((2, 2), symmetric=True)
    b = cp.Variable(2)
    S = cp.Variable(n)
    constraints = [A >> 0]
    constraints += [S[i] == cp.norm(A@U[:, i] + b) for i in range(n)]
    prob = cp.Problem(
        cp.Minimize(cp.norm(S - np.ones(n))),
        constraints)
    prob.solve()

    print(f"A: {A.value} \nb: {b.value} \nS: {S.value}")
    return A.value, b.value, S.value

if __name__ == '__main__':
    a = 20
    b = 10
    xc = 0
    yc = 0
    N = 50
    X = np.linspace(-a+xc, a+xc, N)
    Y = b / a * np.sqrt(a**2 - (X-xc)**2) + yc

    num_sample = 10
    interval = int(N/num_sample)
    samplesX = []
    samplesY = []
    for i in range(0, N, interval):
        samplesX.append(X[i])
        samplesY.append(Y[i])
    samplesX = np.array(samplesX)
    samplesY = np.array(samplesY)
    P_v, b_v, c_v = fit_ellipse3(samplesX, samplesY)
    # P[0, 0]*x**2 + P[1, 1]*y**2 + (P[1, 2] + P[2, 1])*x*y + b[0]*x + b[1]*y + c = 0
    # a * x**2 + b * y**2 + c * x*y + d*x + e*y + f = 0
    a = P_v[0, 0]
    b = P_v[1, 1]
    c = P_v[0, 1] + P_v[1, 0]
    d = b_v[0]
    e = b_v[1]
    f = c
    # reorder equations: b*y**2 + (c*x + e)*y + (a*x**2 + d*x + f) = 0
    # Rewrite notations: aa * y**2 + bb * y + cc = 0
    Y_fit = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        xi = X[i]
        aa = b
        bb = c*xi + e
        cc = a*xi**2 + d*xi + f
        yi = 0
        if bb**2 - 4 * aa * cc >= 0:
            yi = (-bb + np.sqrt(bb**2 - 4 * aa * cc)) / (2 * aa)
        Y_fit[i] = yi

    plt.plot(X, Y)
    plt.scatter(samplesX, samplesY)
    plt.plot(X, Y_fit, 'r')
    # plt.xlim([-20, 30])
    # plt.ylim([0, 50])
    plt.show()



