import numpy as np


def vctmply(a, b):
    n, m = a.shape
    sum = 0
    for i in range(n):
        sum = sum + a[i] * b[i]
    return sum


def matmply(a, b):
    m, n = a.shape
    o, p = b.shape
    if n != o:
        raise ValueError('wrong dimension')
    c = np.zeros([m, p], dtype=np.float32)
    s = 0.
    for i in range(m):
        for j in range(p):
            s = 0.
            for k in range(n):
                s = s + a[i, k] * b[k, j]
            c[i, j] = s
    return c


def matadd(a, b):
    m, n = a.shape
    c = np.zeros([m, n], dtype=np.float32)
    for i in range(m):
        for j in range(n):
            c[i, j] = a[i, j] + b[i, j]
    return c


def symtrc(a):
    m, n = a.shape
    b = np.zeros([m, n], dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            b[i, j] = (a[i, j] + a[j, i]) / 2.
            b[j, i] = b[i, j]
    return b

class AdaptiveKF:
    def __init__(self,h0,x0,p0,z0,r,beta=1.005):



        self.x = np.zeros([5], dtype=np.float32)
        self.xest = np.zeros([5], dtype=np.float32)
        self.p = np.zeros([5, 5], dtype=np.float32)
        self.pest = np.zeros([5, 5], dtype=np.float32)
        self.a = np.zeros([5, 1], dtype=np.float32)
        self.zold = np.zeros([5, 1], dtype=np.float32)

        self.h = np.zeros([5, 5], dtype=np.float32)
        self.htrans = np.zeros([5, 1], dtype=np.float32)

        self.k = np.zeros([5, 1], dtype=np.float32)
        self.c = np.zeros([5, 1], dtype=np.float32)
        self.q = np.zeros([5, 5], dtype=np.float32)
        self.beta=beta
        self.r=r

        for i in range(0, 5):
            self.zold[i] = z0
            self.a[i,1] = 0.


        self.htrans[0] = self.zold[0]
        self.htrans[1] = 1.0

        self.h[0, 0] = self.htrans[0]
        self.h[0, 1] = self.htrans[1]

        xfore = vctmply(self.htrans, self.xest)

    def estimate(self,z0):
        tmp = np.zeros([5, 5], dtype=np.float32)
        tmp2 = np.zeros([5, 5], dtype=np.float32)
        tmpvct = np.zeros([5, 1], dtype=np.float32)

        # Calculate Kalman Gain
        tmpvct = matmply(self.pest, self.htrans)
        scalar2 = vctmply(self.htrans, tmpvct)
        scalar = vctmply(self.htrans, self.c)
        scalar = 2.0 * scalar + scalar2 + R
        tmp2 = matadd(c, tmpvct)
        for i in range(5):
            k[i] = tmp2[i, 1] / scalar

        # determine the forecast for this period and adjust a and zold
        scalar = Z - xfore

        for i in range(4):
            a[5 - i] = a[4 - i]
            zold[5 - i] = zold[4 - i]
        a[0] = scalar
        zold[0] = Z

        # update state matrix x
        for i in range(n):
            tmp[i, 0] = scalar * k[i]

        x = matadd(xest, tmp)

        # update error covariance matrix p

        tmp = matmply(h, pest)

        for i in range(n):
            tmp2[i, 0] = c[i]

        p = matadd(tmp2, tmp)

        p = symtrc(p)

        # project ahead xest and pest for next step:

        tmp = matadd(p, q)

        for i in range(n):
            for j in range(n):
                pest[i, j] = beta * tmp[i, j]

        pest = symtrc(pest)

        for i in range(n):
            xest[i] = x[i]

        htrans[0] = zold[0]
        htrans[1] = 1.0

        h[0, 0] = htrans[0]
        h[0, 1] = htrans[1]
        xfore = vctmply(htrans, xest)

    #
    # beta = 0.0
    # zbar = 0.0
    # scalar = 0.
    # scalar2 = 0.
    #
    # vererr = 0.
    # errvar = 0.
    # mse = 0.
    # tstat = 0.
    # chi = 0.
    # stderr = 0.
    # rcnt = 0.
    #
    # z = 0.
    # y = 0.
    # r = 0.
    # yest = 0.
    #
    # n = 0
    #
    # prdcnt = 0
    # nacf = 0
    # cnt = 0
    # cmpcnt = 0
    # rhocnt = 0
    # runcnt = 0
    #

    #
    # for i in range(0, 5):
    #     for j in range(0, 5):
    #         Q[i, j] = 0.
    #
    # for i in range(0, 5):
    #     C[i] = 0.
    #
    # for i in range(0, 5):
    #     xest[i] = 0.
    #
    # for i in range(0, 5):
    #     for j in range(0, 5):
    #         pest[i, j] = 0.
























