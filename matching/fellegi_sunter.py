
import math
import operator
import numpy as np
import jellyfish


def field_w(a, b, m, u):
    return math.log(m/u, 2) if a == b else math.log((1.0 - m)/(1.0 - u))


def fs_weights(ra, rb, m_u):
    return sum( field_w(ra[i], rb[i], m_u[i][1], m_u[i][2]) for i in range(len(ra)) )


def gamma_pattern(r_ab):
    t1_0 = lambda x: 1 if x else 0
#    return [1 if a_i == b_i else 0 for (a_i, b_i) in zip(r_ab[0], r_ab[1])]
    return [t1_0(jellyfish.jaro_winkler(r_ab[0][0], r_ab[1][0]) > 0.9),
            t1_0(r_ab[0][1] == r_ab[1][1]),
            t1_0(r_ab[0][2] == r_ab[1][2])]


def expectation_step(mh, uh, ph, gammas):
    
    # We calculate an estimate of the indicator function g for every record in the sample
    g_m = np.zeros(len(gammas))
    g_u = np.zeros(len(gammas))

    # for each record, need the gamma pattern
    for j, gamma in enumerate(gammas):
        m_product = reduce(operator.mul, (mh[i] ** gamma[i] * (1 - mh[i]) ** (1 - gamma[i]) for i in range(len(gamma))))
        u_product = reduce(operator.mul, (uh[i] ** gamma[i] * (1 - uh[i]) ** (1 - gamma[i]) for i in range(len(gamma))))

        g_m[j] = (ph * m_product) / (ph * m_product + (1-ph) * u_product)
        g_u[j] = ((1-ph) * u_product) / (ph * m_product + (1-ph) * u_product)

    return g_m, g_u


def maximization_step(gm, gu, gammas):

    sum_g_m, sum_g_u = 0, 0
    m_h = np.zeros(len(gammas[0]))
    u_h = np.zeros(len(gammas[0]))

    for j, gamma in enumerate(gammas):
        m_h += [gm[j] * gamma[i] for i in range(len(gamma))]
        u_h += [gu[j] * gamma[i] for i in range(len(gamma))]

        sum_g_m += gm[j]
        sum_g_u += gu[j]

    return m_h/sum_g_m, u_h/sum_g_u, sum_g_m/len(gammas)


def fs_em(record_pairs, mh, uh, ph, gamma_pattern_func=None):
    gammas = [gamma_pattern(r_ab) for r_ab in record_pairs]

    for i in range(20):
        g_m, g_u = expectation_step(mh, uh, ph, gammas)
        mh_last = mh
        mh, uh, ph = maximization_step(g_m, g_u, gammas)
        #print(mh, uh, ph)
        if sum((mc - ml)**2 for mc, ml in zip(mh, mh_last)) < 1e-6:
            break
    return mh, uh, ph


def fs_em_dataframe(df1, df2, mh, uh, ph, max_step=20, gamma_fields=None):
    
    #dfm = df1.merge(df2, on='nzip')
    gammas = df1[['g1','g2']].values

    for i in range(max_step):
        g_m, g_u = expectation_step(mh, uh, ph, gammas)
        mh_last = mh
        mh, uh, ph = maximization_step(g_m, g_u, gammas)
        #print(mh, uh, ph)
        if sum((mc - ml)**2 for mc, ml in zip(mh, mh_last)) < 1e-8:
            break
    return mh, uh, ph

