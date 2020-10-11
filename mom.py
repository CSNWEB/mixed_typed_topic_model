import scipy.special
import numpy as np
import opt_einsum as oe


def sample_spherical(ndim=3):
    vec = np.random.rand(ndim)
    vec /= np.linalg.norm(vec)
    return vec


def mapA(T, A):
    p1 = np.tensordot(A, T, axes=(0, 0))
    p2 = np.tensordot(p1, A, axes=(1, 0))
    result = np.tensordot(p2, A, axes=(1, 0))
    return result


def mapAv(T, A, v):
    p1 = np.tensordot(A, T, axes=(0, 0))
    p2 = np.tensordot(p1, v, axes=(1, 0))
    result = np.tensordot(p2, v, axes=(1, 0))
    return result


def mapv(T, v):
    p1 = np.tensordot(v, T, axes=(0, 0))
    p2 = np.tensordot(p1, v, axes=(0, 0))
    result = np.tensordot(p2, v, axes=(0, 0))
    return result

# w_matrix adjusted from https://github.com/mruffini/SpectralMethod/blob/master/OtherMethods.py#L107


def w_matrix(M_i, k):
    u, s, _ = np.linalg.svd(M_i)
    u = u[:, :k]
    s = s[:k]
    sqroots = np.sqrt(s)
    W = u.dot(np.diag(1/sqroots))
    B = u.dot(np.diag(sqroots))
    return([W, B])


def whiten(T, W):
    return oe.contract('ijk,ia,jb,kc->abc', T, W, W, W)


def decompose(T, B, k):
    L = 150
    N = 50
    I = np.eye(k)
    mu = np.zeros(B.shape)
    eignvalues = np.zeros((k))
    eignvectors = np.zeros((k, k))
    for i in range(k):
        bestLamb = -1
        for l in range(L):
            theta = sample_spherical(k)
            for n in range(N):
                theta = mapAv(T, I, theta)
                theta = theta / np.linalg.norm(theta)
            lamb = mapv(T, theta)
            if (lamb > bestLamb):
                bestLamb = lamb
                bestTheta = theta
        theta = bestTheta
        for n in range(N):
            theta = mapAv(T, I, theta)
            theta = theta / np.linalg.norm(theta)
        lamb = mapv(T, theta)
        print(lamb)
        print(theta)
        mu[:, i] = lamb.dot(B).dot(theta)
        eignvalues[i] = lamb
        eignvectors[i] = theta
        T = T-lamb * np.tensordot(np.tensordot(theta,
                                               theta, axes=0), theta, axes=0)
    return [mu, eignvalues, eignvectors]


def fit(x, n_c, alpha0, k):
    d = x.shape[1]
    # number of documents
    n = x.shape[0]
    n_d = d - n_c

    # Concatenation of mu quer and p quer
    mean = np.zeros((d))
    if n_c > 0:
        # mu quer
        mean[:n_c] = x[:, :n_c].sum(axis=0) / n
    if n_d > 0:
        numOfWords = x[:, n_c:].sum()
        # p quer
        mean[n_c:] = x[:, n_c:].sum(axis=0) / numOfWords
    I = np.eye(d)

    m2_correction = oe.contract('i,j->ij', (mean), (mean)) * (alpha0/(alpha0+1))

    if n_d > 0:
        # liste von x quer's
        row_sums = x[:, n_c:].sum(axis=1)
        # list of s_dd's
        binoms = 1/scipy.special.binom(row_sums, 2) * 0.5

    c1d0 = np.concatenate((np.ones((n_c)), np.zeros((n_d))))
    c0d1 = np.concatenate((np.zeros((n_c)), np.ones((n_d))))

    masks_i = []
    masks_j = []

    scalings = []
    if n_c > 0:
        masks_i.append(c1d0)
        masks_j.append(c1d0)
        scalings.append(np.ones((n)))
    if n_c > 0 and n_d > 0:
        masks_i.append(c0d1)
        masks_i.append(c1d0)
        masks_j.append(c1d0)
        masks_j.append(c0d1)
        scalings.append(1/row_sums)
        scalings.append(1/row_sums)
    if n_d > 0:
        masks_i.append(c0d1)
        masks_j.append(c0d1)
        scalings.append(binoms)

    mask_i = np.stack(masks_i)
    mask_j = np.stack(masks_j)
    mask_h = np.stack(scalings)

    m2 = oe.contract('hi,hj,sh,si,sj->ij', x, x, mask_h, mask_i, mask_j)

    if n_d > 0:
        m2[n_c:, n_c:] = m2[n_c:, n_c:] - np.diag(oe.contract('i,ij->j', binoms, x[:, n_c:]))
    m2 = m2 / n

    if n_c > 0:
        cov = np.cov(x[:, :n_c].T)
        [lambdas, v] = np.linalg.eigh(cov)
        smallest_index = np.argmin(lambdas)
        # sigma quer
        sigma = lambdas[smallest_index]
        maskSigma = np.zeros((d, d))
        maskSigma[:n_c, :n_c] = I[:n_c, :n_c]
        m2 -= sigma * maskSigma

        # Calculate m1 subterms, eta_ccc
        m1_sub1 = x[:, :n_c] - mean[:n_c]
        m1_sub2 = oe.contract('hi,i->h', m1_sub1, v[:, smallest_index])
        m1_sub3 = oe.contract('h,h->h', m1_sub2, m1_sub2)
        # Calculate final m1, eta_ccc
        m1 = oe.contract('h,hi->i', m1_sub3, x[:, :n_c])/n
        eta_dcc = oe.contract('h,hi->i', m1_sub3, x[:, n_c:])/n

    # c * M in thesis to make un-whitening easier
    m2_corrected = (2/(alpha0+2)) * (m2 - m2_correction)

    [W, B] = w_matrix(m2_corrected, k)

    if n_d > 0:
        # list of s_ddd's
        binoms3 = 1/scipy.special.binom(row_sums, 3) * (1/6)
    if n_c > 0 and n_d > 0:
        mask_i = np.stack((c1d0, c0d1, c1d0, c0d1, c1d0, c0d1, c1d0, c0d1))
        mask_j = np.stack((c1d0, c1d0, c0d1, c0d1, c1d0, c1d0, c0d1, c0d1))
        mask_k = np.stack((c1d0, c1d0, c1d0, c1d0, c0d1, c0d1, c0d1, c0d1))
        mask_h = np.stack((np.ones((n)),  1/row_sums,  1/row_sums, binoms,
                           1/row_sums, binoms, binoms, binoms3))
        # E [z \otimes z \otimes z], but with scalings s_ccc to s_ddd and whitening
        m3 = oe.contract('hi,hj,hk,sh,si,sj,sk,ia,jb,kc->abc',
                         x, x, x, mask_h, mask_i, mask_j, mask_k, W, W, W)

        scaled_eta_dcc = oe.contract('i,j->j', 1/row_sums, eta_dcc)
        repeated_sigma_exd = np.zeros((d, d))
        repeated_sigma_exd[:n_c, n_c:] = np.tile(scaled_eta_dcc, (n_c, 1))
        # E_dcc and its transposes
        cont_corr2 = oe.contract('ij,ia,ib,jc->abc', repeated_sigma_exd, W, W, W)
        cont_corr2 += oe.contract('ij,ia,jb,ic->abc', repeated_sigma_exd, W, W, W)
        cont_corr2 += oe.contract('ij,ja,ib,ic->abc', repeated_sigma_exd, W, W, W)

        #E_cdd and transposes
        cdd_elements = np.zeros((d, d))
        cdd_elements[n_c:, :n_c] = oe.contract('i,ij,ik->jk', binoms, x[:, n_c:], x[:, :n_c])
        cdd_correction = oe.contract('ij,ia,ib,jc->abc', cdd_elements, W, W, W)
        cdd_correction += oe.contract('ij,ia,jb,ic->abc', cdd_elements, W, W, W)
        cdd_correction += oe.contract('ij,ja,ib,ic->abc', cdd_elements, W, W, W)

        m3 = m3 - cont_corr2 - cdd_correction

    elif n_c > 0:
        m3 = oe.contract('hi,hj,hk,ia,jb,kc->abc', x, x, x, W, W, W)
    else:
        m3 = oe.contract('hi,hj,hk,h,ia,jb,kc->abc', x, x, x, binoms3, W, W, W)

    if n_d > 0:
        # calculate E_ddd diag part
        diag_elements = oe.contract('i,ij->j', binoms3, x[:, n_c:])
        m3 = m3 + 2 * oe.contract('i,ia,ib,ic->abc',
                                  np.concatenate((np.zeros((n_c)), diag_elements)), W, W, W)

        # E_ddd off_diag part
        # E[\overlin[x] \otimes \overline[x]] * s_ddd, zero padded
        outer_x = np.zeros((d, d))
        outer_x[n_c:, n_c:] = oe.contract('hi,hj,h->ij', x[:, n_c:], x[:, n_c:], binoms3)
        off_diag = np.zeros((k, k, k))
        # Sum three times for pi_3
        off_diag += oe.contract('ij,ia,ib,jc->abc', outer_x, W, W, W)
        off_diag += oe.contract('ij,ia,jb,ic->abc', outer_x, W, W, W)
        off_diag += oe.contract('ij,ia,jb,jc->abc', outer_x, W, W, W)
        m3 = m3 - off_diag

    m3 = m3 / n

    if n_c > 0:
        #  calculate E_ccc
        repeated_m1 = np.zeros((d, d))
        # copy m1 c times
        repeated_m1[:n_c, :n_c] = np.tile(m1, (n_c, 1))
        cont_corr = oe.contract('ij,ia,ib,jc->abc', repeated_m1, W, W, W)
        cont_corr += oe.contract('ij,ia,jb,ic->abc', repeated_m1, W, W, W)
        cont_corr += oe.contract('ij,ja,ib,ic->abc', repeated_m1, W, W, W)

        # sigma_exd = sigma * mean[n_c:]
        # repeated_sigma_exd = np.zeros((d, d))
        # repeated_sigma_exd[:n_c, n_c:] = np.tile(sigma_exd, (n_c, 1))
        # print(repeated_sigma_exd)
        # # cont_corr += oe.contract('ij,ia,ib,jc->abc', repeated_sigma_exd, W, W, W)
        # cont_corr += oe.contract('ij,ia,jb,ic->abc', repeated_sigma_exd, W, W, W)
        # cont_corr += oe.contract('ij,ja,ib,ic->abc', repeated_sigma_exd, W, W, W)

        m3 -= cont_corr

    # Here m2 is is [[H_mumu, H_mup], [H_pmu, H_pp]]
    correction_subterm_w = oe.contract('ij,k,ia,jb,kc->abc', m2, mean, W, W, W)
    m3_correction_w = (-alpha0 / (alpha0 + 2)) * (
        correction_subterm_w + correction_subterm_w.swapaxes(1, 2) + correction_subterm_w.swapaxes(0, 2)
    ) + (2 * alpha0 * alpha0 / ((alpha0 + 2)*(alpha0+1))) * \
        oe.contract('i,j,k,ia,jb,kc->abc', mean, mean, mean, W, W, W)
    m3_corrected = m3 + m3_correction_w

    # mu is theta so concatenation of mu and p
    [mu, eignvalues, eignvectors] = decompose(m3_corrected, B, k)
    decomposed_alphas = np.zeros(k)
    for i in range(k):
        decomposed_alphas[i] = 1 / (eignvalues[i] * eignvalues[i]) * \
            (alpha0+2) * (alpha0+1) * alpha0 / 2

    if n_d > 0:
        mu_min = mu[n_c:, :].min()
        print(mu_min)
        print(mu.max())
        if mu_min < 0:
            positive_mu = mu[n_c:, :] - mu_min
        else:
            positive_mu = mu[n_c:, :]

        mu[n_c:, :] = positive_mu / positive_mu.sum(axis=0)

    if n_c > 0:
        vectors = np.zeros((n_c, k))
        ea0 = decomposed_alphas.sum()
        print(vectors.shape)
        print(mu.shape)
        print(mu[:n_c].shape)
        for i in range(k):
            vectors[:, i] = mu[:n_c].dot(decomposed_alphas)
            vectors[:, i] += 2 * mu[:n_c, i]
            vectors[:, i] *= (decomposed_alphas[i]*(decomposed_alphas[i]+1)/((ea0+1)*(ea0+2)*ea0))
        sigmas = np.linalg.pinv(vectors).dot(m1)
    else:
        sigmas = np.array([])

    return [mu, decomposed_alphas, sigmas, m2_corrected, m3_corrected, W]


def test(em2, em3, W, alpha, x, k):
    alpha0 = alpha.sum()
    d = x.shape[1]
    print(d)
    real_m2 = np.zeros((d, d))
    for i in range(k):
        real_m2 = real_m2 + (alpha[i]/((alpha0+1)*alpha0)) * oe.contract('i,j->ij', x[i].T, x[i].T)

    # Add factor correction
    real_m2 = (2/(alpha0+2)) * real_m2

    diff = real_m2-em2
    print("m2 diff and norm")
    print(diff)
    print(np.linalg.norm(diff))

    real_m3 = np.zeros((d, d, d))
    for i in range(k):
        real_m3 += (2*alpha[i]/((alpha0+2)*(alpha0+1)*alpha0)) * \
            oe.contract('i,j,k->ijk', x[i].T, x[i].T, x[i].T)
    diff = whiten(real_m3, W)-em3
    print("m3 diff and norms")
    print(diff)
    print([np.linalg.norm(x) for x in diff])
