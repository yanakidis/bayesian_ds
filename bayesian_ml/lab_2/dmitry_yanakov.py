import numpy as np
from scipy import signal

eps = 1e-20

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    H, W, K = X.shape
    h, w = F.shape

    tmp_1 = 2 * s ** 2
    tmp_2 = (X - B[:, :, None]) ** 2
    ones = np.ones((h, w))[:, :, None]

    B_tmp = np.sum(tmp_2, axis=(0, 1)) / tmp_1

    F_tmp = (signal.fftconvolve(X ** 2, ones[::-1, ::-1, ::-1], mode='valid') -
             2 * signal.fftconvolve(X, F[:, :, None][::-1, ::-1, ::-1], mode='valid')) / tmp_1

    ll = H * W * np.log(1 / (np.sqrt(2 * np.pi) * s)) - F_tmp - B_tmp[None, None, :] - (F ** 2).sum() / tmp_1 + \
         signal.fftconvolve(tmp_2 / tmp_1, ones[::-1, ::-1, ::-1], mode='valid')

    return ll



def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    K = X.shape[2]

    tmp = np.log(eps + A[:, :, None]) + calculate_log_probability(X, F, B, s)

    if not use_MAP:
        mask = q != 0

        L = np.sum(q[mask] * tmp[mask] - q[mask] * np.log(q[mask]))
    else:
        L = sum([tmp[q[0, i], q[1, i], i] for i in range(K)])  # 1 * tmp[q[0, i], q[1, i], i]
    return L


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    H, W, K = X.shape

    tmp = np.log(eps + A[:, :, None]) + calculate_log_probability(X, F, B, s)

    if not use_MAP:
        tmp_2 = np.exp(tmp - np.max(tmp, axis=(0, 1)))
        q = tmp_2 / np.sum(tmp_2, axis=(0, 1))

        return q
    else:
        rows = []
        cols = []

        for i in range(K):
            arg = tmp[:, :, i].argmax()

            size = tmp[:, :, i].shape[1]

            rows.append(arg // size)
            cols.append(arg - (arg // size) * size)

        return np.array([rows, cols])

def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape

    if not use_MAP:
        A = np.sum(q, axis=2) / K

        F = signal.fftconvolve(X.transpose(2, 0, 1), q.transpose(2, 0, 1)[::-1, ::-1, ::-1], mode='valid') / K
        F = F[0]

        m = signal.fftconvolve(q, np.ones((h, w))[:, :, None])
        B = np.sum((X * (1 - m)) / np.sum((1 - m), axis=2)[:, :, None], axis=2)

        s = ((signal.fftconvolve(X ** 2, np.ones((h, w))[:, :, None][::-1, ::-1, ::-1], mode='valid') -
              2 * signal.fftconvolve(X, F[:, :, None][::-1, ::-1, ::-1], mode='valid')) * q).sum() + \
            ((1 - m) * (X - B[:, :, None]) ** 2).sum() + (q * (F ** 2).sum()).sum()

        s = np.sqrt((s + eps) / (H * W * K))
    else:
        F = np.zeros((h, w))
        A = np.zeros((H - h + 1, W - w + 1))
        B = np.sum(X, axis=2)
        tmp = np.full((H, W), K)

        for i in range(K):
            h_coord, w_coord = q[:, i]

            tmp[h_coord:h_coord + h, w_coord:w_coord + w] -= 1
            B[h_coord:h_coord + h, w_coord:w_coord + w] -= X[h_coord:h_coord + h, w_coord:w_coord + w, i]

            F += X[h_coord:h_coord + h, w_coord:w_coord + w, i] / K

            A[h_coord, w_coord] += 1 / K

        mask = tmp != 0
        B[~mask] = 0
        B[mask] = B[mask] / tmp[mask]

        s = 0
        for i in range(K):
            h_coord, w_coord = q[:, i]

            tmp = X[h_coord:h_coord + h, w_coord:w_coord + w, i]

            s += ((tmp - F) ** 2).sum() + \
                 ((X[:, :, i] - B) ** 2).sum() - \
                 ((tmp - B[h_coord:h_coord + h, w_coord:w_coord + w]) ** 2).sum()
        s = np.sqrt((s + eps) / (H * W * K))

    return F, B, s, A



def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    H, W, K = X.shape

    if F is None:
        F = 255 * np.random.randn(h, w) # np.random.randint(0, 255, (h, w))

    if B is None:
        B = 255 * np.random.randn(H, W)  #np.random.randint(0, 255, (H, W))

    if s is None:
        s = 200 * np.random.random()

    if A is None:
        A = np.ones((H - h + 1, W - w + 1)) / ((H - h + 1) * (W - w + 1))

    LL = [-np.inf]
    for i in range(max_iter):
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)

        lb = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        LL.append(lb)

        if LL[-1] - LL[-2] < tolerance:
            break

    return F, B, s, A, np.array(LL[1:])




def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    L = -np.inf

    for i in range(n_restarts):
        F_tmp, B_tmp, s_tmp, A_tmp, LL = run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=tolerance,
                                                max_iter=max_iter, use_MAP=use_MAP)

        if L < LL[-1]:
            F, B, s, A = F_tmp, B_tmp, s_tmp, A_tmp
            L = LL[-1]

    return F, B, s, A, L
