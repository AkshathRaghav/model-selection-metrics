def calculate_pac_dir(features_np_all, label_np_all, alpha=1.):
  """Compute the PACTran-Dirichlet estimator."""
  prob_np_all,_ = gmm_estimator(features_np_all, label_np_all)
#   starttime = time.time()
  label_np_all = one_hot(label_np_all)  # [n, v]
  soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
  soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]
  a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10

  # initialize
  qz = prob_np_all  # [n, d]
  log_s = np.log(prob_np_all + 1e-10)  # [n, d]

  for _ in range(10):
    aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz), axis=0)
    logits_qz = (log_s +
                 np.matmul(label_np_all, scipy.special.digamma(aw)) -
                 np.reshape(scipy.special.digamma(np.sum(aw, axis=0)), [1, -1]))
    log_qz = logits_qz - scipy.special.logsumexp(
        logits_qz, axis=-1, keepdims=True)
    qz = np.exp(log_qz)

  log_c0 = scipy.special.loggamma(np.sum(a0)) - np.sum(
      scipy.special.loggamma(a0))
  log_c = scipy.special.loggamma(np.sum(aw, axis=0)) - np.sum(
      scipy.special.loggamma(aw), axis=0)

  pac_dir = np.sum(
      log_c0 - log_c - np.sum(qz * (log_qz - log_s), axis=0))
  pac_dir = -pac_dir / label_np_all.size
  return pac_dir


def calculate_pac_gamma(features_np_all, label_np_all, alpha=1.):
    """Compute the PAC-Gamma estimator."""
    prob_np_all,_ = gmm_estimator(features_np_all, label_np_all)
    #   starttime = time.time()
    label_np_all = one_hot(label_np_all)  # [n, v]
    soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
    soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]

    a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10
    beta = 1.

    # initialize
    qz = prob_np_all  # [n, d]
    s = prob_np_all  # [n, d]
    log_s = np.log(prob_np_all + 1e-10)  # [n, d]
    aw = a0
    bw = beta
    lw = np.sum(s, axis=-1, keepdims=True) * np.sum(aw / bw)  # [n, 1]

    for _ in range(10):
        aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz),
                        axis=0)  # [v, d]
        lw = np.matmul(
            s, np.expand_dims(np.sum(aw / bw, axis=0), axis=1))  # [n, 1]
        logits_qz = (
            log_s + np.matmul(label_np_all, scipy.special.digamma(aw) - np.log(bw)))
        log_qz = logits_qz - scipy.special.logsumexp(
            logits_qz, axis=-1, keepdims=True)
        qz = np.exp(log_qz)  # [n, a, d]

    pac_gamma = (
        np.sum(scipy.special.loggamma(a0) - scipy.special.loggamma(aw) +
                aw * np.log(bw) - a0 * np.log(beta)) +
        np.sum(np.sum(qz * (log_qz - log_s), axis=-1) +
                np.log(np.squeeze(lw, axis=-1)) - 1.))
    pac_gamma /= label_np_all.size
    pac_gamma += 1.
#   endtime = time.time()
    return pac_gamma


def calculate_pac_gauss(features_np_all, label_np_all,
                        lda_factor = 1.):
    """Compute the PAC_Gauss score with diagonal variance."""
    starttime = time.time()
    nclasses = label_np_all.max()+1
    label_np_all = one_hot(label_np_all)  # [n, v]
    
    mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
    features_np_all -= mean_feature  # [n,k]

    bs = features_np_all.shape[0]
    kd = features_np_all.shape[-1] * nclasses
    ldas2 = lda_factor * bs  # * features_np_all.shape[-1]
    dinv = 1. / float(features_np_all.shape[-1])

    # optimizing log lik + log prior
    def pac_loss_fn(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        xent = np.sum(np.sum(
            label_np_all * (np.log(label_np_all + 1e-10) - log_qz), axis=-1)) / bs
        loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
        return loss

    # gradient of xent + l2
    def pac_grad_fn(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
        grad_f -= label_np_all
        grad_f /= bs
        grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
        grad_w += w / ldas2

        grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
        grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
        return grad

    # 2nd gradient of theta (elementwise)
    def pac_grad2(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
        grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
        xx = np.square(features_np_all)  # [n, d]

        grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
        grad2_w += 1. / ldas2
        grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
        grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
        return grad2

    kernel_shape = [features_np_all.shape[-1], nclasses]
    theta = np.random.normal(size=kernel_shape) * 0.03
    theta_1d = np.ravel(np.concatenate(
        [theta, np.zeros([1, nclasses])], axis=0))

    theta_1d = scipy.optimize.minimize(
        pac_loss_fn, theta_1d, method="L-BFGS-B",
        jac=pac_grad_fn,
        options=dict(maxiter=100), tol=1e-6).x

    pac_opt = pac_loss_fn(theta_1d)
    endtime_opt = time.time()

    h = pac_grad2(theta_1d)
    sigma2_inv = np.sum(h) * ldas2  / kd + 1e-10
    endtime = time.time()

    if lda_factor == 10.:
        s2s = [1000., 100.]
    elif lda_factor == 1.:
        s2s = [100., 10.]
    elif lda_factor == 0.1:
        s2s = [10., 1.]
        
    returnv = []
    for s2_factor in s2s:
        s2 = s2_factor * dinv
        pac_gauss = pac_opt + 0.5 * kd / ldas2 * s2 * np.log(
            sigma2_inv)
        
        # the first item is the pac_gauss metric
        # the second item is the linear metric (without trH)
        returnv += [("pac_gauss_%.1f" % lda_factor, pac_gauss),
                    ("time", endtime - starttime),
                    ("pac_opt_%.1f" % lda_factor, pac_opt),
                    ("time", endtime_opt - starttime)]
    return returnv, theta_1d