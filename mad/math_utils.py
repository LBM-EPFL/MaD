import warnings
import numpy as np
from sklearn.metrics import roc_auc_score

def unit_vector(vec):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            v = np.asarray(vec) / np.sqrt(np.dot(vec, vec))
        except Warning:
            print("MaD> ERROR: can't normalize vec", vec)
            return vec
    return v

def euler_rod_mat(axis, angle):
    # Following Euler-Rodriguez' formula
    a = np.cos(angle / 2.0)

    axis = np.asarray(axis)

    b, c, d = - axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def get_rototrans_SVD(mobile, reference):
    # Check shape
    m = mobile.shape
    n = reference.shape
    if m != n or not(m[1] == n[1] == 3):
        raise Exception("Descript> ERROR: Coordinates mismatch for SVD")
    n = reference.shape[0]

    # center
    av1 = sum(mobile) / n
    av2 = sum(reference) / n
    mcoords = mobile - av1
    rcoords = reference - av2

    # correlation matrix
    a = np.dot(mcoords.T, rcoords)
    u, d, vt = np.linalg.svd(a)
    R = np.dot(vt.T, u.T).T

    # check if we have found a reflection
    if np.linalg.det(R) < 0:
        vt[2] = -vt[2]
        R = np.dot(vt.T, u.T).T
    T = av2 - np.dot(av1, R)
    return R, T

def polar_to_cart(theta, phi):
        return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])

def bc_scoring(y, p):
    # prediction
    q = np.round(p)

    # total positive and negatives
    TP = np.sum(q * y)
    TN = np.sum((1.0-q) * (1.0-y))
    FP = np.sum(q * (1.0-y))
    FN = np.sum((1.0-q) * y)

    # auc
    if (np.all(y > 0.5)) or (np.all(y < 0.5)):
        auc = np.nan
    elif np.any(np.isnan(y)) or np.any(np.isnan(p)):
        auc = np.nan
    else:
        auc = roc_auc_score(y, p)

    # store results
    return {
        'bra': (1.0 - np.mean(y)),
        'acc': ((TP + TN) / (TP + TN + FP + FN + 1e-6)),
        'ppv': (TP / (TP + FP + 1e-6)),
        'tpr': (TP / (TP + FN + 1e-6)),
        'tnr': (TN / (TN + FP + 1e-6)),
        'mcc': (((TP*TN) - (FP*FN)) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + 1e-6)),
        'auc': auc,
        'std': np.std(p),
    }

def mcc_scoring(y, p):
    max_mcc = 0
    mcc = []
    thresholds = np.arange(0, 1.001, 0.05)
    for t in thresholds:
        # prediction
        q = (p > t).astype(int)

        # total positive and negatives
        TP = np.sum(q * y)
        TN = np.sum((1.0-q) * (1.0-y))
        FP = np.sum(q * (1.0-y))
        FN = np.sum((1.0-q) * y)
        mcc.append((((TP*TN) - (FP*FN)) / (np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + 1e-6)))
        if mcc[-1] > max_mcc:
            data = [TP, FP, FN, TN, mcc[-1], t]
            max_mcc = mcc[-1]
    return mcc, data

def precision_scoring(y, p):
    prec = []
    max_prec = 0
    thresholds = np.arange(0, 1.001, 0.05)
    for t in thresholds:
        # prediction
        q = (p > t).astype(int)

        # total positive and negatives
        TP = np.sum(q * y)
        TN = np.sum((1.0-q) * (1.0-y))
        FP = np.sum(q * (1.0-y))
        FN = np.sum((1.0-q) * y)
        prec.append(TP / (TP + FP + 1e-6))
        if prec[-1] > max_prec:
            data = [TP, FP, FN, TN, prec[-1], t]
            max_prec = prec[-1]
    return prec, data

def f1_scoring(y, p):
    f1 = []
    thresholds = np.arange(0, 1.001, 0.05)
    for t in thresholds:
        # prediction
        q = (p > t).astype(int)

        # total positive and negatives
        TP = np.sum(q * y)
        TN = np.sum((1.0-q) * (1.0-y))
        FP = np.sum(q * (1.0-y))
        FN = np.sum((1.0-q) * y)
        prec = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        val = 2 * prec * recall / (prec + recall)
        f1.append(val)
    return f1
