# kemans from chatgpt.com 25oct04

import numpy as np
import argparse

# ==========================================
#   Utility: Standardization
# ==========================================

def standardize(X, mean=None, std=None, eps=1e-12):
    """
    Standardize each feature of X to zero mean and unit variance.

    Parameters
    ----------
    X : ndarray (n, d)
    mean : optional precomputed mean (d,)
    std : optional precomputed std (d,)
    eps : small constant to avoid division by zero

    Returns
    -------
    Xs : standardized array
    mean, std : per-feature mean and std used
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    return (X - mean) / np.maximum(std, eps), mean, std

# ==========================================
#   Core K-means Implementation
# ==========================================

def _pairwise_sq_dists(X, C):
    X2 = np.sum(X * X, axis=1, keepdims=True)
    C2 = np.sum(C * C, axis=1, keepdims=True).T
    XC = X @ C.T
    return np.maximum(X2 + C2 - 2 * XC, 0.0)

def _kmeans_plusplus_init(X, k, rng):
    n, d = X.shape
    centers = np.empty((k, d))
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    closest_sq = _pairwise_sq_dists(X, centers[0:1]).reshape(-1)
    for i in range(1, k):
        probs = closest_sq / closest_sq.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        new_d = _pairwise_sq_dists(X, centers[i:i+1]).reshape(-1)
        closest_sq = np.minimum(closest_sq, new_d)
    return centers

def _weighted_means(X, labels, k, sample_weight):
    """
    Compute cluster means weighted by sample_weight.
    """
    d = X.shape[1]
    centers = np.zeros((k, d))
    for j in range(k):
        mask = (labels == j)
        w = sample_weight[mask]
        if w.size > 0:
            centers[j] = np.average(X[mask], axis=0, weights=w)
        else:
            centers[j] = X[np.argmax(sample_weight)]
    return centers

def _weighted_stds(X, labels, k, sample_weight):
    """
    Compute cluster standard deviations weighted by sample_weight.
    Returns array of shape (k, d) where d is the number of features.
    """
    d = X.shape[1]
    stds = np.zeros((k, d))
    for j in range(k):
        mask = (labels == j)
        w = sample_weight[mask]
        if w.size > 0:
            # Weighted mean
            mean_j = np.average(X[mask], axis=0, weights=w)
            # Weighted variance
            var_j = np.average((X[mask] - mean_j)**2, axis=0, weights=w)
            stds[j] = np.sqrt(var_j)
        else:
            # Default to global weighted std if cluster empty
            mean_global = np.average(X, axis=0, weights=sample_weight)
            var_global = np.average((X - mean_global)**2, axis=0, weights=sample_weight)
            stds[j] = np.sqrt(var_global)
    return stds

def kmeans(
    X, k, *,
    sample_weight=None,
    n_init=10,
    max_iter=300,
    tol=1e-6,
    init="k-means++",
    random_state=None,
    return_history=False,
):
    """
    Weighted K-means clustering with optional multiple restarts.

    Parameters
    ----------
    X : ndarray (n, d)
    sample_weight : (n,) or None (uniform=1)
    k, n_init, max_iter, tol, init, random_state : standard
    return_history : whether to return center drifts

    Returns
    -------
    centers, labels, inertia, info
    """
    X = np.asarray(X, float)
    n, d = X.shape
    if not (1 <= k <= n):
        raise ValueError("k must be in [1, n_samples].")

    if sample_weight is None:
        sample_weight = np.ones(n)
    sample_weight = np.asarray(sample_weight, float)
    sample_weight /= np.sum(sample_weight)  # normalize total weight

    rng_base = np.random.default_rng(random_state)
    best = {"inertia": np.inf}

    for run in range(n_init):
        #print("***** run",run)
        rng = np.random.default_rng(rng_base.integers(0, 2**63 - 1))
        if isinstance(init, np.ndarray):
            centers = init.copy()
        elif init == "k-means++":
            centers = _kmeans_plusplus_init(X, k, rng)
        elif init == "random":
            centers = X[rng.choice(n, size=k, replace=False)]
        else:
            raise ValueError("Invalid init type.")

        history = []
        for it in range(1, max_iter + 1):
            d2 = _pairwise_sq_dists(X, centers)
            labels = np.argmin(d2, axis=1)

            new_centers = _weighted_means(X, labels, k, sample_weight)
            stds = _weighted_stds(X, labels, k, sample_weight) # compute std deviation            
            drift = np.linalg.norm(new_centers - centers)
            history.append(drift)
            centers = new_centers

            #print("norm",drift, tol, np.linalg.norm(centers))
            if drift <= tol * max(1.0, np.linalg.norm(centers)):
                break

        inertia = np.sum(sample_weight * np.min(_pairwise_sq_dists(X, centers), axis=1))
        #print("inertia",inertia)
        if inertia < best["inertia"]:
            #print("update best:",inertia, best["inertia"])
            best = {
                "centers": centers.copy(),
                "stds": stds.copy(),
                "labels": labels.copy(),
                "inertia": inertia,
                "n_iter": it,
                "history": history if return_history else None,
            }

    info = {"n_iter": best["n_iter"]}
    if return_history:
        info["history"] = best["history"]
    return best["centers"], best["stds"], best["labels"], best["inertia"], info

def kmeans_predict(X, centers):
    d2 = _pairwise_sq_dists(np.asarray(X, float), np.asarray(centers, float))
    return np.argmin(d2, axis=1)

# ==========================================
#   Example Usage
# ==========================================
def synTest():
    rng = np.random.default_rng(0)
    # Create synthetic data
    a = rng.normal([0, 0], 0.5, (200, 2))
    b = rng.normal([3, 3], 0.5, (200, 2))
    c = rng.normal([0, 4], 0.5, (200, 2))
    X = np.vstack([a, b, c])

    # Standardize
    #Xs, mean, std = standardize(X)

    # Weighted k-means
    w = np.ones(X.shape[0])
    centers, stds, labels, inertia, info = kmeans(
        X, k=3, sample_weight=w, random_state=42, return_history=True
    )

    print("Centers:\n", centers)
    print("Inertia:", inertia)
    print("Iterations:", info["n_iter"])
    print("info", info)
    print("labels",labels)

################################
def main( args ):

    # cycle through list of tsv objects and read into one big array
    arrays = []
    for fname in open(args.dataListTSV).read().splitlines():
        print(f"Loading {fname} ...")
        arr = np.loadtxt(fname)
        arrays.append(arr)
    # Handle the case of single file gracefully
    if len(arrays) == 1:
        data= arrays[0]
    else:
        # Stack vertically (row-wise)
        data= np.vstack(arrays)

    #### compute kmeans
    centers, stds, labels, inertia, info = kmeans(data, k=args.numCenters, random_state=42, return_history=True)
    print("Inertia:", inertia)
    print("Iterations:", info["n_iter"])
    print("info", info)
    print("labels",labels)
    
    #### output to outputfile:
    #### centerNum, "mean", vectorDat
    #### centerNum, "sd", vectorDat
    ofp = open(args.outputfile,"w")
    for kk in range(args.numCenters):
        print('%d,"mean",%s' % (kk, ",".join([ str(xx) for xx in centers[kk]])), file=ofp)
        print('%d,"sd",%s'   % (kk, ",".join([ str(xx) for xx in stds[kk]])), file=ofp)
    ofp.close()
    
if __name__ == "__main__":
    #synTest()
    parser = argparse.ArgumentParser()
    parser.add_argument("--numCenters", type=int, help="kmeans number of centers")
    parser.add_argument("--outputfile", help="file to output mean/sd vectors")
    parser.add_argument("--dataListTSV", help="list of TSV input objects viz: XX_00.embeddings.")
    args = parser.parse_args()
    main(args)

