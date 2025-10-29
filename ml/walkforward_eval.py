import numpy as np
from sklearn.metrics import roc_auc_score

def walk_forward_eval(model, X, y, window=500, step=60):
    aucs = []
    total_folds = 0
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]


    for start in range(0, len(X) - window, step):
        X_train = X.iloc[start:start + window - 120]
        X_val   = X.iloc[start + window - 120:start + window]
        y_train = y.iloc[start:start + window - 120]
        y_val   = y.iloc[start + window - 120:start + window]

        # skip invalid folds (all labels same)
        if len(np.unique(y_val)) < 2:
            continue

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]

        try:
            auc = roc_auc_score(y_val, preds)
            aucs.append(auc)
        except ValueError:
            continue

        total_folds += 1

    valid_aucs = [a for a in aucs if not np.isnan(a)]
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0

    print(f"📊 Evaluated {total_folds} folds, valid AUCs = {len(valid_aucs)}")
    print(f"📈 Walk-forward mean AUC: {mean_auc:.3f}")
    return valid_aucs
