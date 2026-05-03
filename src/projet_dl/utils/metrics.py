import numpy as np
from sklearn.metrics import confusion_matrix


def compute_icbhi_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    sensitivities = []
    specificities = []

    for i in range(4):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        se = tp / (tp + fn) if (tp + fn) else 0.0
        sp = tn / (tn + fp) if (tn + fp) else 0.0
        sensitivities.append(float(se))
        specificities.append(float(sp))

    macro_se = float(np.mean(sensitivities))
    macro_sp = float(np.mean(specificities))
    return {
        "confusion_matrix": cm,
        "sensitivity": macro_se,
        "specificity": macro_sp,
        "icbhi_score": (macro_se + macro_sp) / 2.0,
        "sensitivities_per_class": sensitivities,
        "specificities_per_class": specificities,
    }
