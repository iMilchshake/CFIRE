import numpy as np

from cfire.cfire_module import CFIRE
from .metrics import get_rule_size, get_literal_count


def evaluate_cfire(
    cfire: CFIRE,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_val_model_pred: np.ndarray,
    y_test_model_pred: np.ndarray,
) -> dict:

    val_acc = (cfire(X_val) == y_val_model_pred).mean()
    test_acc = (cfire(X_test) == y_test_model_pred).mean()
    rule_size = get_rule_size(cfire.dnf.rules)
    literal_count = get_literal_count(cfire.dnf.rules)

    # TODO: add more metrics

    return {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "rule_size": rule_size,
        "literal_count": literal_count,
    }
