try:
    from statsmodels.stats.libqsturng import psturng
    HAS_GAMES_HOWELL = True
except ImportError:
    psturng = None
    HAS_GAMES_HOWELL = False

try:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    HAS_SKLEARN_DA = True
except ImportError:
    PCA = LinearDiscriminantAnalysis = None
    HAS_SKLEARN_DA = False

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, LeaveOneGroupOut
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN_METRICS = True
except ImportError:
    StratifiedKFold = cross_val_predict = LeaveOneGroupOut = None
    confusion_matrix = classification_report = None
    HAS_SKLEARN_METRICS = False
