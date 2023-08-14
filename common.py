import lightgbm as lgb
import numpy as np

from scipy.misc import derivative
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from torchvision import datasets, transforms

NUM_CLASS = 10


def load_mnist_train_and_test():
    """Load MNIST dataset."""
    mnist_train = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    mnist_test = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    X_train = mnist_train.train_data.numpy()
    y_train = mnist_train.train_labels.numpy()
    X_test = mnist_test.test_data.numpy()
    y_test = mnist_test.test_labels.numpy()
    return X_train, X_test, y_train, y_test


def load_mnist_pca(n_components:int=50):
    """Load MNIST dataset with PCA."""
    X_train, X_test, y_train, y_test = load_mnist_train_and_test()
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test, y_train, y_test


def load_mnist_pca_train_test_val(n_components:int=50, test_size:float=0.2, val_size:float=0.2):
    """Load MNIST dataset with PCA and train/test split."""
    X_train, X_test, y_train, y_test = load_mnist_pca(n_components=n_components)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def rewrite_label_with_binary_setting(X:np.ndarray, Y:np.ndarray, positive_size:int=200):
    """Rewrite label of MNIST dataset with binary setting.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Label data.
        positive_size (int): Number of positive data.

    Returns:
        X (np.ndarray): Input data.
        y (np.ndarray): Label data.
    """
    # Rewrite label with binary setting
    # 1: Odd, 0: Even
    _Y = Y % 2
    return X, _Y


def rewrite_label_with_pu_setting(X:np.ndarray, Y:np.ndarray, positive_size:int=200):
    """Rewrite label of MNIST dataset with PU setting.
    PU setting is a special case of binary classification where
    the negative class is not observed.
    Label y is 1 if the data is positive, otherwise -1.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Label data.
        positive_size (int): Number of positive data.

    Returns:
        X (np.ndarray): Input data.
        y (np.ndarray): Label data.
    """
    # Rewrite label with PU setting
    # 1: Positive, -1: Unlabeled
    # Only positive label (Y == 1) is rewrited
    # first select unlabeled data from positive data
    # then rewrite label of unlabeled data to -1
    positive_index = np.where(Y == 1)[0]
    unlabeled_index = np.random.choice(
        positive_index, len(positive_index) - positive_size, replace=False
    )
    Y[unlabeled_index] = -1
    Y[Y == 0] = -1
    return X, Y


def rewrite_label_with_pll_setting(X:np.ndarray, Y:np.ndarray, positive_rate:float=0.2):
    """Rewrite label of MNIST dataset with PLL setting.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Label data.
        positive_size (int): Number of positive data.

    Returns:
        X (np.ndarray): Input data.
        y (np.ndarray): Label data.
    """
    # one-hot encoding
    Y = np.eye(NUM_CLASS)[Y]
    Y[np.random.binomial(1, positive_rate, Y.shape) == 1] = 1
    label_count = np.sum(Y, axis=-1)
    while (label_count == 1).any():
        Y[
            label_count == 1,
            np.random.randint(NUM_CLASS, size=(label_count == 1).sum()),
        ] = 1
        label_count = np.sum(Y, axis=-1)
        print((label_count == 1).sum())
    return X, Y


def rewrite_label_with_mil_setting(X:np.ndarray, Y:np.ndarray, bag_size:int):
    """Rewrite label of MNIST dataset with PLL setting.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Label data.
        positive_size (int): Number of positive data.

    Returns:
        X (np.ndarray): Input data.
        y (np.ndarray): Label data.
    """
    # one-hot encoding
    use_size = (len(X) // bag_size) * bag_size
    X = X[:use_size]
    Y = Y[:use_size]
    X = X.reshape(use_size // bag_size, bag_size, -1)
    Y = (Y.reshape(use_size // bag_size, bag_size) == 1).any(-1).astype(int)
    return X, Y


def pll_loss_objective(y_pred:np.ndarray, trn_data:lgb.Dataset):
    # Softmaxの計算
    ## p: num_sample x num_class
    p = softmax(y_pred.reshape(NUM_CLASS, -1).T)
    # 前回推論したp(y|x)を取り出す
    ## weight: num_sample x num_class
    Pyx = trn_data._pweight
    # マルチホットラベルのYを取り出す
    # Y: num_sample x num_class
    Y_mh = trn_data._partial_labels
    # 各ラベルに対する重みを計算する
    # gweight: num_sample x num_class
    gweight = Pyx * Y_mh
    # ワンホットラベルのYを計算する
    # Y_oh: num_sample x num_class x num_class
    Y_oh = Y_mh[:, None] * np.eye(NUM_CLASS)
    # 各ラベル候補に対して、重み付きの勾配、二階微分を計算する
    grad = ((p[:, None] - Y_oh) * gweight[..., None]).sum(1)
    hess = ((p * (1 - p))[:, None] * gweight[..., None]).sum(1)
    grad = grad.T.reshape(-1)
    hess = hess.T.reshape(-1)
    # 次回使用するP(y|x)を計算する
    new_Pyx = softmax(y_pred.reshape(NUM_CLASS, -1).T)
    # ラベル候補以外は0として、合計値で割る
    new_Pyx = new_Pyx * Y_mh
    trn_data._pweight = new_Pyx / new_Pyx.sum(-1, keepdims=True)
    return grad, hess


def mil_loss_objective(
        y_pred:np.ndarray,
        trn_data:lgb.Dataset,
        bag_size:int,
):
    # sigmoidの計算
    p = 1 / (1 + np.exp(-y_pred))
    ## s=0の条件下での重み。0列目がy=0の重み、1列目がy=1の重み
    ## weight0: num_sample x 2
    ## s=1の条件下での重み。0列目がy=0の重み、1列目がy=1の重み
    ## weight1: num_sample x 2
    weight_s0 = trn_data._weight_s0
    weight_s1 = trn_data._weight_s1
    # マルチインスタンス学習のラベルs (バッチ数)を取り出す
    s = trn_data._mil_labels

    # 各ラベルに対する重みを計算
    # ラベルyが0である場合の重み
    gweight_s0 = weight_s0[..., 0] * (s == 0) + weight_s1[..., 0] * (s == 1)
    # ラベルy=1である場合の重み
    gweight_s1 = weight_s0[..., 1] * (s == 0) + weight_s1[..., 1] * (s == 1)

    # 各ラベル候補に対して、重み付きの勾配、二階微分を計算する
    grad = p * gweight_s0 - (1 - p) * gweight_s1
    hess = p * (1 - p) * gweight_s0 + p * (1 - p) * gweight_s1

    # 次回使用する重みを計算する
    # 各インスタンスがy=0,1である確率をlogで計算
    ## logp0: num_instance x bag_size
    ## logp1: num_instance x bag_size
    logp0 = np.log((1 - p.reshape(-1, bag_size)) + 1e-12)
    logp1 = np.log(p.reshape(-1, bag_size) + 1e-12)
    # 自身を除いたバッグ内のlog確率の和
    weights_other_logp0 = np.tile(logp0[:, None], [1, bag_size, 1]).sum(-1) - logp0
    # s, yで条件づけた重みの計算
    weights_log_s0 = logp0.sum(-1)[:, None]
    weights_log_s1 = np.log((1 - np.exp(logp0.sum(-1)) + 1e-12))[:, None]
    weights_log_s0y0 = (logp0 + weights_other_logp0 - weights_log_s0).reshape(-1)
    weights_log_s1y0 = (
        logp0 + np.log(1 - np.exp(weights_other_logp0) + 1e-12) - weights_log_s1
    ).reshape(-1)
    weights_log_s1y1 = (logp1 - weights_log_s1).reshape(-1)
    # 重みの結合、保存
    weights_s0 = np.stack(
        [np.exp(weights_log_s0y0), np.zeros_like(weights_log_s0y0)], axis=-1
    )
    weights_s1 = np.stack([np.exp(weights_log_s1y0), np.exp(weights_log_s1y1)], axis=-1)
    trn_data._weight_s0 = weights_s0
    trn_data._weight_s1 = weights_s1
    return grad, hess


def logloss(p:np.ndarray):
    return -np.log(p)


def multiclass_loss(x:np.ndarray, t:np.ndarray):
    # x: (N, C)
    # t: N
    x = x.reshape(NUM_CLASS, -1).T
    p = softmax(x)
    return -np.log(p[np.arange(len(p)), t.astype(int)] + 1e-12)


def softmax(x:np.ndarray):
    # x: (N, C)
    x = x - np.max(x, axis=-1, keepdims=True)
    out = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    return out.clip(1e-12, 1 - 1e-12)


def multiclass_metric(y_pred:np.ndarray, trn_data:lgb.Dataset):
    """Original loss function."""
    y_true = trn_data.get_label()
    return "custom_loss", multiclass_loss(y_pred, y_true).mean(), False


def binary_loss(x:np.ndarray, t:np.ndarray):
    p = 1 / (1 + np.exp(-x))
    return -(np.log(p + 1e-12) * t + np.log(1 - p + 1e-12) * (1 - t))


def binary_metric(y_pred:np.ndarray, trn_data:lgb.Dataset):
    """Original loss function."""
    y_true = trn_data.get_label()
    return "custom_loss", binary_loss(y_pred, y_true).mean(), False


def pll_metric(y_pred:np.ndarray, trn_data:lgb.Dataset):
    """Original loss function."""
    y_true = trn_data.get_label()
    return "custom_loss", pll_loss_objective(y_pred, y_true).mean(), False


def pu_loss(x: np.ndarray, t: np.ndarray, positive_ratio: float):
    """
    Args:
        x (np.array): モデルの出力配列
        t (np.array): ラベルの配列
        positive_ratio (float): 正例データの割合

    Returns:
        np.array
    """
    # sigmoid関数
    p = 1 / (1 + np.exp(-x))
    #
    loss_positive = (
        positive_ratio
        * (logloss(p) - logloss(1 - p))
        * (t == 1)
        / (t == 1).sum()  # 平均を取る
    )
    loss_unlabeled = logloss(1 - p) * (t == -1) / (t == -1).sum()  # 平均を取る
    return loss_positive + loss_unlabeled


def pu_loss_objective(y_pred:np.ndarray, trn_data:lgb.Dataset):
    """Original loss function."""
    y_true = trn_data._pu_label
    partial_fl = lambda x: pu_loss(x, y_true, positive_ratio=0.5)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess
