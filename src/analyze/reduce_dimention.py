import numpy as np
from sklearn.manifold import TSNE as sk_TSNE
from cuml.manifold import TSNE as cuml_TSNE


def run_tsne(
    data: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 20.0,
    n_iter: int = 1000,
    backend: str = 'cuml'
) -> np.ndarray:
    if backend == 'cuml':
        TSNE = cuml_TSNE
    else:
        TSNE = sk_TSNE

    tnse = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        verbose=1
    )
    embedded_data = tnse.fit_transform(data)

    return embedded_data
