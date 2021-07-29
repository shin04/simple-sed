class EarlyStopping:
    def __init__(
        self, patience: int = 5, delta: float = 0.
    ) -> None:
        """
        Args:
            patience: How long to wait after last time validation loss improved.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
        """

        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
