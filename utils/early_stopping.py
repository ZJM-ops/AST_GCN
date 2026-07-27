class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"  ✅ EarlyStopping: val_loss 改善至 {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ⏳ EarlyStopping: 未改善 {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print("  🔄 已恢复最优权重")