class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, verbose=True, monitor='val_acc'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.monitor = monitor  # 'val_acc' or 'val_loss'
        self.counter = 0
        self.early_stop = False
        self.best_state = None

        # if monitor == 'val_loss':
        #     self.best = float('inf')
        # else:
        #     self.best = float('-inf')
        
        if monitor == 'val_acc':
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def __call__(self, val_loss, val_acc, model):
        # if self.monitor == 'val_loss':
        #     current = val_loss
        #     improved = current < self.best - self.min_delta
        #     metric_name = 'val_loss'
        #     metric_value = f"{current:.4f}"
        # else:
        #     current = val_acc
        #     improved = current > self.best + self.min_delta
        #     metric_name = 'val_acc'
        #     metric_value = f"{current:.4f}"

        if self.monitor == 'val_acc':
            current = val_acc
            improved = current > self.best + self.min_delta
            metric_name = 'val_acc'
            metric_value = f"{current:.4f}"
        else:
            current = val_loss
            improved = current < self.best - self.min_delta
            metric_name = 'val_loss'
            metric_value = f"{current:.4f}"

        if improved:
            self.best = current
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"  ✅ EarlyStopping: {metric_name} 改善至 {metric_value}")
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