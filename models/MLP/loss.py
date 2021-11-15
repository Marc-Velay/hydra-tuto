from utils.abstracts.loss import Loss

class CategoricalLoss(Loss):
    def __init__(self) -> None:
        super().__init__()