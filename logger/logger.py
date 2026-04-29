

class Logger:

    def __init__(self, model_name: str, optimizer_name: str, loss_name: str, lr: float, batch_size: int, epochs: int) -> None:
        
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.train_loss_arr = []
        self.val_loss_arr = []
    
    def log(self, train_loss: float, val_loss: float) -> None :

        self.train_loss_arr.append(train_loss)
        self.val_loss_arr.append(val_loss)

    def save(self)-> None:
        pass