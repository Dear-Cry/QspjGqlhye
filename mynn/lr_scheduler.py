from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones=[30, 60, 90], gamma=0.1) -> None:
        """
        Args:
            optimizer: 优化器对象
            milestones: 学习率下降的步数列表
            gamma: 每次下降的乘数因子
        """
        super().__init__(optimizer)
        self.milestones = sorted(milestones)  
        self.gamma = gamma
        self.current_milestone = 0 

    def step(self) -> None:
        self.step_count += 1
        if (self.current_milestone < len(self.milestones) and 
            self.step_count >= self.milestones[self.current_milestone]):
            self.optimizer.init_lr *= self.gamma
            self.current_milestone += 1

class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.95) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.optimizer.lr *= self.gamma