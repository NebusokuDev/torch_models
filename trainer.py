from abc import ABC, abstractmethod


class TrainerBase(ABC):
    def __init__(self, epochs):
        self.epochs = epochs

    @abstractmethod
    def _training_step(self):
        pass

    @abstractmethod
    def _test_step(self):
        pass

    @abstractmethod
    def _validation_step(self):
        pass

    @abstractmethod
    def _visualize(self):
        pass

    def fit(self):
        for epoch in range(self.epochs):
            self._training_step()
            self._test_step()
            self._validation_step()
            self._visualize()

    def __call__(self):
        self.fit()


class Trainer(TrainerBase):
    def _training_step(self):
        pass

    def _test_step(self):
        pass

    def _validation_step(self):
        pass

    def _visualize(self):
        pass


class GANTrainer(TrainerBase):
    def __init__(self,
                 generator,
                 discriminator,
                 criterion_g,
                 criterion_d,
                 optimizer_g,
                 optimizer_d,
                 train_dataset,
                 test_dataset,
                 epochs):
        super().__init__(epochs)

    def _validation_step(self):
        pass

    def _visualize(self):
        pass

    def _training_step(self):
        pass

    def _test_step(self):
