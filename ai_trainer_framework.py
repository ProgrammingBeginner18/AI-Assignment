from abc import ABC, abstractmethod


class ModelConfig:
    def __init__(self, model_name, learning_rate=0.01, epochs=10):
        # OOP Concept: Instance Attribute
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs

    # OOP Concept: Magic Method
    def __repr__(self):
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"


# OOP Concept: Abstraction (ABC)
class BaseModel(ABC):

    # OOP Concept: Class Attribute
    model_count = 0

    def __init__(self, config: ModelConfig):
        # OOP Concept: Composition (BaseModel owns a ModelConfig instance)
        self.config = config

        # increment model count
        BaseModel.model_count += 1

    # OOP Concept: Instance Method + Abstraction
    @abstractmethod
    def train(self, data):
        pass

    # OOP Concept: Instance Method + Abstraction
    @abstractmethod
    def evaluate(self, data):
        pass


# OOP Concept: Single Inheritance (LinearRegressionModel → BaseModel)
class LinearRegressionModel(BaseModel):

    def __init__(self, learning_rate=0.01, epochs=10):

        # create configuration object
        config = ModelConfig("LinearRegression", learning_rate, epochs)

        # OOP Concept: super() (child calling parent constructor)
        super().__init__(config)

    # OOP Concept: Instance Method + Method Overriding
    def train(self, data):
        print(f"\n--- Training {self.config.model_name} ---")
        print(
            f"{self.config.model_name}: Training on {len(data)} samples "
            f"for {self.config.epochs} epochs (lr={self.config.learning_rate})"
        )

    # OOP Concept: Instance Method + Method Overriding
    def evaluate(self, data):
        print(f"{self.config.model_name}: Evaluation MSE = 0.042")


# OOP Concept: Single Inheritance (NeuralNetworkModel → BaseModel)
class NeuralNetworkModel(BaseModel):

    def __init__(self, layers, learning_rate=0.001, epochs=20):

        # additional attribute specific to neural networks
        self.layers = layers

        config = ModelConfig("NeuralNetwork", learning_rate, epochs)

        # OOP Concept: super()
        super().__init__(config)

    # OOP Concept: Instance Method + Method Overriding
    def train(self, data):
        print(f"\n--- Training {self.config.model_name} ---")
        print(
            f"{self.config.model_name} {self.layers}: Training on {len(data)} samples "
            f"for {self.config.epochs} epochs (lr={self.config.learning_rate})"
        )

    # OOP Concept: Instance Method + Method Overriding
    def evaluate(self, data):
        print(f"{self.config.model_name}: Evaluation Accuracy = 91.5%")


class DataLoader:
    def __init__(self, dataset):
        # store dataset
        self.dataset = dataset

    # OOP Concept: Instance Method
    def get_data(self):
        return self.dataset


class Trainer:
    def __init__(self, model: BaseModel, loader: DataLoader):
        # OOP Concept: Aggregation (Trainer receives DataLoader externally)
        self.model = model
        self.loader = loader

    # OOP Concept: Instance Method
    def run(self):

        data = self.loader.get_data()

        # OOP Concept: Polymorphism
        # Trainer.run() works with ANY BaseModel (LR or NN)
        self.model.train(data)
        self.model.evaluate(data)


if __name__ == "__main__":

    # create models
    lr_model = LinearRegressionModel(learning_rate=0.01, epochs=10)
    nn_model = NeuralNetworkModel(layers=[64, 32, 1], learning_rate=0.001, epochs=20)

    # show configs
    print(lr_model.config)
    print(nn_model.config)

    # show class attribute usage
    print("\nModels created:", BaseModel.model_count)

    # dataset
    dataset = [1, 2, 3, 4, 5]

    # create DataLoader
    loader = DataLoader(dataset)

    # create Trainers
    trainer1 = Trainer(lr_model, loader)
    trainer2 = Trainer(nn_model, loader)

    # run pipeline
    trainer1.run()
    trainer2.run()