from dataclasses import dataclass, field

@dataclass
class Config:
    # models to train
    models = ["RF", "XGB", "ANN"]
    # dataset
    target_column = "median_house_value"
    test_size = 0.2
    split_seed = 42
    experiment_name = 'housing'

    # save directories
    save_dir = "C:/Explainable AI/results"
    plot_dir = "C:\Explainable AI\plots"
    model_dir = "C:\Explainable AI\models"


    # ANN
    ann_epochs = 50
    ann_batch_size = 32
    ann_learning_rate = 0.001