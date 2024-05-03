
from datasets.dataset import Dataset
from models.naive_models import AverageYieldModel
from models.sklearn_model import SklearnModel
from models.nn_models import ExampleLSTM
from evaluation.eval import evaluate_predictions
from collections import defaultdict

from config import KEY_LOC, KEY_YEAR, KEY_TARGET
from config import SOIL_COLS, REMOTE_SENSING_COLS, METEO_COLS


def run_benchmark(model_name, model_constructor, model_kwargs):
    benchmark_models = {
        "AverageYieldModel" : AverageYieldModel,
        "LSTM" : ExampleLSTM,
        model_name : model_constructor
    }
    models_kwargs = defaultdict(dict)
    models_kwargs[model_name] = model_kwargs

    dataset = Dataset.load("maize_us")
    all_years = dataset.years
    for test_year in all_years:
        train_years = [y for y in all_years if y != test_year]
        test_years = [test_year]
        train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))

        compiled_results = {}
        for model_name, model_constructor in benchmark_models.items():
            model = model_constructor(**models_kwargs[model_name])
            model.fit(train_dataset)
            predictions, _ = model.predict(test_dataset)
            # save predictions
            labels = test_dataset.labels
            results = evaluate_predictions(labels, predictions)
            compiled_results[model_name] = results

    return compiled_results

def evaluate_saved_predictions():
    pass
