from collections import defaultdict

from models.model import BaseModel

from datasets.dataset import Dataset

from evaluation.eval import evaluate_model, evaluate_predictions

from models.naive_models import AverageYieldModel
from models.sklearn_model import SklearnModel
from models.nn_models import ExampleLSTM


def run_benchmark(model_name, model_constructor, model_kwargs):
    benchmark_models = {
        "AverageYieldModel": AverageYieldModel,
        "LSTM": ExampleLSTM,
        model_name: model_constructor
    }
    models_kwargs = defaultdict(dict)
    models_kwargs[model_name] = model_kwargs
    models_kwargs['LSTM'] = {
        'n_ts_features': 9,
        'n_static_features': 1,
        'hidden_size': 32,
        'num_layers': 3,
    }

    dataset = Dataset.load("test_maize_us")

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
            labels = test_dataset.targets()
            results = evaluate_predictions(labels, predictions)
            compiled_results[model_name] = results

    return compiled_results




if __name__ == '__main__':

    run_benchmark(model_name="AverageYieldModel",
                  model_constructor=AverageYieldModel,
                  model_kwargs={},
                  )


#
# def evaluate_saved_predictions():
#     pass


# def run_benchmark(model: BaseModel,
#                   dataset_name: str,
#                   model_fit_kwargs: dict,
#                   verbose: bool = True,
#                   ) -> dict:





    # For all algorithms in the benchmark
    #    For all years that span the benchmark dataset
    #        do train test split
    #        initialize a model
    #        fit on the training dataset
    #        predict on test dataset
    #        save predictions to csv

    # For all algorithms in the benchmark
    #    For all years that span the benchmark dataset
    #         load csv of predictions
    #         load csv of reference predictions in same fold
    #         compute metrics



#
#     # dataset_train, dataset_test = Dataset.load(dataset_name)
#     dataset = Dataset.load(dataset_name)
#     dataset_train, dataset_test = Dataset.split(...)
#
#     model.fit(dataset_train, **model_fit_kwargs)
#
#     eval_result_train = evaluate_model(model,
#                                        dataset_train,
#                                        )
#     eval_result_test = evaluate_model(model,
#                                       dataset_test,
#                                       )
#
#     return {
#         'model': model,
#         'eval_train': eval_result_train,
#         'eval_test': eval_result_test,
#     }
#
#
# def _create_table(eval_results: dict, title: str) -> list:
#
#     pass



# from datasets.dataset import Dataset
# from models.naive_models import AverageYieldModel
# from models.sklearn_model import SklearnModel
# from models.nn_models import ExampleLSTM
# from evaluation.eval import evaluate_predictions
# from collections import defaultdict
#
# from config import KEY_LOC, KEY_YEAR, KEY_TARGET
# from config import SOIL_COLS, REMOTE_SENSING_COLS, METEO_COLS
#
#
# def run_benchmark(model_name, model_constructor, model_kwargs):
#     benchmark_models = {
#         "AverageYieldModel" : AverageYieldModel,
#         "LSTM" : ExampleLSTM,
#         model_name : model_constructor
#     }
#     models_kwargs = defaultdict(dict)
#     models_kwargs[model_name] = model_kwargs
#
#     dataset = Dataset.load("maize_us")
#     all_years = dataset.years
#     for test_year in all_years:
#         train_years = [y for y in all_years if y != test_year]
#         test_years = [test_year]
#         train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))
#
#         compiled_results = {}
#         for model_name, model_constructor in benchmark_models.items():
#             model = model_constructor(**models_kwargs[model_name])
#             model.fit(train_dataset)
#             predictions, _ = model.predict(test_dataset)
#             # save predictions
#             labels = test_dataset.labels
#             results = evaluate_predictions(labels, predictions)
#             compiled_results[model_name] = results
#
#     return compiled_results
#
# def evaluate_saved_predictions():
#     pass


