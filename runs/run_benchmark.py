import os
from collections import defaultdict

import pandas as pd
import torch

import config
from config import PATH_RESULTS_DIR
from models.model import BaseModel

from datasets.dataset import Dataset

from evaluation.eval import evaluate_model, evaluate_predictions

from models.naive_models import AverageYieldModel
from models.sklearn_model import SklearnModel
from models.nn_models import ExampleLSTM


def run_benchmark(run_name: str, model_name: str, model_constructor, model_kwargs, model_fit_kwargs) -> dict:
    path_results = os.path.join(PATH_RESULTS_DIR, run_name)
    os.makedirs(path_results, exist_ok=True)

    benchmark_models = {
        "AverageYieldModel": AverageYieldModel,
        "LSTM": ExampleLSTM,
        model_name: model_constructor
    }
    models_kwargs = defaultdict(dict)
    models_fit_kwargs = defaultdict(dict)
    models_kwargs[model_name] = model_kwargs
    models_fit_kwargs[model_name] = model_fit_kwargs

    models_kwargs['LSTM'] = {
        'n_ts_features': 9,
        'n_static_features': 1,
        'hidden_size': 32,
        'num_layers': 1,
    }

    models_fit_kwargs['LSTM'] = {
        'batch_size': 32,
        'num_epochs': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'optim_fn': torch.optim.Adam,
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'scheduler_kwargs': {"step_size": 2, "gamma": 0.8},
        'val_fraction': 0.1,

        'optimize_hyperparameters': True,
        'param_space': {
            'optim_kwargs': {
                "lr": [0.01, 0.001],
                'weight_decay': [0.0001],
            },
        },
        'do_kfold': False,
        'kfolds': 5,
    }


    dataset = Dataset.load("test_maize_us")

    all_years = dataset.years
    for test_year in all_years:
        train_years = [y for y in all_years if y != test_year]
        test_years = [test_year]
        train_dataset, test_dataset = dataset.split_on_years((train_years, test_years))

        labels = test_dataset.targets()

        model_output = {
            config.KEY_LOC: [loc_id for loc_id, _ in test_dataset.indices()],
            config.KEY_YEAR: [year for _, year in test_dataset.indices()],
            'targets': labels,
        }

        compiled_results = {}
        for model_name, model_constructor in benchmark_models.items():
            model = model_constructor(**models_kwargs[model_name])
            model.fit(train_dataset, **models_fit_kwargs[model_name])
            predictions, _ = model.predict(test_dataset)
            # save predictions
            results = evaluate_predictions(labels, predictions)
            compiled_results[model_name] = results

            model_output[model_name] = predictions

        df = pd.DataFrame.from_dict(model_output)
        df.set_index([config.KEY_LOC, config.KEY_YEAR], inplace=True)
        df.to_csv(os.path.join(path_results, f'year_{test_year}.csv'))

    return compiled_results


if __name__ == '__main__':

    model_name = 'LSTM_2_layers'
    model_constructor = ExampleLSTM

    model_kwargs = {
        'n_ts_features': 9,
        'n_static_features': 1,
        'hidden_size': 32,
        'num_layers': 2,
    }


    model_fit_kwargs = {
        'batch_size': 32,
        'num_epochs': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'optim_fn': torch.optim.Adam,
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'scheduler_kwargs': {"step_size": 2, "gamma": 0.8},
        'val_fraction': 0.1,

        'optimize_hyperparameters': True,
        'param_space': {
            'optim_kwargs': {
                "lr": [0.01, 0.001],
                'weight_decay': [0.0001],
            },
        },
        'do_kfold': False,
        'kfolds': 5,
    }




    result = run_benchmark(
        run_name='test_lstm_2_layers',
        model_name=model_name,
        model_constructor=model_constructor,
        model_kwargs=model_kwargs,
        model_fit_kwargs=model_fit_kwargs,
        )

    print(result)



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


