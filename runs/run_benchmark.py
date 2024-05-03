
from models.model import BaseModel

from datasets.dataset import Dataset

from evaluation.eval import evaluate_model





def run_benchmark(model: BaseModel,
                  dataset_name: str,
                  model_fit_kwargs: dict,
                  verbose: bool = True,
                  ) -> dict:


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




    # dataset_train, dataset_test = Dataset.load(dataset_name)
    dataset = Dataset.load(dataset_name)
    dataset_train, dataset_test = Dataset.split(...)

    if isinstance(model, ...):  # TODO -- check if model is a torch model
        model.train()

    model.fit(dataset_train, **model_fit_kwargs)

    if isinstance(model, ...):  # TODO -- check if model is a torch model
        model.eval()

    eval_result_train = evaluate_model(model,
                                       dataset_train,
                                       )
    eval_result_test = evaluate_model(model,
                                      dataset_test,
                                      )

    return {
        'model': model,
        'eval_train': eval_result_train,
        'eval_test': eval_result_test,
    }


def _create_table(eval_results: dict, title: str) -> list:
    
    pass
