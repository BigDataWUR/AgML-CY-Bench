# Contributing to AgML Crop Yield Forecasting

At the moment, there are two ways to contribute to the AgML Subnational Crop Yield Forecasting benchmark.
1. [Contribute to data documentation and preparation](data_preparation/CONTRIBUTING.md).
2. Contribute to code and documentation of the python package.
3. Contribute models.

## Join the AgML team
If you have not yet joined the AgML team, please send an email to agml+join@mail.agml.org.

## Request collaborator access
Add your GitHub username to this [document](https://docs.google.com/document/d/1Hhk2BEHmvHxg8ghc4pVRcGNvvIoX8XKN3Mj5hsSmC4A/edit?usp=sharing) to get collaborator access. Another option is to fork the project or repository; this does not require collaborator access.

## Create an issue
For any changes or enhancements, check if a related issue already exists. If not, open a new issue. If you do not have collaborator access, what you are fixing can be described in the pull request (see below).

## Make changes
Clone the repository or fork the project. Create a working branch from `main` to make changes. The branch can be named something like `<username>/<short-summary-of-issue>`.

Make necessary changes and test them. Commit your changes (add the issue number in the commit message). Before pushing your changes, run [`black`](https://github.com/psf/black) to format your code.

## Pull request
When you have finished making changes, create a pull request and request review from at least one reviewer. After the reviewers approve your changes, your branch can be merged with `main`.

## Contributing models
To contribute a model, first write a model class `your_model` that extends the `BaseModel` class. The base model class definition is
inside `models.model`. Then run the model along with the baseline models included in the benchmark.

```
from cybench.models.model import BaseModel
from cybench.runs.run_benchmark import run_benchmark

class MyModel(BaseModel): 
    pass


run_name = <run_name>
dataset_name = "maize_US"
run_benchmark(run_name=run_name, 
              model_name="my_model",
              model_constructor=MyModel,
              model_init_kwargs: <int args>,
              model_fit_kwargs: <fit params>,
              dataset_name=dataset_name)

```

After your model runs and produces results, please send an email to AgML list email (agml@mail.agml.org) to include the new model in the benchmark. Include "CY-Bench" or "Subnational crop yield forecasting" in the subject.
