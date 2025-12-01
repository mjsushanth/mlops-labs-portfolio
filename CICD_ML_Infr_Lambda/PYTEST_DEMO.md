

### PYTEST_DEMO.md - Shows sample output of running pytest with coverage for the CICD_ML_Infr_Lambda project

#### Command to run tests with coverage
Command used: `pytest tests/ -v --cov=src --cov-report=term-missing`


#### Sample Output

configfile: pyproject.toml
plugins: cov-5.0.0
collected 21 items

```
tests/test_inference_service.py::TestInferenceService::test_valid_setosa_prediction PASSED                                                  [  4%]
tests/test_inference_service.py::TestInferenceService::test_valid_virginica_prediction PASSED                                               [  9%]
tests/test_inference_service.py::TestInferenceService::test_probabilities_sum_to_one PASSED                                                 [ 14%]
....
....
tests/test_model_loader.py::TestModelLoader::test_singleton_returns_same_instance PASSED                                                    [ 71%] 
....
....
tests/test_model_loader.py::TestModelPaths::test_metadata_file_exists PASSED                                                                [100%] 
```

---------- coverage: platform win32, python 3.11.1-final-0 -----------
Name                       Stmts   Miss  Cover   Missing
--------------------------------------------------------
src\__init__.py                1      0   100%
src\inference_service.py      38      0   100%
src\lambda_function.py        14     14     0%   6-62
src\model_loader.py           44      2    95%   46, 67
src\path_config.py            11      1    91%   31
--------------------------------------------------------
TOTAL                        108     17    84%

