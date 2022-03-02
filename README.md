# dbx_scalable_dl

## Local environment setup

To set up the local environment, do the following:
- create a new conda environment:
```
conda env create --file environment.yml 
```
- install Horovod separately via pip (it's important so the Horovod<>Tensorflow binding need specific env variable):
```
HOROVOD_WITH_TENSORFLOW=1 pip install "horovod[tensorflow,spark]==0.22.1"
```


## Resources

- [Horovod installation guide](https://horovod.readthedocs.io/en/stable/install_include.html)
- [MLflow custom Python Models](https://mlflow.org/docs/1.6.0/python_api/mlflow.pyfunc.html)
- [Amazon datasets](https://nijianmo.github.io/amazon/index.html)