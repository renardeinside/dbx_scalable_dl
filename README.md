# dbx_scalable_dl

## Local environment setup

To set up the local environment, do the following:
- create the new conda environment:
```
conda env create --file environment.yml 
```
- install Horovod separately via pip (it's important so the Horovod<>Tensorflow binding need specific env variable):
```
HOROVOD_WITH_TENSORFLOW=1 pip install "horovod[tensorflow,spark]==0.22.1"
```


## Resources

- [Amazon datasets](https://nijianmo.github.io/amazon/index.html)