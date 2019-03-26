See https://github.com/peikexin9/deepxplore

# Execution Method from Deep Saucer

* Set `channels_last` to `image_data_format` of `keras.json`

Assuming the following directory structure, the execution method from DeepSaucer will be described.

```text
Any Directory
|-- deep_saucer_core
|   `-- mnist
|       |-- data
|       |   `-- dataset_test_images.py
|       `-- model
|           `-- model_deepxplore.py
`-- deep_xplore
    |-- deep_xplore_setup.sh
    `-- MNIST
        |-- args.json
        `-- deepxplore_verification.py
```

1. Start `DeepSaucer`
1. Select `File` - `Env Setup Script`
    1. Select [deep_xplore_setup.sh](deep_xplore_setup.sh)
1. Select `File` - `Dataset Load Script`
   1. Select `deep_saucer_core/mnist/data/dataset_test_images.py`
   1. Select `Env Setup Script` selected above
1. Select `File` - `Model Load Script`
   1. Select `deep_saucer_core/mnist/model/model_deepxplore.py`
   1. Select `Env Setup Script` selected above
1. Select `File` - `Verification Script`
   1. Select [MNIST/deepxplore_verification.py](MNIST/deepxplore_verification.py)
   1. Select `Env Setup Script` selected above
1. Select the 3 scripts of `Dataset Load`, `Model Load`, `Verification` selected above on DeepSaucer
   1. Select `Run` - `Run Test Function`
   1. Select `Next`
   1. Press `Select`, and select [MNIST/args.json](MNIST/args.json)
   1. Verification starts with `Run`