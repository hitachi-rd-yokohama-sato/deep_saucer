See https://github.com/matthewwicker/SafeCV

# Execution Method from Deep Saucer

Assuming the following directory structure, the execution method from DeepSaucer will be described.

```text
Any Directory
|-- deep_saucer_core
|   `-- mnist
|       |-- data
|       |   `-- dataset_test.py
|       `-- model
|           `-- model_safecv.py
`-- safecv
    |-- Examples
    |   `-- MNIST-Example
    |       |-- config.json
    |       `-- safecv_verification.py
    `-- safecv_setup.sh
```

1. Start `DeepSaucer`
1. Select `File` - `Env Setup Script`
    1. Select [safecv_setup.sh](safecv_setup.sh)
1. Select `File` - `Dataset Load Script`
   1. Select `deep_saucer_core/mnist/data/dataset_test.py`
   1. Select `Env Setup Script` from the selection above
1. Select `File` - `Model Load Script`
   1. Select `deep_saucer_core/mnist/model/model_safecv.py`
   1. Select `Env Setup Script` from the selection above
1. Select `File` - `Verification Script`
   1. Select [Examples/MNIST-Example/safecv_verification.py](Examples/MNIST-Example/safecv_verification.py)
   1. Select `Env Setup Script` selected above
1. Select the 3 scripts (`Dataset Load`, `Model Load`, and `Verification`) selected above on DeepSaucer
   1. Select `Run` - `Run Test Function`
   1. Select `Next`
   1. Press `Select`, and select [Examples/MNIST-Example/config.json](Examples/MNIST-Example/config.json)
   1. Verification starts with `Run`