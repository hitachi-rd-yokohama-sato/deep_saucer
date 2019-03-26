# Assertion Testing
A function that executes DNN with the data set (input value only, no expected output value) given by the user, and checks whether the value output by DNN and input value given by the user satisfy a specific property.

## Requirement
* python 3.6
* See [lib/assertion_testing_setup.sh](lib/assertion_testing_setup.sh)

## Tutorial
* There are 2 ways to use this function: importing as a module and invoking from DeepSaucer.

* When used as a module: [1. Common operation](#1-common-operation) → [2. Direct execution](#2-direct-execution) → [4. Output](#4-output)
* When invoked from DeepSaucer: [1. Common operation](#1-common-operation) → [3. Invoking from DeepSaucer](#3-invoking-from-deepsaucer) → [4. Output](#4-output)


### 1. Common Operation
---
1. **Create the following files:**
   * Structure Information File(json format)<br>
     ex) [examples/xor/model/train_name.json](examples/xor/model/train_name.json)<br>

   * Variable Name List<br>
     ex) [examples/xor/model/vars_list.txt](examples/xor/model/vars_list.txt)<br>

   **Automatic Generation Method**
     1. Prepare code that uses the model definition from existing Python code ([train_and_create_struct.py](examples/xor/train_and_create_struct.py))
     1. Include the instance creation of the `NetworkStruct` class of `structutil.py` in the code. (line: [30](examples/xor/train_and_create_struct.py#L30))
     1. Add `set_input` process taking the `placeholder used as input` on the model definition code (line: [40](examples/xor/train_and_create_struct.py#L40))
     1. Add `set_output` process taking the node `used as output` on the model definition code (line: [55](examples/xor/train_and_create_struct.py#L55))
     1. After the model training process, add `set_info_by_session` process taking `Session` [75](examples/xor/train_and_create_struct.py#L75))
     1. Add `save` process taking `Session` and `model output destination path` (line: [77](examples/xor/train_and_create_struct.py#L77))
     1. Add `print_vars` process taking the `destination file stream` (line: [84](examples/xor/train_and_create_struct.py#L84))
     1. Execute the above code to generate the `structure file` and `variable name list`

   **Manual Generation**
     1. Create a json file with the following content
        ```json
        {
          "input_placeholder": [
            {
              "name": "Placeholder_Name_XX",
              "shape": "(X, Y)"
            },
            {
              "name": "Placeholder_Name_YY",
              "shape": "(X, Y)"
            }, ...
          ],
          "out_node": {
            "name": "Out_Node_Name_XX",
            "shape": "(X, Y)"
          }
        }
        ```

        **json Structure**
        * **input_placeholder**

            | Name | Value | Description |
            | ---- | ---- | ---- |
            | input_placeholder | List | Dictionary List using the following "name", "shape", "description" as the key |
            | name | String | Input placeholder Name |
            | shape | String | Input placeholder shape |
            | description | String | Input placeholder Description (optional)<br>When describing in the json list format, the explanation is displayed for each shape at the time of the variable name listing<br>Example: <br>"name": "VAR1",<br>"shape": "(1, 3)",<br>"description": "[\\"value0\\", \\"value1\\", \\"value2\\"]"<br><br>Variable Name List<br>Input:<br>VAR1_0 : value0<br>VAR1_1 : value1<br>VAR1_2 : value2<br> |

        * **output_node**

            | Name | Value | Description |
            | ---- | ---- | ---- |
            | output_node | Dictionary | Dictionary using the following "name", "shape", "description" as the key |
            | name | String | Output node name |
            | shape | String | Output node shape |
            | description | String | Output node description (optional)<br>Same specification as the input_placeholder’s description |

     1. Create a variable name list by executing the following code in which the path of the created json file is referred
        ```python
        network_struct = NetworkStruct()
        network_struct.load_struct('/Any/Struct/json/path')
        network_struct.print_vars()
        with open('vars_list.txt', 'w') as ws:
            ns.print_vars(ws=ws)

        ```

1. **Create a property file by referring to the variable name list above**

   Operators available for property files are as follows

   | Operator | Usage Example | Meaning
   | ---- | ---- | ---- |
   | + | VAR_0 + 10 | addition |
   | - | VAR_0 - VAR_1 | subtraction |
   | * | VAR_0 * 10 | multiplication |
   | / | VAR_0 / VAR_1 | division |
   | % | VAR_0 % 10 | modulo |
   | ** | VAR_0 ** 10 | exponentiation |
   | == | VAR_0 == VAR_1 | equal |
   | != | VAR_0 != VAR_1 | not equal |
   | > | VAR_0 > 10 | greater |
   | < | VAR_0 < 10 | lesser |
   | >= | VAR_0 >= 10 | greater equal |
   | <= | VAR_0 <= 10 | lesser equal |
   | && | VAR_0 > 10 && VAR_1 == 0 | conjunction |
   | \|\| | (VAR_0 > 10) \|\| (VAR_1 == 0) | disjunction |
   | => | VAR_0 => VAR_1  | implication |
   | ! | !(VAR_0 > 10) | negation |


   ex ) [examples/xor/express/express.txt](examples/xor/express/express.txt)

1. **Create the config file (json format) that describes the `structure information file path`, `property file path`, and `test data number`**

   | Setting Items | Key | Type | Note |
   | ---- | ---- | ----| ---- |
   | Structure Information File Path | `NameList` | String |  |
   | Property File Path | `Prop` | String |  |
   | Test Data Number Used | `NumTest` | Numerical Value | Optional<br>If omitted, all data used|

   ex ) [examples/xor/configs/config_mnist.json](examples/xor/configs/config_xor.json)

### 2. Direct Execution
---
For direct execution, execute by giving 3 parameters to the main function as follows
```python
main(sess, dataset, config_path)
```
| Parameters | Type | Description |
| --- | --- | --- |
| sess | Session | Graph Session Created by Tensorflow |
| dataset | List | Data Set Used for Testing |
| conf_path | String | Config File Path used for Test |

```python
import numpy as np
import tensorflow as tf

from lib.assertion_check_verification import main

if __name__ == '__main__':
    # Any dataset
    X = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    sess = tf.Session()
    # create or restore tensorflow graph session

    config_path = 'XXX.json' # Any config file path

    # dataset: List of values given each input placeholder
    main(sess, [X], config_path)
```

Practical Example using XOR: [examples/xor/run_xor.py](examples/xor/run_xor.py)

### 3. Invoking from DeepSaucer
---
* Tutorial Using XOR

Execute from DeepSaucer assuming the following directory structure
```text
Any Directory
|-- deep_saucer_core
|   |-- downloaded_data
|   |   `-- xor_tensorflow (empty also OK)
|   `-- xor
|       |-- configs
|       |   `-- config_assertion_check.json
|       |-- data
|       |   `-- dataset_assertion.py
|       `-- model
|           `-- model_assertion_check.py
`-- assertion_testing
    `-- lib
        |-- assertion_testing_setup.sh
        `-- assertion_check_verification.py
```
1. Start `DeepSaucer`
1. Select `File` - `Env Setup Script`
    1. Select [lib/assertion_testing_setup.sh](lib/assertion_testing_setup.sh)
1. Select `File` - `Dataset Load Script`
   1. Select `deep_saucer_core/xor/data/dataset_assertion.py`
   1. Select `Env Setup Script` selected above
1. Select `File` - `Model Load Script`
   1. Select `deep_saucer_core/xor/model/model_assertion.py`
   1. Select `Env Setup Script` selected above
1. Select `File` - `Verification Script`
   1. Select [lib/assertion_check_verification.py](lib/assertion_check_verification.py)
   1. Select `Env Setup Script` selected above
1. Select the 3 scripts selected above on `Dataset Load`, `Model Load`, and `Verification`
   1. Select `Run` - `Run Test Function`
   1. Select `Next`
   1. Press Select `Select` to select `deep_saucer_core/xor/configs/config_assertion_check.json`
   1. Verification starts with `Run`

### 4. Output
---
For each test data the results from verifying the properties are displayed to the standard output and the log file as follows

```text
0 : preserved
1 : preserved
2 : violated
3 : preserved
```

* preserved: Meets the properties
* violated: Violates the properties
