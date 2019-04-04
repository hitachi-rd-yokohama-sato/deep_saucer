# DNN Encoding
Verifying whether the input value and the output value satisfy a specific property, by converting DNN into a logical expression and
solving it with the SMT solver. This tool supports only Multilayer Perceptrons

## Requirement
* python 3.6
* See [lib/dnn_encoding_setup.sh](lib/dnn_encoding_setup.sh)

## Tutorial
* There are 2 ways to use this function: importing as a module and invoking from DeepSaucer.

* When used as a module: [1. Common operation](#1-common-operation) → [2. Direct execution](#2-direct-execution) → [4. Output](#4-output)
* When invoked from DeepSaucer: [1. Common operation](#1-common-operation) → [3. Invoking from DeepSaucer](#3-invoking-from-deepsaucer) → [4. Output](#4-output)

### 1. Common Operation
---
1. **Create the following files**
   * Structure Information File (json format)<br>ex) [examples/xor/model/train_name.json](examples/xor/model/train_name.json)

   * Variable Name List<br>ex) [examples/xor/model/vars_list.txt](examples/xor/model/vars_list.txt)

   **Automatic Generation Method**
     1. Prepare code that uses the model definition from existing Python code ([train_and_create_struct.py](examples/xor/train_and_create_struct.py))
     1. Include the instance creation of the NetworkStruct class of structutil.py in the code (line: [31](examples/xor/train_and_create_struct.py#L31))
     1. Add `set_input` process taking the `placeholder used as input` on the model definition code (line: [41](examples/xor/train_and_create_struct.py#L41))
     1. Add `set_hidden` processing taking a `layer used as a hidden layer`, `weight of hidden layer`, and `hidden layer bias` on the model definition (line: [49](examples/xor/train_and_create_struct.py#L49))
     1. Add `set_output` processing taking `node to be used as output`, `output node weight`, and `output node bias` on the model definition code (line: [56](examples/xor/train_and_create_struct.py#L56))
     1. After model training processing, add `set_info_by_session` processing taking `Session` (line: [76](examples/xor/train_and_create_struct.py#L76))
     1. Add save processing taking `Session` and `model output destination path` (line: [78](examples/xor/train_and_create_struct.py#L78))
     1. Add `print_vars` process taking `output destination file stream (line: [84](examples/xor/train_and_create_struct.py#L84))
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
          "hidden_node": [
            {
              "name": "Hidden_Node_Name_XX",
              "shape": "(X, Y)",
              "weight": "Weight_Variable_Name_XX",
              "bias": "Bias_Variable_Name_XX"
            },
            {
              "name": "Hidden_Node_Name_YY",
              "shape": "(X, Y)",
              "weight": "Weight_Variable_Name_YY",
              "bias": "Bias_Variable_Name_YY"
            }, ...
          ],
          "out_node": {
            "name": "Out_Node_Name_ZZ",
            "shape": "(X, Y)",
            "weight": "Weight_Variable_Name_ZZ",
            "bias": "Bias_Variable_Name_ZZ"
          }
        }
        ```
        **json Structure**
        * **input_placeholder**

            | Name | Value | Description |
            | ---- | ---- | ---- |
            | input_placeholder | List | Dictionary using the following “name”, “shape”, “description” as the key  |
            | name | String | Input placeholder name  |
            | shape | String | Input placeholder shape |
            | description | String | Input placeholder description (optional)<br>When describing in the json list format, the explanation is displayed for each shape at the time of the variable name listing<br>Ex：<br>"name": "VAR1",<br>"shape": "(1, 3)",<br>"description": "[\\"value0\\", \\"value1\\", \\"value2\\"]"<br><br>Variable Name List<br>Input:<br>VAR1_0 : value0<br>VAR1_1 : value1<br>VAR1_2 : value2<br> |

        * **hidden_node**

            | Name | Value | Description |
            | ---- | ---- | ---- |
            | output_node | List | Use the dictionary with the following “name”, “shape”, “weight”, “bias”, “description” as the key  |
            | name | String | Hidden Layer Name |
            | shape | String | Hidden Layer Shape |
            | weight | String | Variable Name Where Hidden Layer Weight is Stored |
            | bias | String | Variable Name Where Hidden Layer Bias is Stored |
            | description | String | Hidden Layer Description (optional)<br>Same as the input_placeholder description |

         * **output_node**

            | Name | Value | Description |
            | ---- | ---- | ---- |
            | output_node | dictionary | Dictionary using the following “name”, “shape”, “weight”, “bias”, “description” as the key |
            | name | String | Output Node Name |
            | shape | String | Output Node Shape |
            | weight | String | Variable Name where the Weight String Output Node is Stored |
            | bias | String | Variable Name where the Bias String Output Node is Stored |
            | description | String |  Output Node Description (optional)<br>Same as the input_placeholder description |

     1. Create a variable name list by executing the following code in which the path of the created json file is used
        ```python
        network_struct = NetworkStruct()
        network_struct.load_struct('/Any/Struct/json/path')
        network_struct.print_vars()
        with open('vars_list.txt', 'w') as ws:
            ns.print_vars(ws=ws)
        ```

1. **Refer to the above variable name list to create a property file**

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

   ex ) [examples/xor/express/express_sat.txt](examples/xor/express/express_sat.txt)

1. **Create prerequisite files, if necessary**

   The operators available for the prerequisite file are the same as the properties file

   ex ) [examples/xor/condition/condition.txt](examples/xor/condition/condition.txt)

1. **Create a config file (json format) describing the `structure information file path`, `property file path`, `prerequisite file path`**

   | Setting Item | Key | Type | Note |
   | ---- | ---- | ----| ---- |
   | Structure Information File Path | `NameList` | String |  |
   | Property File Path | `Prop` | String |  |
   | Prerequisite File Path | `Condition` | String | Optional |

   ex ) [examples/xor/configs/config_xor.json](examples/xor/configs/config_xor.json)

### 2. Direct Executio
---
For execution as a module, execute by giving 2 parameters to the main function as follows
```python
main(sess, config_path)
```
| Parameter | Type | Description |
| --- | --- | --- |
| sess | Session | Graph Session Created by Tensorflow |
| conf_path | String| Config File Path used for Test |
```python
import tensorflow as tf
from lib.dnn_encoding_verification import main

if __name__ == '__main__':
    sess = tf.Session()

    # create or restore tensorflow graph session

    conf_path = 'XXX.json' # Any config file path

    main(sess, config_path=conf_path)
```

Practical Example using XOR： [examples/xor/run_xor.py](examples/xor/run_xor.py)

### 3. Invoke from DeepSaucer
---
* Tutorial Using XOR

Execute from DeepSaucer assuming the following directory structureう
```text
Any Directory
|-- deep_saucer_core
|   |-- downloaded_data
|   |   `-- xor_tensorflow (empty also OK)
|   `-- xor
|       |-- configs
|       |   `-- config_dnn_encoding.json
|       `-- model
|           `-- model_assertion_check.py
`-- dnn_encoding
    `-- lib
        |-- dnn_encoding_setup.sh
        `-- dnn_encoding_verification.py
```

1. Start `DeepSaucer`
1. Select `File` - `Add Env Setup Script`
    1. Select [lib/dnn_encoding_setup.sh](lib/dnn_encoding_setup.sh)
1. Select `File` - `Add Model Load Script`
   1. Select `deep_saucer_core/xor/model/model_dnn_encoding.py`
   1. Select `Env Setup Script` selected above
1. Select `File` - `Add Verification Script`
   1. Select [lib/dnn_encoding_verification.py](lib/dnn_encoding_verification.py)
   1. Select `Env Setup Script` selected above
1. Select the 3 scripts of `Model Load`, `Verification` selected above on DeepSaucer
   1. Select Run` - `Run Test Function`
   1. Select `Next`
   1. Press `Select` and select `deep_saucer_core/xor/configs/config_dnn_encoding.json`
   1. Verification starts with `Run`

### 4. Output
---
* Output the verification intermediate file (`~.smt`) to the smt directory
* Output counterexample to sat/satisfiable.txt
* The verification result and counterexample are displayed to the standard output in the following format
    ```text
    --------------------
    Property is [preserved/violated]
    Counterexample is
    input_0:0,input_1:1,output_0:0
    --------------------
    ```
    * preserved: Meets the properties
    * violated: Violates the properties
