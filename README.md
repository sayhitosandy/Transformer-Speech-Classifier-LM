# Transformer-Speech-Classifier-LM
#### Spring 2024, CSE 256: Statistical Natural Language Processing, UC San Diego

<br>
<br>

## Overview
### Objective
This project aims to deepen the understanding of Transformer architectures by implementing and experimenting with different components, improving model performance, and gaining insights into the nuances of speech segment classification and language modeling tasks.

### Tasks Overview
1. **Part 1: Encoder Implementation and Classification**
   - Implement a Transformer Encoder from scratch.
   - Train it jointly with a FeedForward Classifier to predict the US President who delivered a given speech segment.
   - Dataset: Speech segments labeled with politicians (Barack Obama, George W. Bush, George H. Bush).
   - Evaluate classifier accuracy.

2. **Part 2: Decoder Implementation and Language Modeling**
   - Implement a Transformer Decoder with masked self-attention.
   - Pretrain the Decoder on an autoregressive Language Modeling task to predict the next word in a sequence.
   - Dataset: Unlabeled text from speeches.
   - Report perplexity on test sets from different politicians.

3. **Part 3: Architectural Exploration**
   - Experiment with various transformer architecture components such as positional encoding, sparse attention patterns, etc.
   - Aim to improve the classifier’s accuracy or the decoder’s perplexity.

### Evaluation Metrics
- **Classification Task:**
  - Accuracy on test dataset.
  - Track accuracy across 15 epochs.

- **Language Modeling Task:**
  - Perplexity on test sets for different politicians.
  - Track perplexity every 100 iterations up to 500 iterations.

### Reporting
- Document the implementation process, results, and insights gained.
- Include plots and visualizations for attention matrices.
- Summarize performance improvements and architectural exploration findings.

<br>
<br>

## How to Run?
### Virtual Environment
It is recommended to run the code inside a virtual environment. You can create one using the following command:
```sh
python3 -m venv ./.venv
```
This will create a virtual environment called `.venv`. Activate the environment using the following command:
```sh
source ./.venv/bin/activate
```

### Dependencies
You will need Python3, PyTorch, tqdm, Pandas, matplotlib, and JSON installed to run the code. Please install any dependencies using the following command:
```sh
python3 -m pip install -r requirements.txt
```

### Run Code
Once the dependencies are installed, please use the following commands to run different parts of the assignment:
```sh
# Classification
python3 main.py part1

# Language Modeling  
python3 main.py part2

# Exploration
python3 main.py part3
```

<br>

In case you want to see the metrics at each iteration, please set `--verbose=True`. It is `False` by default. For example:
```sh
# Classification
python3 main.py part1 --verbose=True
```

<br>

In case you want to perform sanity check on the attention maps, please set `--perform-sanity-check=True`. It is `False` by default. For example:
```sh
# Classification
python3 main.py part1 --perform-sanity-check=True
```

<br>
<br>

## Implementation Details
### Part 1: Classification
Here, I implement a transformer encoder and train it jointly from scratch with a feedforward classifier for a downstream task of predicting which politician delivered a given speech segment. Following are steps involved:
1. Load text from `speechesdataset/train_CLS.tsv` and `speechesdataset/train_LM.txt` files using the `load_texts()` function.

2. Build a tokenizer using the `SimpleTokenizer` class, which outputs a vocabulary from the given text and encodes/decodes text into indices.

3. Run the `classification_task()` function.

    1. Get an iterable over the train dataset (`speechesdataset/train_CLS.tsv`) and test dataset (`speechesdataset/test_CLS.tsv`) using the `get_cls_data_loader()` function. This uses `SpeechesClassificationDataset()` and `DataLoader()` functions.

    2. Define the `classifier` object using the `transformer.Classifier` class. This consists of the following:
        - A transformer `Encoder` which consists of 2 embeddings layers, followed by 4 layers of transformer `Block`s. 
            - Each `Block` of the transformer consists of a `MultiHeadAttention` layer and a `FeedForward` layer, along with `LayerNorm` layers and residual connections. 
            - Each `MultiHeadAttention` layer contains 2 `AttentionHead`s followed by a `Linear` layer. Each `AttentionHead` performs the attention operation using the key (`k`), query (`q`), and value (`v`) vectors on the input (`x`) vector. 
            - The `FeedForward` layer consists of 2 `Linear` layers along with a `ReLU` activation function to introduce non-linearity.
        - 2 `Linear` layers along with a `ReLU` activation function.

    3. Define the `criterion` and the `optimizer`, and train and evaluate the `classifier` for `epochs_cls` epochs.

    4. Save and output the `train_loss`, `train_accuracy`, and `test_accuracy` for each epoch.

    5. Perform a sanity check on the attention maps using the `Utilities` class.

4. Write the output to a JSON file using the `write_output_to_json()` function.

<br>
<br>

### Part 2: Language Modeling
Here, I implement a word-level, GPT-like transformer decoder, pretrain it on an autoregressive language modeling task, and report perplexity numbers on speeches from different politicians. Following are steps involved:
1. Load text from `speechesdataset/train_CLS.tsv` and `speechesdataset/train_LM.txt` files using the `load_texts()` function.

2. Build a tokenizer using the `SimpleTokenizer` class, which outputs a vocabulary from the given text and encodes/decodes text into indices.

3. Run the `language_modeling_task()` function.

    1. Get an iterable over the train dataset (`speechesdataset/train_LM.txt`) and test datasets (`speechesdataset/test_LM_hbush.txt`, `speechesdataset/test_LM_obama.txt`, and `speechesdataset/test_LM_wbush.txt`) using the `get_lm_data_loader()` function. This uses `LanguageModelingDataset()` and `DataLoader()` functions.

    2. Define the `decoder` object using the `transformer.Decoder` class. This consists of the following:
        - A transformer `Decoder` which consists of 2 embeddings layers, 4 layers of transformer `Block`s, a final `LayerNorm` layer, and a `Linear` layer.
            - Each `Block` of the transformer consists of a `MultiHeadAttention` layer and a `FeedForward` layer, along with `LayerNorm` layers and residual connections. 
            - Each `MultiHeadAttention` layer contains 2 `AttentionHead`s followed by a `Linear` layer. Each `AttentionHead` performs the attention operation using the key (`k`), query (`q`), and value (`v`) vectors on the input (`x`) vector. 
            - The `FeedForward` layer consists of 2 `Linear` layers along with a `ReLU` activation function to introduce non-linearity.

    3. Define the `optimizer` and train the `decoder` for `max_iters` epochs, evaluating after every `eval_interval` interval.

    4. Save and output the `train_perplexity`, `hbush_test_perplexity`, `obama_test_perplexity` and `wbush_test_perplexity` at each interval.

    5. Perform a sanity check on the attention maps using the `Utilities` class.

4. Write the output to a JSON file using the `write_output_to_json()` function.

<br>
<br>

### Part 3: Exploration
#### Part 3.1: Architectural Exploration
This involves running the `classification_task()` and `language_modeling_task()` functions by setting the `use_alibi` argument to be `True`. The steps are the same as Parts 1 and 2. The only difference is that the transformer now uses ALiBi positional embeddings instead of absolute positional embeddings. This means the following:
- The `Classifier` and `Block` classes are unchanged.
- The `Encoder` and the `Decoder` have only 1 embedding layer (instead of 2, as in Parts 1 and 2) each. 
- The `MultiHeadAttention` class has a new parameter called `m` which has a different constant value (power of 2) for each transformer head.
- The `AttentionHead` class now implements ALiBi where it adds a `bias` matrix to the attention weights to encode position of the key (`k`) vectors relative to the position of the query (`q`) vector. The value (`v`) vectors do not encode position information. 

#### Part 3.2: Performance Improvement
This involves running the `classification_task()` and `language_modeling_task()` functions by setting the `use_init_weights` argument to be `True`. The steps are the same as Parts 1 and 2. The only difference is that the transformer now initializes the weights by sampling from a normal distribution with mean of `0` and standard deviation of `0.05`. This is different from Parts 1 and 2 where the weights are initialized randomly.

<br>
<br>

## Outputs
The outputs are written to JSON files. Each file contains the following details:
1. `task`: The task, whether classification or language modeling
2. `num_params`: The number of model parameters
3. `use_alibi`: Whether the model uses absolute positional embeddings or ABiLi positional embeddings
4. `use_init_weights`: Whether the model weights are initialized randomly or from a normal distribution
5. `history`: Rhe model history, which includes train and test metrics for each training epoch.

Following are the JSON files generated:
- Part 1: `part1_classification_task.json`
- Part 2: `part2_language_modeling_task.json`
- Part 3:  
    - `part3_architectural_exploration_classification_task.json`
    - `part3_architectural_exploration_language_modeling_task.json`
    - `part3_performance_improvement_classification_task.json`
    - `part3_performance_improvement_language_modeling_task.json`

<br>
<br>

## Project Report, Tables, Plots, and Visualizations
To regenerate the tables, plots, and other visualizations used in the [Project Report](./Project_Report.pdf), please refer to the `visualize.ipynb` Jupyter notebook.

<br>
<br>

## Code References
1. Transformer Architecture: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
2. ALiBi Positional Embeddings: https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py

<br>
<br>
