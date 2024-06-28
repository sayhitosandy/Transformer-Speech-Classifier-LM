import argparse
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from tokenizer import SimpleTokenizer
from transformer import Classifier, Decoder
from transformer_alibi import Classifier as ClassifierALiBi, Decoder as DecoderALiBi

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters to use for training to roughly match 
# the numbers mentioned in the assignment description
batch_size = 16  # Number of independent sequences we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a
# while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly,
# this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

# Classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
# size of 64, hidden size of 50 and output size of 3.
n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_cls = 15  # Epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The
    text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training,
    but we still need to ignore the test data.

    :param directory: The directory to load the texts from.
    :return: Texts
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  # don't "read test files"
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """
    Collate a batch of data into a single tensor with padding.

    :param batch: A batch of data.
    :return: Sequences padded to a fixed length, labels
    """

    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(
        padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """
    Compute the accuracy of the classifier on the data in data_loader.

    :param classifier: The classifier
    :param data_loader: The data loader
    :return: Classification accuracy
    """

    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs, _ = classifier(x)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == y).sum().item()
            total_samples += y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoder_model, data_loader, iters_eval=100):
    """
    Compute the perplexity of the decoder_model on the data in data_loader.
    Make sure to use the cross entropy loss for the decoder_model.

    :param decoder_model: The decoder model
    :param data_loader: The data loader
    :param iters_eval: The number of iterations
    :return: Decoder perplexity
    """

    decoder_model.eval()
    losses = []
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        loss, _ = decoder_model(x, y)  # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= iters_eval:
            break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoder_model.train()
    return perplexity


def get_cls_data_loader(file_path, tokenizer):
    """
    Get an iterable over the dataset for classification task.

    :param file_path: The file path
    :param tokenizer: The tokenizer
    :return: Dataset iterator
    """

    dataset = SpeechesClassificationDataset(tokenizer=tokenizer, file_path=file_path)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    return loader


def get_lm_data_loader(file_path, tokenizer):
    """
    Get an iterable over the dataset for language modeling task.

    :param file_path: The file path
    :param tokenizer: The tokenizer
    :return: Dataset iterator
    """

    with open(file=file_path, mode="r", encoding="utf-8") as f:
        text = f.read()
    dataset = LanguageModelingDataset(tokenizer=tokenizer, text=text, block_size=block_size)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return loader


def classification_task(tokenizer, use_alibi=False, use_init_weights=False,
                        perform_sanity_check=False, verbose=True):
    """
    Perform the classification task on the dataset.

    :param tokenizer: The tokenizer
    :param use_alibi: Use ALiBi positional embeddings
    :param use_init_weights: Use weights initialized from normal distribution
    :param perform_sanity_check: Perform sanity check
    :param verbose: Debug mode
    :return: Number of model parameters, the train loss, the train accuracy, and the test accuracy
    """

    train_cls_loader = get_cls_data_loader(
        file_path="speechesdataset/train_CLS.tsv", tokenizer=tokenizer)
    test_cls_loader = get_cls_data_loader(
        file_path="speechesdataset/test_CLS.tsv", tokenizer=tokenizer)

    func = Classifier if use_alibi == False else ClassifierALiBi
    classifier = func(
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        block_size=block_size,
        dropout_prob=0.,
        n_input=n_input,
        n_hidden=4 * n_embd,
        n_output=n_output,
        use_init_weights=use_init_weights
    )
    classifier.to(device)

    num_params = sum(p.numel() for p in classifier.parameters())
    if verbose:
        print(f"No. of parameters: {num_params}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    classifier.train()

    n_batches = len(train_cls_loader)

    train_loss = []
    train_accuracy = []
    test_accuracy = []
    for epoch in tqdm(range(epochs_cls)):
        current_loss = 0.
        for xb, yb in train_cls_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, _ = classifier(xb)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pred, yb)
            current_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(current_loss / n_batches)
        train_accuracy.append(compute_classifier_accuracy(
            classifier=classifier, data_loader=train_cls_loader))
        test_accuracy.append(compute_classifier_accuracy(
            classifier=classifier, data_loader=test_cls_loader))

        if verbose:
            print(f"Iteration: {epoch + 1}")
            print(f"Train Loss: {train_loss[-1]}")
            print(f"Train Accuracy: {train_accuracy[-1]}")
            print(f"Test Accuracy: {test_accuracy[-1]}")
            print("-" * 20)

    # Sanity Check
    if perform_sanity_check:
        import utilities

        utility = utilities.Utilities(tokenizer=tokenizer, model=classifier)
        utility.sanity_check(sentence="This is an immensely important day, a day that belongs to all of you.", block_size=block_size)
        utility.sanity_check(sentence="None of these changes happened overnight.", block_size=block_size)

    return num_params, train_loss, train_accuracy, test_accuracy


def language_modeling_task(tokenizer, use_alibi=False, use_init_weights=False,
                           perform_sanity_check=False, verbose=True):
    """
    Perform the language modeling task on the dataset.

    :param tokenizer: The tokenizer
    :param use_alibi: Use ALiBi positional embeddings
    :param use_init_weights: Use weights initialized from normal distribution
    :param perform_sanity_check: Perform sanity check
    :param verbose: Debug mode
    :return: Number of model parameters, the train perplexity, the hbush test perplexity,
    the obama test perplexity, and the wbush test perplexity
    """

    train_lm_loader = get_lm_data_loader(
        file_path="speechesdataset/train_LM.txt", tokenizer=tokenizer)
    hbush_test_lm_loader = get_lm_data_loader(
        file_path="speechesdataset/test_LM_hbush.txt", tokenizer=tokenizer)
    obama_test_lm_loader = get_lm_data_loader(
        file_path="speechesdataset/test_LM_obama.txt", tokenizer=tokenizer)
    wbush_test_lm_loader = get_lm_data_loader(
        file_path="speechesdataset/test_LM_wbush.txt", tokenizer=tokenizer)

    func = Decoder if use_alibi == False else DecoderALiBi
    decoder = func(
        vocab_size=tokenizer.vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_heads=n_head,
        block_size=block_size,
        n_hidden=4 * n_embd,
        dropout_prob=0.,
        use_init_weights=use_init_weights
    )
    decoder.to(device)

    num_params = sum(p.numel() for p in decoder.parameters())
    if verbose:
        print(f"No. of parameters: {num_params}")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    decoder.train()

    train_perplexity = []
    hbush_test_perplexity = []
    obama_test_perplexity = []
    wbush_test_perplexity = []
    for i, (xb, yb) in tqdm(enumerate(train_lm_loader)):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        loss, _ = decoder(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            train_perplexity.append(compute_perplexity(
                decoder_model=decoder, data_loader=train_lm_loader))

            hbush_test_perplexity.append(compute_perplexity(
                decoder_model=decoder, data_loader=hbush_test_lm_loader))
            obama_test_perplexity.append(compute_perplexity(
                decoder_model=decoder, data_loader=obama_test_lm_loader))
            wbush_test_perplexity.append(compute_perplexity(
                decoder_model=decoder, data_loader=wbush_test_lm_loader))

            if verbose:
                print(f"Iteration: {i + 1}")
                print(f"Train Perplexity: {train_perplexity[-1]}")
                print(f"H Bush Test Perplexity: {hbush_test_perplexity[-1]}")
                print(f"Obama Test Perplexity: {obama_test_perplexity[-1]}")
                print(f"W Bush Test Perplexity: {wbush_test_perplexity[-1]}")
                print("-" * 20)

    # Sanity Check
    if perform_sanity_check:
        import utilities

        utility = utilities.Utilities(tokenizer=tokenizer, model=decoder)
        utility.sanity_check(sentence="Our relations abroad were strained.", block_size=block_size)
        utility.sanity_check(sentence="America, we cannot turn back.", block_size=block_size)

    return num_params, train_perplexity, hbush_test_perplexity, obama_test_perplexity, wbush_test_perplexity


def write_output_to_json(is_classification, history, params, file_path):
    """
    Write output to json file

    :param is_classification: Is classification task?
    :param history: Model history
    :param params: Model parameters
    :param file_path: Output file path
    :return: Output dictionary
    """

    import json

    if is_classification:
        num_params, train_loss, train_accuracy, test_accuracy = history
        output = {
            "task": "classification",
            "num_params": num_params
        }
        output.update(params)

        for epoch in range(epochs_cls):
            output["history"].append({
                "epoch": epoch + 1,
                "train_loss": train_loss[epoch],
                "train_accuracy": train_accuracy[epoch],
                "test_accuracy": test_accuracy[epoch]
            })
        with open(file_path, "w") as f:
            json.dump(output, f, indent=4)

    else:
        num_params, train_perplexity, hbush_test_perplexity, \
            obama_test_perplexity, wbush_test_perplexity = history
        output = {
            "task": "language_modeling",
            "num_params": num_params
        }
        output.update(params)
        for epoch in range(0, max_iters, eval_interval):
            output["history"].append({
                "epoch": epoch + eval_interval,
                "train_perplexity": train_perplexity[epoch // eval_interval],
                "hbush_test_perplexity": hbush_test_perplexity[epoch // eval_interval],
                "obama_test_perplexity": obama_test_perplexity[epoch // eval_interval],
                "wbush_test_perplexity": wbush_test_perplexity[epoch // eval_interval]
            })
        with open(file_path, "w") as f:
            json.dump(output, f, indent=4)

    return output


def main(part, perform_sanity_check=False, verbose=False):
    """
    Main function

    :param part: Part (part1, part2, part3)
    :param perform_sanity_check: Perform sanity check
    :param verbose: Debug mode
    :return:  
    """
    
    print("Loading data and creating tokenizer...")
    texts = load_texts(directory="speechesdataset")
    tokenizer = SimpleTokenizer(text=" ".join(texts))  # create a tokenizer from the data
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    if part == "part1":  # Classification
        params = {
            "use_alibi": False,
            "use_init_weights": False,
            "history": []
        }
        history = classification_task(
            tokenizer=tokenizer,
            use_alibi=params["use_alibi"],
            use_init_weights=params["use_init_weights"],
            perform_sanity_check=perform_sanity_check,
            verbose=verbose
        )
        write_output_to_json(is_classification=True, history=history, params=params,
                             file_path="./part1_classification_task.json")

    if part == "part2":  # Language Modeling
        params = {
            "use_alibi": False,
            "use_init_weights": False,
            "history": []
        }
        history = language_modeling_task(
            tokenizer=tokenizer,
            use_alibi=params["use_alibi"],
            use_init_weights=params["use_init_weights"],
            perform_sanity_check=perform_sanity_check,
            verbose=verbose
        )
        write_output_to_json(is_classification=False, history=history, params=params,
                             file_path="./part2_language_modeling_task.json")

    if part == "part3":  # Exploration
        # ----------------Architectural Exploration----------------
        # Use Attention with Linear Biases (ALiBi) positional embeddings

        # Classification
        params_ae_ct = {
            "use_alibi": True,
            "use_init_weights": False,
            "history": []
        }
        history_ae_ct = classification_task(
            tokenizer=tokenizer,
            use_alibi=params_ae_ct["use_alibi"],
            use_init_weights=params_ae_ct["use_init_weights"],
            perform_sanity_check=perform_sanity_check,
            verbose=verbose
        )
        write_output_to_json(is_classification=True, history=history_ae_ct, params=params_ae_ct,
                             file_path="./part3_architectural_exploration_classification_task.json")

        # Language Modeling
        params_ae_lm = {
            "use_alibi": True,
            "use_init_weights": False,
            "history": []
        }
        history_ae_lm = language_modeling_task(
            tokenizer=tokenizer,
            use_alibi=params_ae_lm["use_alibi"],
            use_init_weights=params_ae_lm["use_init_weights"],
            perform_sanity_check=perform_sanity_check,
            verbose=verbose
        )
        write_output_to_json(is_classification=False, history=history_ae_lm, params=params_ae_lm,
                             file_path="./part3_architectural_exploration_language_modeling_task.json")

        # -----------------Performance Improvement-----------------
        # Use better weights initialization

        # Classification
        params_pi_ct = {
            "use_alibi": False,
            "use_init_weights": True,
            "history": []
        }
        history_pi_ct = classification_task(
            tokenizer=tokenizer,
            use_alibi=params_pi_ct["use_alibi"],
            use_init_weights=params_pi_ct["use_init_weights"],
            perform_sanity_check=perform_sanity_check,
            verbose=verbose
        )
        write_output_to_json(is_classification=True, history=history_pi_ct, params=params_pi_ct,
                             file_path="./part3_performance_improvement_classification_task.json")

        # Language Modeling
        params_pi_lm = {
            "use_alibi": False,
            "use_init_weights": True,
            "history": []
        }
        history_pi_lm = language_modeling_task(
            tokenizer=tokenizer,
            use_alibi=params_pi_lm["use_alibi"],
            use_init_weights=params_pi_lm["use_init_weights"],
            perform_sanity_check=perform_sanity_check,
            verbose=verbose
        )
        write_output_to_json(is_classification=False, history=history_pi_lm, params=params_pi_lm,
                             file_path="./part3_performance_improvement_language_modeling_task.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA2")
    parser.add_argument("part", type=str, choices=["part1", "part2", "part3"],
                        help="Select either part1, part2, or part3")
    parser.add_argument("--verbose", type=bool, choices=[True, False], default=False, 
                        help="Enable verbose mode")
    parser.add_argument("--perform-sanity-check", type=bool, choices=[True, False], default=False, 
                        help="Perform a sanity check")

    args = parser.parse_args()
    main(part=args.part, verbose=args.verbose, perform_sanity_check=args.perform_sanity_check)
