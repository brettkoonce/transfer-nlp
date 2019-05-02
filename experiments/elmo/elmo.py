import logging
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
# from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm

from transfer_nlp.common.tokenizers import CustomTokenizer
from transfer_nlp.embeddings.embeddings import Embedding
from transfer_nlp.loaders.loaders import DatasetSplits, DataFrameDataset, DatasetHyperParams
from transfer_nlp.loaders.vectorizers import Vectorizer
from transfer_nlp.loaders.vocabulary import Vocabulary, SequenceVocabulary
from transfer_nlp.plugins.config import ExperimentConfig
from transfer_nlp.plugins.config import register_plugin
from transfer_nlp.plugins.helpers import ObjectHyperParams
from transfer_nlp.plugins.predictors import PredictorABC, PredictorHyperParams

tqdm.pandas()

logger = logging.getLogger(__name__)

options_file = Path.home() / "work/transfer-nlp-data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = Path.home() / "work/transfer-nlp-data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file=options_file, weight_file=weight_file, num_output_representations=2, requires_grad=False, dropout=0)


# elmo = ElmoEmbedder(options_file, weight_file)
# Vectorizer class
@register_plugin
class NewsVectorizer(Vectorizer):
    def __init__(self, data_file: str, cutoff: int):

        super().__init__(data_file=data_file)
        self.cutoff = cutoff

        self.tokenizer = CustomTokenizer()
        df = pd.read_csv(data_file)

        target_vocab = Vocabulary(add_unk=False)
        for category in sorted(set(df.category)):
            target_vocab.add_token(category)

        word_counts = Counter()
        max_title = 0
        for title in df.title:
            tokens = self.tokenizer.tokenize(text=title)
            max_title = max(max_title, len(tokens))
            for token in tokens:
                if token not in string.punctuation:
                    word_counts[token] += 1

        data_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= self.cutoff:
                data_vocab.add_token(word)

        self.data_vocab = data_vocab
        self.target_vocab = target_vocab
        self.max_title = max_title + 2

    def vectorize(self, title: str) -> np.array:

        tokens = self.tokenizer.tokenize(text=title)
        indices = [self.data_vocab.begin_seq_index]
        indices.extend(self.data_vocab.lookup_token(token)
                       for token in tokens)
        indices.append(self.data_vocab.end_seq_index)
        vector_length = self.max_title

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.data_vocab.mask_index

        return out_vector


# Dataset class
@register_plugin
class NewsDataset(DatasetSplits):

    def __init__(self, data_file: str, batch_size: int, dataset_hyper_params: DatasetHyperParams):
        self.df = pd.read_csv(data_file)
        np.random.shuffle(self.df.values)  # Use this code in dev mode
        N = 10000
        self.df = self.df.head(n=N)

        # preprocessing
        self.vectorizer: Vectorizer = dataset_hyper_params.vectorizer

        self.df['x_in'] = self.df.progress_apply(lambda row: self.vectorizer.vectorize(row.title), axis=1)
        self.df['y_target'] = self.df.progress_apply(lambda row: self.vectorizer.target_vocab.lookup_token(row.category), axis=1)
        # self.df['elmo'] = self.df.progress_apply(lambda row: elmo.embed_sentence(row.title.split(' ')), axis=1)
        list_of_tokens = [title.split(' ') for title in self.df.title]
        character_ids = batch_to_ids(list_of_tokens)

        self.embeddings = elmo(character_ids)
        elmos = [torch.sum(emb, dim=0) for emb in tqdm(self.embeddings['elmo_representations'][0])]
        self.df['elmo'] = elmos

        # self.df['elmo'] = self.df.progress_apply(lambda row: torch.sum(elmo(batch_to_ids([row.title.split(' ')]))['elmo_representations'][0], dim=1), axis=1)

        train_df = self.df[self.df.split == 'train'][['x_in', 'y_target', 'elmo']]
        val_df = self.df[self.df.split == 'val'][['x_in', 'y_target', 'elmo']]
        test_df = self.df[self.df.split == 'test'][['x_in', 'y_target', 'elmo']]

        super().__init__(train_set=DataFrameDataset(train_df), train_batch_size=batch_size,
                         val_set=DataFrameDataset(val_df), val_batch_size=batch_size,
                         test_set=DataFrameDataset(test_df), test_batch_size=batch_size)


# Model
@register_plugin
class NewsCNNHyperParams(ObjectHyperParams):

    def __init__(self, dataset_splits: DatasetSplits):
        super().__init__()
        self.num_embeddings = len(dataset_splits.vectorizer.data_vocab)
        self.num_classes = len(dataset_splits.vectorizer.target_vocab)


@register_plugin
class EmbeddingtoModelHyperParams(ObjectHyperParams):

    def __init__(self, embeddings: Embedding):
        super().__init__()
        self.embeddings = embeddings.embeddings


@register_plugin
class NewsClassifier(torch.nn.Module):

    def __init__(self, model_hyper_params: ObjectHyperParams, embedding_size: int, num_channels: int,
                 hidden_dim: int, dropout_p: float, padding_idx: int = 0, embeddings2model_hyper_params: ObjectHyperParams = None):
        super(NewsClassifier, self).__init__()

        self.num_embeddings: int = model_hyper_params.num_embeddings
        self.num_classes: int = model_hyper_params.num_classes
        self.num_channels: int = num_channels
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

        if embeddings2model_hyper_params:
            logger.info("Using pre-trained word embeddings...")
            self.embeddings = embeddings2model_hyper_params.embeddings
            self.embeddings = torch.from_numpy(self.embeddings).float()
            self.emb: torch.nn.Embedding = torch.nn.Embedding(embedding_dim=self.embedding_size,
                                                              num_embeddings=self.num_embeddings,
                                                              padding_idx=self.padding_idx,
                                                              _weight=self.embeddings)

        else:
            logger.info("Not using pre-trained word embeddings...")
            self.emb: torch.nn.Embedding = torch.nn.Embedding(embedding_dim=self.embedding_size,
                                                              num_embeddings=self.num_embeddings,
                                                              padding_idx=self.padding_idx)

        self.convnet = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.embedding_size,
                            out_channels=self.num_channels, kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3, stride=1),
            torch.nn.ELU(),
            torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels,
                            kernel_size=3),  # Experimental change from 3 to 2
            torch.nn.ELU()
        )

        self._dropout_p: float = dropout_p
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.num_channels + 1024, self.hidden_dim)
        self.fc2: torch.nn.Linear = torch.nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x_in: torch.tensor, elmo: torch.tensor) -> torch.Tensor:
        """

        :param x_in: input data tensor
        :param apply_softmax: flag for the softmax activation
                should be false if used with the Cross Entropy losses
        :return: the resulting tensor. tensor.shape should be (batch, num_classes)
        """

        # embed and permute so features are channels
        x_embedded = self.emb(x_in).permute(0, 2, 1)

        features = self.convnet(x_embedded)

        # average and remove the extra dimension
        remaining_size = features.size(dim=2)
        features = torch.nn.functional.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = self.dropout(features)

        features = torch.cat(tensors=(features, elmo.squeeze(dim=1)), dim=1)

        # mlp classifier
        intermediate_vector = torch.nn.functional.relu(self.dropout(self.fc1(features)))
        prediction_vector = self.fc2(intermediate_vector)

        return prediction_vector


# Predictors
@register_plugin
class NewsPredictor(PredictorABC):
    """
    Toy example: we want to make predictions on inputs of the form {"inputs": ["hello world", "foo", "bar"]}
    """

    def __init__(self, predictor_hyper_params: PredictorHyperParams):
        super().__init__(predictor_hyper_params=predictor_hyper_params)

    def json_to_data(self, input_json: Dict) -> Dict:
        return {
            'x_in': torch.LongTensor([self.vectorizer.vectorize(title=input_string) for input_string in input_json['inputs']])}

    def output_to_json(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "outputs": outputs}

    def decode(self, output: torch.tensor) -> List[Dict[str, Any]]:
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probability_values, indices = probabilities.max(dim=1)

        return [{
            "class": self.vectorizer.target_vocab.lookup_index(index=int(res[1])),
            "probability": float(res[0])} for res in zip(probability_values, indices)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment_config = {
        "data_file": "HOME/ag_news/news_with_splits.csv",
        "tensorboard_logs": "HOME/ag_news/tensorboard/cnn",
        "glove_filepath": "HOME/glove/glove.6B.100d.txt",
        "embedding_size": 100,
        "num_channels": 10,
        "hidden_dim": 10,
        "dropout_p": 0.3,
        "seed": 1337,
        "lr": 0.001,
        "cutoff": 5,
        "batch_size": 32,
        "num_epochs": 5,
        "early_stopping_criteria": 5,
        "alpha": 0.01,
        "gradient_clipping": 0.25,
        "mode": "min",
        "factor": 0.5,
        "patience": 1,
        "vectorizer": {
            "_name": "NewsVectorizer"
        },
        "embeddings": {
            "_name": "Embedding"
        },
        "embedding_hyper_params": {
            "_name": "EmbeddingsHyperParams"
        },
        "embeddings2model_hyper_params": {
            "_name": "EmbeddingtoModelHyperParams"
        },
        "dataset_hyper_params": {
            "_name": "DatasetHyperParams"
        },
        "dataset_splits": {
            "_name": "NewsDataset"
        },
        "model": {
            "_name": "NewsClassifier"
        },
        "model_hyper_params": {
            "_name": "NewsCNNHyperParams"
        },
        "model_params": {
            "_name": "TrainableParameters"
        },
        "loss": {
            "_name": "CrossEntropyLoss"
        },
        "optimizer": {
            "_name": "Adam",
            "params": "model_params"
        },
        "regularizer": {
            "_name": "L1"
        },
        "scheduler": {
            "_name": "ReduceLROnPlateau"
        },
        "accuracy": {
            "_name": "Accuracy"
        },
        "lossMetric": {
            "_name": "LossMetric",
            "loss_fn": "loss"
        },
        "trainer": {
            "_name": "BasicTrainer",
            "metrics": [
                "accuracy",
                "lossMetric"
            ]
        },
        "predictor": {
            "_name": "NewsPredictor"
        },
        "predictor_hyper_params": {
            "_name": "PredictorHyperParams"
        }
    }
    home_env = str(Path.home() / 'work/transfer-nlp-data')
    experiment = ExperimentConfig(experiment_config, HOME=home_env)
    experiment['trainer'].train()

    # sentences = [['First', 'sentence', '.']]*100 + [['Second', 'longer', 'sentence', '.']]*100
    # character_ids = batch_to_ids(sentences)
    #
    # embeddings = elmo(character_ids)
