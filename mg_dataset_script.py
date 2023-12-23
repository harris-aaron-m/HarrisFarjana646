from __future__ import absolute_import, division, print_function

import csv
import os

import nlp


#Here we have to specify the path where the dataset is
#Since i'm using colab, i simply put the colab "path"

_TEST_FILE_NAME = "/home/amharris/aug/mg_test.tsv"
#_TRAIN_FILE_NAME = "/content/train.tsv"


class BFPConfig(nlp.BuilderConfig):

    """BuilderConfig for Break"""

    def __init__(self, **kwargs):
        super(BFPConfig, self).__init__(
            version=nlp.Version("1.0.0", "New split API (https://tensorflow.org/datasets/splits)"), **kwargs
        )


class Bfp(nlp.GeneratorBasedBuilder):

    VERSION = nlp.Version("0.1.0")
    BUILDER_CONFIGS = [
        BFPConfig(
            name="bfp",
            #data_url=_DATA_URL,
        )
    ]

    def _info(self):
        return nlp.DatasetInfo(
            # nlp.features.FeatureConnectors
            features=nlp.Features(
                {
                    "buggy": nlp.Value("string"),
                    "fixed": nlp.Value("string")
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        #data_dir = dl_manager.download_and_extract(_DATA_URL)

        #test_csv_file = os.path.join(data_dir, _TEST_FILE_NAME)
        #train_csv_file = os.path.join(data_dir, _TRAIN_FILE_NAME)

        test_csv_file = _TEST_FILE_NAME
        #train_csv_file = _TRAIN_FILE_NAME

        if self.config.name == "bfp":
            return [
                
                nlp.SplitGenerator(
                    name=nlp.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={"file_path": test_csv_file},
                )
                
            ]
        else:
            raise NotImplementedError("{} does not exist".format(self.config.name))

    def _generate_examples(self, file_path):
        """Yields examples."""

        with open(file_path, encoding="ISO-8859-1") as f:
            data = csv.reader(f, delimiter='\t', quotechar='"')
            for row_id, row in enumerate(data):
                buggy,fixed = row
                yield "{}".format(row_id), {
                    "buggy": buggy,
                    "fixed": fixed
                }