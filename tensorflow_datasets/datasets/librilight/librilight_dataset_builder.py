"""librilight dataset."""

import os
from typing import Dict, Iterator, Tuple

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf

_CITATION = """\
@inproceedings{librilight,
  author={J. {Kahn} and M. {Rivière} and W. {Zheng} and E. {Kharitonov} and Q. {Xu} and P. E. {Mazaré} and J. {Karadayi} and V. {Liptchinsky} and R. {Collobert} and C. {Fuegen} and T. {Likhomanenko} and G. {Synnaeve} and A. {Joulin} and A. {Mohamed} and E. {Dupoux}},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Libri-Light: A Benchmark for ASR with Limited or No Supervision}, 
  year={2020},
  pages={7669-7673},
}
"""
_URL = "https://github.com/facebookresearch/libri-light"
_DL_URLS = "/Users/zhiyunlu/Documents/projects/data/LibriLight"


class Librilight(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for librilight dataset."""

    VERSION = tfds.core.Version("5.0.1")
    RELEASE_NOTES = {
        "5.0.1": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    manual_dir should contain three folders large medium and small. The instructions for
    downloading this file are found in {}.
    """.format(
        _URL
    )

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="LibriLight dataset.",
            features=tfds.features.FeaturesDict(
                {
                    "speech": tfds.features.Audio(sample_rate=16000, dtype=tf.int16),
                    "id": tf.string,
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(
                sample_rate=16000,
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        extracted_dirs = dict()
        for subdir in tf.io.gfile.listdir(dl_manager.manual_dir):
            extracted_dirs[subdir] = os.path.join(dl_manager.manual_dir, subdir)
        splits = {
            split: self._generate_examples(directory)
            for split, directory in extracted_dirs.items()
        }
        return splits

    # pylint: disable-next=no-self-use
    def _generate_examples(
        self, split_dir: str
    ) -> tfds.core.split_builder.SplitGenerator:
        beam = tfds.core.lazy_imports.apache_beam

        def _generate_librispeech_examples(
            audio_file,
        ) -> Iterator[Tuple[str, Dict[str, str]]]:
            """Generate examples from a Librispeech directory."""
            key = "-".join(
                os.path.dirname(audio_file).split("/")[-3:]
                + [os.path.basename(audio_file)[:-5]]
            )
            example = {
                "id": key,
                "speech": audio_file,
            }
            yield key, example

        return (
            beam.Create(tf.io.gfile.glob(os.path.join(split_dir, "*/*/*.flac")))
            | beam.FlatMap(_generate_librispeech_examples)
            | beam.Reshuffle()
        )
