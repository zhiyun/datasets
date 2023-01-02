"""librilight dataset.

Use with
tfds build --manual_dir=/Users/zhiyunlu/Documents/projects/data/LibriLight
"""
import os
from etils import epath
import tensorflow_datasets as tfds
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
#  "/Users/zhiyunlu/Documents/projects/tfds/data/"


class LibriLight(tfds.core.BeamBasedBuilder):
    """DatasetBuilder for librilight dataset."""
    
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    manual_dir should contain three folders large medium and small. The instructions for
    downloading this file are found in {}.
    """.format(_URL)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
          builder=self,
          description="LibriLight dataset.",
          features=tfds.features.FeaturesDict({
              "speech": tfds.features.Audio(sample_rate=16000, dtype=tf.int16),
              "id": tf.string,
          }),
          supervised_keys=None,
          homepage=_URL,
          citation=_CITATION,
          metadata=tfds.core.MetadataDict(sample_rate=16000, ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        extracted_dirs = dict()
        for subdir in tf.io.gfile.listdir(dl_manager.manual_dir):
            extracted_dirs[subdir] = epath.Path(os.path.join(dl_manager.manual_dir, subdir))
        splits = [
            tfds.core.SplitGenerator(name=split, gen_kwargs={"directory": directory})
            for split, directory in extracted_dirs.items()
        ]
        return splits


    def _build_pcollection(self, pipeline, directory):
        """Generates examples as dicts."""
        beam = tfds.core.lazy_imports.apache_beam
        return (pipeline
                | beam.Create([directory])
                | beam.FlatMap(_generate_librispeech_examples)
                | beam.Reshuffle())


def _generate_librispeech_examples(directory):
    """Generate examples from a Librispeech directory."""
    audio_glob = os.path.join(directory, "*/*/*.flac")
    for audio_file in tf.io.gfile.glob(audio_glob):
        key = "-".join(os.path.dirname(audio_file).split("/")[-3:] + [os.path.basename(audio_file)[:-5]])
        example = {
            "id": key,
            "speech": audio_file,
        }
        yield key, example