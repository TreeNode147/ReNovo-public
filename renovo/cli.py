# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Melih Yilmaz.
# Derived from Casanovo; modified for ReNovo.
# See LICENSE and licenses/ at the repository root.

"""The command line entry point for ReNovo."""
import datetime
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple
import tempfile
warnings.formatwarning = lambda message, category, *args, **kwargs: (
    f"{category.__name__}: {message}"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    ".*Consider increasing the value of the `num_workers` argument*",
)

warnings.filterwarnings(
    "ignore",
    ".*The PyTorch API of nested tensors is in prototype stage*",
)
warnings.filterwarnings(
    "ignore",
    ".*Converting mask without torch.bool dtype to bool*",
)

import rich_click as click
from lightning.pytorch import seed_everything

from .denovo import ModelRunner
from .config import Config

logger = logging.getLogger("renovo")
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True


class _SharedParams(click.RichCommand):
    """Options shared between ReNovo commands"""

    def __init__(self, *args, **kwargs) -> None:
        """Define shared options."""
        super().__init__(*args, **kwargs)
        self.params += [
            click.Option(
                ("-m", "--model"),
                help="""
                The ReNovo model weights (.ckpt file). Required for building
                a datastore and evaluating spectra; optional for training.
                """,
                type=click.Path(exists=True, dir_okay=False),
            ),
            click.Option(
                ("-o", "--output"),
                help="The output log prefix (a .log suffix is added).",
                type=click.Path(dir_okay=False),
            ),
            click.Option(
                ("-c", "--config"),
                help="""
                The YAML configuration file overriding the default options.
                """,
                type=click.Path(exists=True, dir_okay=False),
            ),
            click.Option(
                ("-v", "--verbosity"),
                help="""
                Set the verbosity of console logging messages. Log files are
                always set to 'debug'.
                """,
                type=click.Choice(
                    ["debug", "info", "warning", "error"],
                    case_sensitive=False,
                ),
                default="info",
            ),
        ]


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """ReNovo experiment reproduction: train, build, and evaluate.

    Adapted from Casanovo. License terms and upstream license texts are provided
    in LICENSE and licenses/ at the repository root.
    """
    return


@main.command(cls=_SharedParams)
# @click.argument(
#     "--annotated_peak_path",
#     required=True,
#     nargs=-1,
#     type=click.Path(exists=True, dir_okay=False),
# )
@click.option(
    "--annotated_peak_path",
    required=True,
    multiple=True,
)
@click.option(
    "--is_knn_inference",
    is_flag=True,
    help="Whether to use knn inference",
)
@click.option(
    "--datastore_path",
    required=False,
    type=click.Path(exists=False, dir_okay=True),
)
@click.option(
    "--knn_k",
    default=8,
    type=click.IntRange(min=1),
)
@click.option(
    "--knn_temperature",
    default=10,
    type=click.FloatRange(min=0.0)
)
def evaluate(
    annotated_peak_path: Tuple[str],
    is_knn_inference: bool,
    datastore_path: Optional[str],
    knn_k: Optional[int],
    knn_temperature: Optional[float],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """Evaluate de novo peptide sequencing performance.

    ANNOTATED_PEAK_PATH must be one or more annoated MGF files,
    such as those provided by MassIVE-KB.
    """
    output = setup_logging(output, verbosity)
    config, model = setup_model(model, config, output, False)
    with ModelRunner(config, model) as runner:
        if is_knn_inference:
            logger.info("KNN inference peptides from:")
        else:
            logger.info("Sequencing and evaluating peptides from:")
            
        for peak_file in annotated_peak_path:
            logger.info("  %s", peak_file)

        knn_args = (knn_k, knn_temperature)

        if is_knn_inference:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_datastore_path = os.path.join(temp_dir, os.path.basename(datastore_path))
                shutil.copytree(datastore_path, temp_datastore_path)
                runner.evaluate(annotated_peak_path, is_knn_inference=is_knn_inference, datastore_path=temp_datastore_path, knn_args=knn_args)
        else:
            runner.evaluate(annotated_peak_path, is_knn_inference=is_knn_inference)

    logger.info("DONE!")
    

@main.command(cls=_SharedParams)
@click.option(
    "--build_data",
    required=True,
    multiple=True,
)
@click.option(
    "--datastore_path",
    required=True,
    type=click.Path(exists=False, dir_okay=True),
)
def build(
    build_data: Tuple[str],
    datastore_path: Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """Build a kNN datastore from annotated spectra.

    Use --build_data for each annotated MGF file to include in the datastore.
    """
    output = setup_logging(output, verbosity)
    if model is None:
        raise RuntimeError("No given model when building datastore!")
    config, model = setup_model(model, config, output, False)
    
    with ModelRunner(config, model) as runner:
    
        logger.info("Building datastore from mgf data:")
        for peak_file in build_data:
            logger.info("  %s", peak_file)
            
        logger.info(f"Datastore saved in: {datastore_path}")
        runner.build_datastore(build_data, datastore_path)

    logger.info("DONE!")


@main.command(cls=_SharedParams)
##### 012
# @click.argument(
#     "train_peak_path",
#     required=True,
#     nargs=-1,
#     type=click.Path(exists=True, dir_okay=False),
# )
@click.option(
    "--train_peak_path",
    help="""
    An annotated MGF file for validation, like from MassIVE-KB. Use this
    option multiple times to specify multiple files.
    """,
    required=True,
    multiple=True,
    # type=click.Path(exists=True, dir_okay=False),
)
##### 012
@click.option(
    "-p",
    "--validation_peak_path",
    help="""
    An annotated MGF file for validation, like from MassIVE-KB. Use this
    option multiple times to specify multiple files.
    """,
    required=True,
    multiple=True,
    # type=click.Path(exists=True, dir_okay=False), ##### 012
)
def train(
    train_peak_path: Tuple[str],
    validation_peak_path: Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """Train a ReNovo model on your own data.

    TRAIN_PEAK_PATH must be one or more annoated MGF files, such as those
    provided by MassIVE-KB, from which to train a new ReNovo model.
    """
    output = setup_logging(output, verbosity)
    config, model = setup_model(model, config, output, True)
    with ModelRunner(config, model) as runner:
        logger.info("Training a model from:")
        for peak_file in train_peak_path:
            logger.info("  %s", peak_file)

        logger.info("Using the following validation files:")
        for peak_file in validation_peak_path:
            logger.info("  %s", peak_file)

        runner.train(train_peak_path, validation_peak_path)

    logger.info("DONE!")


def setup_logging(
    output: Optional[str],
    verbosity: str,
) -> Path:
    """Set up the logger.

    Logging occurs to the command-line and to the given log file.

    Parameters
    ----------
    output : Optional[str]
        The provided output file name.
    verbosity : str
        The logging level to use in the console.

    Return
    ------
    output : Path
        The output file path.
    """
    if output is None:
        output = (Path(__file__).resolve().parents[1] / "logs" / "model" /
                  f"renovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Configure logging.
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    warnings_logger = logging.getLogger("py.warnings")

    # Formatters for file vs console:
    console_formatter = logging.Formatter("{levelname}: {message}", style="{")
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    ##### 999
    # console_handler.setLevel(logging_levels[verbosity.lower()])
    console_handler.setLevel(logging.DEBUG)
    ##### 999
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    warnings_logger.addHandler(console_handler)
    file_handler = logging.FileHandler(output.with_suffix(".log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    warnings_logger.addHandler(file_handler)

    # Disable dependency non-critical log messages.
    ##### 999
    # logging.getLogger("depthcharge").setLevel(
    #     logging_levels[verbosity.lower()]
    # )
    # logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("fsspec").setLevel(logging.DEBUG)
    ##### 999
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return output


def setup_model(
    model: Optional[str],
    config: Optional[str],
    output: Optional[Path],
    is_train: bool,
) -> Config:
    """Set up ReNovo for training, datastore construction, and evaluation.

    Parameters
    ----------
    model : Optional[str]
        The provided model weights file.
    config : Optional[str]
        The provided configuration file.
    output : Optional[Path]
        The provided output file name.
    is_train : bool
        Whether to train. Building and evaluation require an explicit checkpoint.

    Return
    ------
    config : Config
        The parsed configuration
    """
    # Read parameters from the config file.
    config = Config(config)
    seed_everything(seed=config["random_seed"], workers=True)

    # Reproduction uses an explicitly selected ReNovo checkpoint.
    if model is None and not is_train:
        raise click.UsageError("Provide the ReNovo checkpoint with --model.")

    # Log the active configuration.
    logger.debug("model = %s", model)
    logger.debug("config = %s", config.file)
    logger.debug("output = %s", output)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    return config, model


##### 012
##### 012

if __name__ == "__main__":
    main()
