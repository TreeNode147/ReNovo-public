# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Melih Yilmaz.
# Derived from Casanovo; modified for ReNovo experiment reproduction.
# See LICENSE and licenses/ at the repository root.

"""Small utility functions"""
import logging
import os
import platform

import psutil
import torch


logger = logging.getLogger("renovo")


def n_workers() -> int:
    """
    Get the number of workers to use for data loading.

    This is the maximum number of CPUs allowed for the process, scaled for the
    number of GPUs being used.

    On Windows and MacOS, we only use the main process. See:
    https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/4
    https://github.com/pytorch/pytorch/issues/70344

    Returns
    -------
    int
        The number of workers.
    """
    # Windows or MacOS: no multiprocessing.
    if platform.system() in ["Windows", "Darwin"]:
        logger.warning(
            "Dataloader multiprocessing is currently not supported on Windows "
            "or MacOS; using only a single thread."
        )
        return 0
    # Linux: scale the number of workers by the number of GPUs (if present).
    try:
        n_cpu = len(psutil.Process().cpu_affinity())
    except AttributeError:
        n_cpu = os.cpu_count()
    return (
        n_cpu // n_gpu if (n_gpu := torch.cuda.device_count()) > 1 else n_cpu
    )


