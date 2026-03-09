#!/usr/bin/env python
# coding: utf-8

#  Spinal Cord fMRI figures
# ____________________________________________________

# ### Project: acdc_spine_7T
# ____________________________________________________

# ------------------------------------------------------------------
# ------ Initialization
# ------------------------------------------------------------------
# Imports
import json, os, argparse

import pandas as pd
from figure import FigureTSNR, FigureEpiComparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs='+', default=[""])
    parser.add_argument("--tasks", nargs='+', default=[""])
    parser.add_argument("--verbose", default="False")
    parser.add_argument("--redo", default="True")
    parser.add_argument("--path-data", required=True)
    args = parser.parse_args()

    # get path of the parent location of this file, and go up one level
    path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fname_config = os.path.join(path_code, 'config', 'config_spine_7t_fmri.json')
    with open(fname_config) as config_file:
        config = json.load(config_file)  # load config file should be open first and the path inside modified

    IDs = args.ids
    tasks = args.tasks
    verbose = args.verbose.lower() == "true"
    redo = args.redo.lower() == "true"
    path_data = args.path_data

    config["raw_dir"] = path_data
    config["code_dir"] = path_code

    for task_available in config['design_exp']['task_names']:
        if task_available not in tasks and tasks != [""]:
            raise ValueError(f"All tasks need to be run to generate the figure")

    # Load participants info
    participants_tsv = pd.read_csv(os.path.join(path_code, 'config', 'participants.tsv'), sep='\t',
                                   dtype={'participant_id': str})

    new_IDs = []
    if IDs == [""]:
        for ID in participants_tsv["participant_id"]:
            new_IDs.append(ID)
        IDs = new_IDs

    if tasks != [""]:
        config["design_exp"]["task_names"] = tasks

    # ------------------------------------------------------------------
    # ------ Figures
    # ------------------------------------------------------------------
    print("=== Figure script Start ===", flush=True)
    print("Participant(s) included : ", IDs, flush=True)
    print("===================================", flush=True)
    print("")

    # TSNR figure
    fig_tsnr = FigureTSNR(config, IDs, redo)
    fig_tsnr.create_figure()

    # EPI comparison figure
    fig_epi_comparison = FigureEpiComparison(config, IDs, redo)
    fig_epi_comparison.create_figure(show_avg=False)

    print("=== Figure script End ===", flush=True)


if __name__ == "__main__":
    main()
