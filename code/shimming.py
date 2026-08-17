import json,sys, os, glob, re, argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import subprocess

from nibabel.processing import resample_from_to
from postprocess import pair_ttest

# Get the environment variable PATH_CODE
path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(path_code, "code"))
import figures

with open(os.path.join(path_code, "config", "config_spine_7t_fmri.json")) as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
    config = json.load(config_file) # load config file should be open first and the path inside modified

parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs='+', default=[""])
parser.add_argument("--tasks", nargs='+', default=[""])
parser.add_argument("--verbose", default="False")
parser.add_argument("--redo", default="True")
parser.add_argument("--path-data", required=True)
args = parser.parse_args()

IDs = args.ids
tasks = args.tasks
verbose = args.verbose.lower() == "true"
redo = args.redo.lower() == "true"
path_data = os.path.abspath(args.path_data)

config["raw_dir"]=path_data
config["code_dir"]=path_code
figs = figures.Figures_main(config, IDs=IDs)
participants_tsv = pd.read_csv(os.path.join(path_code, "config", "participants.tsv"), sep='\t',dtype={'participant_id': str})

new_IDs=[]
if IDs == [""]:
    for ID in participants_tsv["participant_id"]:
        new_IDs.append(ID)

    IDs = new_IDs

if tasks != [""]:
    config["design_exp"]["task_names"] = tasks

# Import scripts
sys.path.append(os.path.join(path_code, "code"))  # Change this line according to your directory


def main():
    print("Starting shimming analysis...")
    path_shimming = os.path.join(path_data, "derivatives", "processing", "shimming")
    path_figures = os.path.join(path_data, "derivatives", "processing", "figures")
    df = pd.DataFrame(columns=["ID", "Experiment", "rmse", "std"])

    exps = {
        "shimmed_volume_orders_012_linlsq": {
            "order": "0,1,2",
            "sig_loss": None,
            "opt_meth": "lin_lsq",
            "opt_cri": None,
            "slices": "volume"
        },
        "shimmed_volume_orders_012_pi": {
            "order": "0,1,2",
            "sig_loss": None,
            "opt_meth": "pseudo_inverse",
            "opt_cri": None,
            "slices": "volume"
        },
        "shimmed_volume_orders_0123_linlsq": {
            "order": "0,1,2,3",
            "sig_loss": None,
            "opt_meth": "lin_lsq",
            "opt_cri": None,
            "slices": "volume"
        },
        "shimmed_volume_orders_0123_pi": {
            "order": "0,1,2,3",
            "sig_loss": None,
            "opt_meth": "pseudo_inverse",
            "opt_cri": None,
            "slices": "volume"
        },
        "shimmed_slicewise_pi_sigloss": {
            "order": "0,1",
            "sig_loss": "0.1",
            "opt_meth": "pseudo_inverse",
            "opt_cri": None,
            "slices": "auto"
        },
        "shimmed_slicewise_pi": {
            "order": "0,1",
            "sig_loss": None,
            "opt_meth": "pseudo_inverse",
            "opt_cri": None,
            "slices": "auto"
        },
        "shimmed_slicewise_sigint_slsqp": {
            "order": "0,1",
            "sig_loss": "10",
            "opt_meth": "slsqp",
            "opt_cri": "rmse",
            "slices": "auto"
        },
        "shimmed_slicewise_nosigint_slsqp": {
            "order": "0,1",
            "sig_loss": None,
            "opt_meth": "slsqp",
            "opt_cri": "rmse",
            "slices": "auto"
        },
        "shimmed_slicewise_sigint_linlsq": {
            "order": "0,1",
            "sig_loss": "0.1",
            "opt_meth": "lin_lsq",
            "opt_cri": None,
            "slices": "auto"
        },
        "shimmed_slicewise_nosigint_linlsq": {
            "order": "0,1",
            "sig_loss": None,
            "opt_meth": "lin_lsq",
            "opt_cri": None,
            "slices": "auto"
        },
    }

    fname_metrics = os.path.join(path_shimming, "metrics.csv")
    if not os.path.exists(fname_metrics) or redo:
        if not os.path.exists(os.path.dirname(fname_metrics)):
            os.makedirs(os.path.dirname(fname_metrics), exist_ok=True)

        for ID in IDs:
            print("Processing ID:", ID)
            fname_seg = os.path.join(path_data, config["preprocess_dir"]["main_dir"].format(ID), config["preprocess_dir"]["anat_seg"], f"sub-{ID}_t2star_seg.nii.gz")
            fname_anat = os.path.join(path_data, f"sub-{ID}", "anat", f"sub-{ID}_T2star.nii.gz")
            fname_mask_25 = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_mask_25.nii.gz")
            fname_mask_40 = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_mask_40.nii.gz")

            if not os.path.exists(os.path.dirname(fname_mask_40)):
                os.makedirs(os.path.dirname(fname_mask_40), exist_ok=True)

            # Create fmap and shim masks
            if not os.path.exists(fname_mask_25) or redo:
                cmd = f"sct_create_mask -i {fname_anat} -p centerline,{fname_seg} -size 25 -o {fname_mask_25}"
                subprocess.run(cmd, shell=True, check=True)
            if not os.path.exists(fname_mask_40) or redo:
                cmd = f"sct_create_mask -i {fname_anat} -p centerline,{fname_seg} -size 40 -o {fname_mask_40}"
                subprocess.run(cmd, shell=True, check=True)

            fname_phase = os.path.join(path_data, f"sub-{ID}", "fmap", f"sub-{ID}_phasediff.nii.gz")
            fname_mag = os.path.join(path_data, f"sub-{ID}", "fmap", f"sub-{ID}_magnitude1.nii.gz")
            fname_fmap = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_fieldmap.nii.gz")

            if not os.path.exists(fname_fmap) or redo:
                cmd = ["st_prepare_fieldmap", fname_phase, "--mag", fname_mag, "--unwrapper", "prelude", "--gaussian-filter", "true", "--sigma", "1", "--mask", fname_mask_40, "-o", fname_fmap]
                subprocess.run(" ".join(cmd), shell=True, check=True)

            fname_target = glob.glob(os.path.join(path_data, f"sub-{ID}", "func", f"sub-{ID}_*.nii.gz"))[0]

            for exp in exps.keys():
                path_output = os.path.join(path_shimming, f"sub-{ID}", exp)

                if not os.path.exists(os.path.join(path_output, "fieldmap_calculated_shim.nii.gz")) or redo:
                    # Shim
                    cmd = ["st_b0shim dynamic",
                            "--scanner-coil-order", exps[exp]["order"],
                            "--fmap", fname_fmap,
                            "--target", fname_target,
                            "--mask", fname_mask_25,
                            "--mask-dilation-kernel-size", "5",
                            "--optimizer-method", exps[exp]["opt_meth"],
                            "--slices", exps[exp]["slices"],
                            # "--output-file-format-scanner", "slicewise-hrd",
                            "--output-value-format", "delta",
                            "--output", path_output]

                    if exps[exp]["sig_loss"] is not None:
                        cmd.append("--weighting-signal-loss")
                        cmd.append(exps[exp]["sig_loss"])
                    if exps[exp]["opt_cri"] is not None:
                        cmd.append("--optimizer-criteria")
                        cmd.append(exps[exp]["opt_cri"])

                    print("Running command: " + " ".join(cmd))
                    subprocess.run(" ".join(cmd), shell=True, check=True)


            # Extract data from orig fmap
            std, rmse = get_metrics_from_fmap(fname_fmap, fname_mask_25, fname_target)

            data = {
                "IDs": ID,
                "Experiment": "baseline",
                "std": std,
                "rmse": rmse,
            }
            df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

            # Create panda dataframe with different metrics
            for exp in exps.keys():
                fname_fmap_shimmed = os.path.join(path_shimming, f"sub-{ID}", exp, "fieldmap_calculated_shim.nii.gz")
                # Extract data from orig fmap
                std, rmse = get_metrics_from_fmap(fname_fmap_shimmed, fname_seg, fname_target)
                data = {
                    "IDs": ID,
                    "Experiment": exp,
                    "std": std,
                    "rmse": rmse,
                }
                df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

        # Save the dataframe to a csv file
        df.to_csv(fname_metrics, index=False)

    df = pd.read_csv(fname_metrics)
    metrics = {
        "std": "STD (Hz)",
        "rmse": "RMSE (Hz)"
    }
    plots = []
    ymax = {
        "std": 60,
        "rmse": 100
    }
    for metric in metrics.keys():
        # Plot boxcar plots for the different experiments across participants
        fig = plt.figure(figsize=(15, 5))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.1)

        plot_exp = ['baseline', 'shimmed_volume_orders_012_linlsq', 'shimmed_volume_orders_0123_linlsq', 'shimmed_slicewise_nosigint_linlsq']
        plot_labels = ['shimBase (orders 0, 1, 2)', 'shimVolume (orders 0, 1, 2)', 'shimVolume (orders 0, 1, 2, 3)', 'shimSlice (orders 0, 1)']
        df_sub = df[(df['Experiment'] == 'baseline') | (df['Experiment'] == 'shimmed_volume_orders_012_linlsq') | (df['Experiment'] == 'shimmed_volume_orders_0123_linlsq') | (df['Experiment'] == 'shimmed_slicewise_nosigint_linlsq')]

        fname_stats = os.path.join(path_shimming, f"stats_{metric}.csv")
        if os.path.exists(fname_stats):
            os.remove(fname_stats)
        comparisons = (("baseline", "shimmed_volume_orders_012_linlsq"),
                       ("shimmed_volume_orders_012_linlsq", "shimmed_volume_orders_0123_linlsq"),
                       ("shimmed_volume_orders_0123_linlsq", "shimmed_slicewise_nosigint_linlsq"))
        for comparison in comparisons:
            fname_newstats = os.path.join(path_shimming, f"{comparison[0]}_vs_{comparison[1]}_{metric}.csv")
            pair_ttest(df=df_sub,
                       output_fname=fname_newstats,
                       value_col=metric,
                       acq_col="Experiment",
                       cond1=comparison[0],
                       cond2=comparison[1])
            df_newstats = pd.read_csv(fname_newstats, index_col=0)
            if os.path.exists(fname_stats):
                df_stats = pd.read_csv(fname_stats, index_col=0)
                df_stats = pd.concat([df_stats, df_newstats], axis=0)
            else:
                df_stats = df_newstats
            df_stats.to_csv(fname_stats)

        for i, exp in enumerate(plot_exp):
            print(f"Average {metric}: {df[df['Experiment'] == exp][metric].mean()} for {plot_labels[i]}")

        plots.append(figs.boxplots(df=df_sub, output_fname=os.path.join(path_shimming, "boxplot_" + metric + ".png"),
                                   stats_file=fname_stats,
                                   stats_height_scaling=0.93,
                                  ymin=None, ymax=ymax[metric],
                                  specify_y_label=metrics[metric],
                         color=["#ADA8A8","#4A6B82", "#E5A93C", "#ED263F"],
                                  x_data="Experiment", x_order=plot_exp,
                                  indiv_values=False, x_labels=plot_labels,
                                  y_data=metric, redo=True, aspect=1 , height=3.7))

    figs.combine_plots(os.path.join(path_figures, "shim_boxplots.png"), plots,
                       figsize=(5, 3), redo=True)


def get_metrics_from_fmap(fname_fmap, fname_seg, fname_target):
    nii_fmap = nib.load(fname_fmap)
    nii_seg = nib.load(fname_seg)
    nii_target = nib.load(fname_target)
    nii_target_3d = nib.Nifti1Image(nii_target.get_fdata()[..., 0], nii_target.affine, header=nii_target.header)

    nii_seg_epi_space = resample_from_to(nii_seg, nii_target_3d)
    nii_seg_fmap_space = resample_from_to(nii_seg_epi_space, nii_fmap)

    ma_fmap = np.ma.array(nii_fmap.get_fdata(), mask=nii_seg_fmap_space.get_fdata() == False, fill_value=np.nan)
    std = np.ma.std(ma_fmap)
    rmse = np.ma.sqrt(np.ma.mean(ma_fmap ** 2))
    return std, rmse


if __name__ == "__main__":
    main()