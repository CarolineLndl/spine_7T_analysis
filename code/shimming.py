import json,sys, os, glob, re, argparse
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import subprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

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
    mask_vert = False

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
        # "shimmed_volume_orders_012_pi": {
        #     "order": "0,1,2",
        #     "sig_loss": None,
        #     "opt_meth": "pseudo_inverse",
        #     "opt_cri": None,
        #     "slices": "volume"
        # },
        "shimmed_volume_orders_0123_linlsq": {
            "order": "0,1,2,3",
            "sig_loss": None,
            "opt_meth": "lin_lsq",
            "opt_cri": None,
            "slices": "volume"
        },
        # "shimmed_volume_orders_0123_pi": {
        #     "order": "0,1,2,3",
        #     "sig_loss": None,
        #     "opt_meth": "pseudo_inverse",
        #     "opt_cri": None,
        #     "slices": "volume"
        # },
        # "shimmed_slicewise_pi_sigloss": {
        #     "order": "0,1",
        #     "sig_loss": "0.1",
        #     "opt_meth": "pseudo_inverse",
        #     "opt_cri": None,
        #     "slices": "auto"
        # },
        # "shimmed_slicewise_pi": {
        #     "order": "0,1",
        #     "sig_loss": None,
        #     "opt_meth": "pseudo_inverse",
        #     "opt_cri": None,
        #     "slices": "auto"
        # },
        # "shimmed_slicewise_sigint_slsqp": {
        #     "order": "0,1",
        #     "sig_loss": "10",
        #     "opt_meth": "slsqp",
        #     "opt_cri": "rmse",
        #     "slices": "auto"
        # },
        # "shimmed_slicewise_nosigint_slsqp": {
        #     "order": "0,1",
        #     "sig_loss": None,
        #     "opt_meth": "slsqp",
        #     "opt_cri": "rmse",
        #     "slices": "auto"
        # },
        # "shimmed_slicewise_sigint_linlsq": {
        #     "order": "0,1",
        #     "sig_loss": "0.1",
        #     "opt_meth": "lin_lsq",
        #     "opt_cri": None,
        #     "slices": "auto"
        # },
        "shimmed_slicewise_nosigint_linlsq": {
            "order": "0,1",
            "sig_loss": None,
            "opt_meth": "lin_lsq",
            "opt_cri": None,
            "slices": "auto"
        },
    }

    #################################################################
    # Shim different scenarios and add metrics to metrics.csv
    #################################################################
    fname_metrics = os.path.join(path_shimming, "metrics.csv")
    if not os.path.exists(fname_metrics) or redo:
        if not os.path.exists(os.path.dirname(fname_metrics)):
            os.makedirs(os.path.dirname(fname_metrics), exist_ok=True)

        for ID in IDs:
            print("Processing ID:", ID)
            fname_seg = os.path.join(path_data, config["preprocess_dir"]["main_dir"].format(ID), config["preprocess_dir"]["anat_seg"], f"sub-{ID}_t2star_seg.nii.gz")
            fname_anat = os.path.join(path_data, f"sub-{ID}", "anat", f"sub-{ID}_T2star.nii.gz")
            fname_mask_25 = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_mask_25.nii.gz")
            fname_mask_100 = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_mask_100.nii.gz")

            if not os.path.exists(os.path.dirname(fname_mask_100)):
                os.makedirs(os.path.dirname(fname_mask_100), exist_ok=True)

            # Create fmap and shim masks
            if not os.path.exists(fname_mask_25) or redo:
                cmd = f"sct_create_mask -i {fname_anat} -p centerline,{fname_seg} -size 25 -o {fname_mask_25}"
                subprocess.run(cmd, shell=True, check=True)
            if not os.path.exists(fname_mask_100) or redo:
                cmd = f"sct_create_mask -i {fname_anat} -p centerline,{fname_seg} -size 100 -o {fname_mask_100}"
                subprocess.run(cmd, shell=True, check=True)

            # Use total spine seg segmentation
            fname_total_spine_seg = os.path.join(path_data, config["preprocess_dir"]["main_dir"].format(ID), "anat", "sct_deepseg_totalspineseg", f"sub-{ID}_T2star_totalspineseg_all.nii.gz")
            nii_totspineseg = nib.load(fname_total_spine_seg)
            mask = nii_totspineseg.get_fdata()
            mask[mask == 0] = 1
            mask[mask == 50] = 0
            mask[mask > 1] = 1
            fname_mask_no_vert = os.path.join(path_shimming, f"sub-{ID}", f"mask_totspineseg.nii.gz")
            nib.Nifti1Image(mask, nii_totspineseg.affine, header=nii_totspineseg.header).to_filename(fname_mask_no_vert)

            if mask_vert:
                # Remove to show with verta
                cmd = f"st_image logical-and {fname_mask_no_vert} {fname_mask_25} -o {fname_mask_25}"
                subprocess.run(cmd, shell=True, check=True)
                cmd = f"st_image logical-and {fname_mask_no_vert} {fname_mask_100} -o {fname_mask_100}"
                subprocess.run(cmd, shell=True, check=True)

            fname_phase = os.path.join(path_data, f"sub-{ID}", "fmap", f"sub-{ID}_phasediff.nii.gz")
            fname_mag = os.path.join(path_data, f"sub-{ID}", "fmap", f"sub-{ID}_magnitude1.nii.gz")
            fname_fmap = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_fieldmap.nii.gz")

            if not os.path.exists(fname_fmap) or redo:
                cmd = ["st_prepare_fieldmap", fname_phase, "--mag", fname_mag, "--unwrapper", "prelude", "--gaussian-filter", "true", "--sigma", "1", "--mask", fname_mask_100, "-o", fname_fmap]
                subprocess.run(" ".join(cmd), shell=True, check=True)

            fname_target = glob.glob(os.path.join(path_data, f"sub-{ID}", "func", f"sub-{ID}_*.nii.gz"))[0]

            for exp in exps.keys():
                path_output = os.path.join(path_shimming, f"sub-{ID}", exp)

                if not os.path.exists(os.path.join(path_output, "fieldmap_calculated_shim_masked.nii.gz")) or redo:
                    # Shim
                    cmd = ["st_b0shim dynamic",
                            "--scanner-coil-order", exps[exp]["order"],
                            "--fmap", fname_fmap,
                            "--target", fname_target,
                            "--mask", fname_mask_25,
                            "--mask-dilation-kernel-size", "5",
                            "--optimizer-method", exps[exp]["opt_meth"],
                            "--slices", exps[exp]["slices"],
                            "--output-file-format-scanner", "slicewise-coil",
                            "--output-value-format", "delta",
                            "--output", path_output,
                            "--verbose", "debug"]

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
                fname_fmap_shimmed = os.path.join(path_shimming, f"sub-{ID}", exp, "fieldmap_calculated_shim_masked.nii.gz")
                # Extract data from orig fmap
                std, rmse = get_metrics_from_fmap(fname_fmap_shimmed, fname_mask_25, fname_target)
                data = {
                    "IDs": ID,
                    "Experiment": exp,
                    "std": std,
                    "rmse": rmse,
                }
                df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

        # Save the dataframe to a csv file
        df.to_csv(fname_metrics, index=False)

    plot_exp = ['baseline',
                'shimmed_volume_orders_012_linlsq',
                'shimmed_volume_orders_0123_linlsq',
                'shimmed_slicewise_nosigint_linlsq']
    plot_labels = ['shimBase (orders 0, 1, 2)',
                   'shimVolume (orders 0, 1, 2)',
                   'shimVolume (orders 0, 1, 2, 3)',
                   'shimSlice (orders 0, 1)']

    #################################################################
    # Show representative participant before and after simulated shim
    #################################################################
    # Show everything in the field map space
    # for ID in IDs:
    for ID in ["099",]:
        fname_anat = os.path.join(path_data, f"sub-{ID}", "anat", f"sub-{ID}_T2star.nii.gz")
        fname_mask_25 = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_mask_25.nii.gz")
        fname_mask_100 = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_mask_100.nii.gz")
        fname_fmap = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_fieldmap.nii.gz")
        fname_target = glob.glob(os.path.join(path_data, f"sub-{ID}", "func", f"sub-{ID}_*.nii.gz"))[0]

        nii_anat = nib.load(fname_anat)
        nii_mask_25 = nib.load(fname_mask_25)
        nii_mask_100 = nib.load(fname_mask_100)
        nii_fmap = nib.load(fname_fmap)
        nii_target = nib.load(fname_target)
        nii_target_3d = nib.Nifti1Image(nii_target.get_fdata()[..., 0], nii_target.affine, header=nii_target.header)

        nii_seg_epi_space = resample_from_to(nii_mask_25, nii_target_3d)
        nii_seg_fmap_space = resample_from_to(nii_seg_epi_space, nii_fmap)
        nii_seg100_epi_space = resample_from_to(nii_mask_100, nii_target_3d)
        nii_seg100_fmap_space = resample_from_to(nii_seg100_epi_space, nii_fmap)
        nii_anat_fmap_space = resample_from_to(nii_anat, nii_fmap)
        data_fmap = nii_fmap.get_fdata()
        slices_to_show = []

        x_min, x_max, y_min, y_max, z_min, z_max = get_bounds_to_zoom_in(nii_seg_fmap_space.get_fdata(), [5, 5, 2])
        z_min += 1
        y_min += 1
        baseline_fmap_zoomed = data_fmap[x_min:x_max, y_min:y_max, z_min:z_max]
        slices_to_show.append(baseline_fmap_zoomed[baseline_fmap_zoomed.shape[0] // 2, :, :])
        x_mina, x_maxa, y_mina, y_maxa, z_mina, z_maxa = get_bounds_to_zoom_in(nii_seg_fmap_space.get_fdata(), [5, 5, 2])
        z_mina += 1
        y_mina += 1
        anat_zoomed = nii_anat_fmap_space.get_fdata()[x_mina:x_maxa, y_mina:y_maxa, z_mina:z_maxa]
        anat_sag_slice = anat_zoomed[anat_zoomed.shape[0] // 2, :, :]

        seg = nii_seg_fmap_space.get_fdata()
        seg_zoomed = seg[x_min:x_max, y_min:y_max, z_min:z_max]
        seg_sag_slice = seg_zoomed[seg_zoomed.shape[0] // 2, :, :]
        seg_zoomed_anat = seg[x_mina:x_maxa, y_mina:y_maxa, z_mina:z_maxa]
        seg_sag_slice_anat = seg_zoomed_anat[anat_zoomed.shape[0] // 2, :, :]

        # Plot all 4 scenarios (baseline, vol012, vol0123, slice01)
        for i in range(1, 4):
            fname_fmap_shimmed = os.path.join(path_shimming, f"sub-{ID}", plot_exp[i], "fieldmap_calculated_shim_not_masked.nii.gz")
            nii_fmap_shimmed = nib.load(fname_fmap_shimmed)
            if mask_vert:
                # Multiply by mask to remove vertebrae
                data_fmap_shimmed = nii_fmap_shimmed.get_fdata() * nii_seg100_fmap_space.get_fdata()
            else:
                data_fmap_shimmed = nii_fmap_shimmed.get_fdata()
            shimmed_fmap_zoomed = data_fmap_shimmed[x_min:x_max, y_min:y_max, z_min:z_max]
            slices_to_show.append(shimmed_fmap_zoomed[shimmed_fmap_zoomed.shape[0] // 2, :, :])

        # vmin = min(a_slice.min() for a_slice in slices_to_show)
        # vmax = max(a_slice.max() for a_slice in slices_to_show)
        vmin, vmax = (-100, 100)
        print(f"vmin: {vmin}, vmax: {vmax}")
        fig = plt.figure(figsize=(8, 4))
        fontsize = 7
        width_ratios = [1.0 * anat_sag_slice.shape[0] / slices_to_show[0].shape[0 ], 0.05, 1.0, 0.05, 1.0, 0.05, 1.0, 0.05, 1.0875]
        gs = gridspec.GridSpec(nrows=1, ncols=9, width_ratios=width_ratios, figure=fig, hspace=0, wspace=0)
        for i in range(0, 5):
            ax = fig.add_subplot(gs[0, 2*i])
            if i == 0:
                delta = anat_sag_slice.max() - anat_sag_slice.min()
                if ID == "099":
                    vmin_anat = -26
                    vmax_anat = 171
                else:
                    vmin_anat = anat_sag_slice.min()
                    vmax_anat = anat_sag_slice.max() - (0.2 * delta)
                ax.imshow(np.rot90(anat_sag_slice, k=1), cmap='gray', vmin=vmin_anat, vmax=vmax_anat)
                ax.contour(np.rot90(seg_sag_slice_anat, k=1), levels=[0.5], colors="red", linewidths=1.5)
                ax.set_title("Anatomical", fontsize=fontsize-1, weight='bold')
                if ID == "099":
                    # Add top and bottom vertebrae levels on the anatomical image
                    ax.text(0.1, 0.97, "C3", color='white', fontsize=fontsize-1, weight='bold', ha='center', va='top', transform=ax.transAxes)
                    ax.text(0.1, 0.03, "T1", color='white', fontsize=fontsize-1, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
            else:
                im = ax.imshow(np.rot90(slices_to_show[i - 1], k=1), cmap='jet', vmin=vmin, vmax=vmax)
                ax.contour(np.rot90(seg_sag_slice, k=1), levels=[0.5], colors="red", linewidths=1.5)
                ax.set_title(plot_labels[i - 1], fontsize=fontsize-1, weight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        # Create a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label('Hz', fontsize=fontsize, loc= 'center', rotation=0)
        cbar.ax.locator_params(nbins=5)
        cbar.update_ticks()
        plt.tight_layout()
        fig.tight_layout()
        fname_fmap_comparison = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_fmap_comparison.png")
        fig.savefig(fname_fmap_comparison, dpi=600)
        if ID == "099":
            shutil.copy(fname_fmap_comparison, os.path.join(path_figures, f"fmap_comparison.png"))


    #################################################################
    # Boxplot of different scenarios
    #################################################################
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

    #################################################################
    # Simulate movement
    #################################################################
    movements = []
    for x in np.linspace(-20, 20, 5):
        for y in np.linspace(-20, 20, 5):
            for z in np.linspace(-20, 20, 5):
                movements.append((x, y, z))

    os.makedirs(os.path.join(path_shimming, "movement"), exist_ok=True)

    # Rotations?
    fname_dfmov = os.path.join(path_shimming, "movement", "dfmov.csv")
    if os.path.exists(fname_dfmov):
        df_mov = pd.read_csv(fname_dfmov)
    else:
        df_mov = pd.DataFrame(columns=["ID", "Experiment", "rmse", "std", "movement_x", "movement_y", "movement_z"])

    for movement in movements:
        for ID in IDs:
            os.makedirs(os.path.join(path_shimming, "movement", f"sub-{ID}"), exist_ok=True)
            fname_fmap = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_fieldmap.nii.gz")
            fname_target = glob.glob(os.path.join(path_data, f"sub-{ID}", "func", f"sub-{ID}_*.nii.gz"))[0]
            fname_mask_25 = os.path.join(path_shimming, f"sub-{ID}", f"sub-{ID}_mask_25.nii.gz")
            fname_mask_25_moved = os.path.join(path_shimming, "movement", f"sub-{ID}", f"sub-{ID}_mask_25_moved.nii.gz")

            nii_fmap = nib.load(fname_fmap)
            nii_fmap_moved = apply_movement(nii_fmap, movement)
            fname_fmap_moved = os.path.join(path_shimming, "movement", f"sub-{ID}", f"sub-{ID}_fieldmap_moved.nii.gz")
            nii_fmap_moved.to_filename(fname_fmap_moved)
            shutil.copy(fname_fmap.rsplit(".nii.gz")[0] + ".json", fname_fmap_moved.rsplit(".nii.gz")[0] + ".json")

            nii_mask_25 = nib.load(fname_mask_25)
            nii_mask_25_moved = apply_movement(nii_mask_25, movement)
            nib.save(nii_mask_25_moved, fname_mask_25_moved)

            for exp in exps.keys():
                # Look if an entry in df_mov is there
                n_entries = ((df_mov['Experiment'] == exp) & (df_mov['ID'] == int(ID)) & (df_mov['movement_x'] == movement[0]) & (df_mov['movement_y'] == movement[1]) & (df_mov['movement_z'] == movement[2])).sum()
                if n_entries == 1:
                    continue
                elif n_entries > 1:
                    # Delete all rows except 1 that match the criteria
                    print(f"Deleted {n_entries - 1} entries for ID {ID}, Experiment {exp}, Movement {movement}")
                    df_mov = df_mov[~((df_mov['Experiment'] == exp) & (df_mov['ID'] == int(ID)) & (df_mov['movement_x'] == movement[0]) & (df_mov['movement_y'] == movement[1]) & (df_mov['movement_z'] == movement[2])) | (df_mov.index == df_mov[((df_mov['Experiment'] == exp) & (df_mov['ID'] == int(ID)) & (df_mov['movement_x'] == movement[0]) & (df_mov['movement_y'] == movement[1]) & (df_mov['movement_z'] == movement[2]))].index[0])]
                    continue
                path_output = os.path.join(path_shimming, "movement", f"sub-{ID}", exp)
                fname_coefs_no_movement = os.path.join(path_shimming, f"sub-{ID}", exp, "coefs_coil0_Investigational_Device_7T_79017.txt")
                # st_b0shim using fixed coef option
                channels_per_order = {0: 1, 1: 3, 2: 5, 3: 4}
                i = 0
                off_channels = []
                for order in exps[exp]["order"].split(','):
                    for _ in range(channels_per_order[int(order)]):
                        off_channels.append(str(i))
                        i += 1

                off_channels = ",".join(off_channels)
                # Shim
                # All channels are off, this is a trick to apply the shim coefficients without optimizing
                cmd = ["st_b0shim dynamic",
                       "--scanner-coil-order", exps[exp]["order"],
                       "--fmap", fname_fmap_moved,
                       "--target", fname_target,
                       "--mask", fname_mask_25,
                       "--mask-dilation-kernel-size", "5",
                       "--optimizer-method", exps[exp]["opt_meth"],
                       "--slices", exps[exp]["slices"],
                       "--output-file-format-scanner", "slicewise-coil",
                       "--output-value-format", "delta",
                       "--off-channels", off_channels,
                       "--off-channels-values", fname_coefs_no_movement,
                       "--verbose", "debug",  # To output not masked calculated shim
                       "--output", path_output]

                if exps[exp]["sig_loss"] is not None:
                    cmd.append("--weighting-signal-loss")
                    cmd.append(exps[exp]["sig_loss"])
                if exps[exp]["opt_cri"] is not None:
                    cmd.append("--optimizer-criteria")
                    cmd.append(exps[exp]["opt_cri"])

                print("Running command: " + " ".join(cmd))
                subprocess.run(" ".join(cmd), shell=True, check=True)
                fname_fmap_shimmed = os.path.join(path_output, "fieldmap_calculated_shim_not_masked.nii.gz")
                # Compute rmse and std in SC
                std, rmse = get_metrics_from_fmap(fname_fmap_shimmed, fname_mask_25_moved, fname_target)

                # Add to df
                stats = {
                    "ID": ID,
                    "Experiment": exp,
                    "rmse": rmse,
                    "std": std,
                    "movement_x": movement[0], "movement_y": movement[1], "movement_z": movement[2]
                }
                df_mov = pd.concat([df_mov, pd.DataFrame([stats])], ignore_index=True)
            df_mov.to_csv(fname_dfmov, index=False)

    # Once we have all the data, compute average rmse and std for in_plane and out of plane movement
    for metric in metrics:
        # Compute in-plane movement
        df_mov["in_plane_movement"] = np.sqrt(df_mov["movement_x"]**2 + df_mov["movement_y"]**2)
        df_mov["out_of_plane_movement"] = df_mov["movement_z"]
        df_mov["total_movement"] = np.sqrt(df_mov["movement_x"]**2 + df_mov["movement_y"]**2 + df_mov["movement_z"]**2)

        for movement_type in ["in_plane_movement", "out_of_plane_movement", "total_movement"]:
            data_movement = df_mov.groupby(["Experiment", movement_type])[metric].mean()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            for exp in exps.keys():
                ax.plot(data_movement[exp].index.values, data_movement[exp].values)

            # Add as reference hlines of the mean value obtained when no movement
            # Maybe not necessary, since we will see it show up on the graph (we expect some sort of quadratic)

            ax.set_xlabel(movement_type)
            ax.set_ylabel(metric)
            ax.set_title(f"Movement {metric}")
            fig.savefig(os.path.join(path_shimming, "movement", f"{movement_type}_{metric}.png"))

    # debug
    # import pandas as pd
    # import numpy as np
    # df_mov = pd.read_csv("/Users/alexandredastous/Documents/School/Polytechnique/Master/project/spine_7T/spine_7t_fmri_data/derivatives/processing/shimming/movement/dfmov.csv")

def apply_movement(nii, movement):
    new_affine = nii.affine
    new_affine[:3, 3] += movement
    return nib.Nifti1Image(nii.get_fdata(), new_affine)


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


def get_bounds_to_zoom_in(mask, margins):
    """ Get the x and y bounds that would allow to zoom in on the valid voxels in the mask, with a margin around them.
     The bounds are the same for all slices.

    Args:
        mask (np.array): 3d array filled from 0 to 1.
        margins (tuple): Number of voxels to add around the valid voxels in each dimension (x, y, z).

    Returns:
        tuple: x_min, x_max, y_min, y_max, z_min, z_max
    """
    valid_voxels = np.any(mask != 0, axis=2)
    if np.any(valid_voxels):
        x_idx, y_idx = np.where(valid_voxels)
        x_min, x_max = max(x_idx.min() - margins[0], 0), min(x_idx.max() + margins[0], mask.shape[0] - 1)
        y_min, y_max = max(y_idx.min() - margins[1], 0), min(y_idx.max() + margins[1], mask.shape[1] - 1)
    else:
        x_min, x_max, y_min, y_max = 0, mask.shape[0], 0, mask.shape[1]

    valid_voxelsz = np.any(mask != 0, axis=0)
    if np.any(valid_voxelsz):
        _, z_idx = np.where(valid_voxelsz)
        z_min, z_max = max(z_idx.min() - margins[2], 0), min(z_idx.max() + margins[2], mask.shape[2] - 1)
    else:
        z_min, z_max = 0, mask.shape[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


if __name__ == "__main__":
    main()