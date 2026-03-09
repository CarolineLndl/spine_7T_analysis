#!/usr/bin/env python
# coding: utf-8

#  Spinal Cord fMRI figures
# ____________________________________________________

# ### Project: acdc_spine_7T
# ____________________________________________________

import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Arrow
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import nibabel as nib
from nilearn.image import smooth_img
import numpy as np
import os
import pandas as pd
from scipy.ndimage import center_of_mass
from utils import compute_tsnr_map, extract_mean_within_mask
import warnings

path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FigureTSNR:
    # ------------------------------------------------------------------
    # ------ Compute tSNR
    # ------------------------------------------------------------------

    # On tSNR map in PAM50 space : sub-{}_task-{}_acq-{}_bold_moco_mean_coreg_in_PAM50
    # On tSNR map in Original space : sub-{}_task-{}_acq-{}_bold_moco
    # Todo: Use nn for moco
    # Use the run with the most volumes
    # Use the same number of volumes for each tsnr calculation
    # ------------------------------------------------------------------

    def __init__(self, config, IDs, redo):
        self.IDs = IDs
        self.config = config
        self.redo = redo

        self.path_main_fig = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"])
        self.path_fig_tsnr = os.path.join(self.path_main_fig, "tsnr")
        self.path_fig_data = os.path.join(self.path_fig_tsnr, "data")
        self.fname_tsnr_baseline_avg = os.path.join(self.path_fig_tsnr, "data", "tsnr_baseline_avg_in_PAM50.nii.gz")
        self.fname_tsnr_slicewise_avg = os.path.join(self.path_fig_tsnr, "data", "tsnr_slicewise_avg_in_PAM50.nii.gz")

        self.fname_tsnr_metrics = os.path.join(self.path_fig_tsnr, "tsnr_metrics.csv")

    def _generate_tsnr_maps_and_csv(self):
        df_tsnr = pd.DataFrame(columns=["ID", "task", "acq", "tsnr_mean"])

        print("=== Compute tSNR map on longest moco neighbour run ===", flush=True)
        # Find the minimum number of volumes across all runs to standardize tSNR calculation
        min_vols_for_tsnr = 1000
        for ID in self.IDs:
            for task in self.config["design_exp"]["task_names"]:
                for acq_name in self.config["design_exp"]["acq_names"]:
                    selected_file = find_moco_for_tsnr_calculation(self.config, ID, task, acq_name)
                    if selected_file is None:
                        continue
                    n_vols = nib.load(selected_file).shape[3]
                    if n_vols < min_vols_for_tsnr:
                        min_vols_for_tsnr = n_vols

        print(f"Minimum number of volumes across all runs: {min_vols_for_tsnr}", flush=True)
        # Minimum number of volumes across all runs: 30 (2026-01-28)
        # Compute_tsnr
        for ID in self.IDs:
            for task in self.config["design_exp"]["task_names"]:
                for acq_name in self.config["design_exp"]["acq_names"]:
                    tag = "task-" + task + "_acq-" + acq_name

                    selected_file = find_moco_for_tsnr_calculation(self.config, ID, task, acq_name)
                    if selected_file is None:
                        continue

                    # Compute tSNR map in native space
                    path_tsnr_sub_folder = os.path.join(self.path_fig_data, f"sub-{ID}", tag)
                    fname_tsnr = compute_tsnr_map(selected_file, path_tsnr_sub_folder, self.redo, min_vols_for_tsnr)

                    # Warp tSNR in PAM50 space
                    fname_tsnr_in_template = fname_tsnr.replace("_bold_moco_tSNR.nii.gz",
                                                                "_bold_moco_tsnr_in_PAM50.nii.gz")
                    if not os.path.exists(fname_tsnr_in_template) or self.redo:
                        print("=== Warp tSNR map to PAM50 space ===", flush=True)

                        fname_warp_from_func_to_template = os.path.join(
                            self.config["raw_dir"],
                            self.config["preprocess_dir"]["main_dir"].format(ID),
                            "func",
                            tag,
                            f"sub-{ID}_{tag}_from-func_to_PAM50_mode-image_xfm.nii.gz")

                        if not os.path.exists(fname_warp_from_func_to_template):
                            raise RuntimeError(f"Warp file not found: {fname_warp_from_func_to_template}")

                        fname_template = os.path.join(self.config["code_dir"], "template", self.config["PAM50_t2"])
                        cmd_coreg = f"sct_apply_transfo -i {fname_tsnr} -d {fname_template} -w {fname_warp_from_func_to_template} -o {fname_tsnr_in_template} -x nn"
                        os.system(cmd_coreg)

                    # Extract metrics from native space
                    if fname_tsnr is not None:
                        fname_mask = os.path.join(
                            self.config["raw_dir"],
                            self.config["preprocess_dir"]["main_dir"].format(ID),
                            "func",
                            tag,
                            f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz")

                        if not os.path.exists(fname_mask):
                            raise RuntimeError(f"Mask file not found: {fname_mask}")

                        tsnr_mean = extract_mean_within_mask(fname_tsnr, fname_mask)
                        if len(df_tsnr) == 0:
                            df_tsnr = pd.DataFrame([[ID, task, acq_name, tsnr_mean]], columns=df_tsnr.columns)
                        df_tsnr = pd.concat(
                            [pd.DataFrame([[ID, task, acq_name, tsnr_mean]], columns=df_tsnr.columns), df_tsnr],
                            ignore_index=True)

        df_tsnr.to_csv(self.fname_tsnr_metrics, index=False)

    def _extract_baseline_and_slicewise_tsnr_from_csv(self):
        name_baseline = [a for a in self.config["design_exp"]["acq_names"] if a.find("Base") != -1][0]
        name_slicewise = [a for a in self.config["design_exp"]["acq_names"] if a.find("Slice") != -1][0]
        df_tsnr = pd.read_csv(self.fname_tsnr_metrics)
        list_baseline_tsnr = []
        list_slicewise_tsnr = []
        for ID in self.IDs:
            df_sub = df_tsnr[df_tsnr["ID"] == int(ID)]
            done = False
            # Try rest task
            if len(df_sub[df_sub["task"] == "rest"]) >= 2:
                done = True
                df_task = df_sub[df_sub["task"] == "rest"]
                if len(df_task) != 2:
                    raise RuntimeError(f"We don't have 2 tSNR metric for sub-{ID} task-rest")

                tsnr_baseline = df_task[df_task["acq"] == name_baseline]["tsnr_mean"].values
                tsnr_slicewise = df_task[df_task["acq"] == name_slicewise]["tsnr_mean"].values
                list_baseline_tsnr.append(tsnr_baseline[0])
                list_slicewise_tsnr.append(tsnr_slicewise[0])

            # If rest task not found, use motor task
            if not done:
                # Todo: If no rest task, use the motor task, we could use the volumes at rest during the motor task
                print(f"No rest task found for sub-{ID}, using motor task instead", flush=True)
                df_task = df_sub[df_sub["task"] == "motor"]
                if len(df_task) != 2:
                    warnings.warn(f"We don't have 2 tSNR metric for sub-{ID} task-motor")
                    continue

                tsnr_baseline = df_task[df_task["acq"] == name_baseline]["tsnr_mean"].values
                tsnr_slicewise = df_task[df_task["acq"] == name_slicewise]["tsnr_mean"].values
                list_baseline_tsnr.append(tsnr_baseline[0])
                list_slicewise_tsnr.append(tsnr_slicewise[0])

        return list_baseline_tsnr, list_slicewise_tsnr

    def create_figure(self):
        print("=== Generate tSNR violin plot figure ===", flush=True)

        self._generate_tsnr_maps_and_csv()

        # Create a figure
        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs_main = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[0.08, 1, 1])

        list_baseline_tsnr, list_slicewise_tsnr = self._extract_baseline_and_slicewise_tsnr_from_csv()
        create_tsnr_violin_plot(fig, gs_main[2], list_baseline_tsnr, list_slicewise_tsnr)

        self._generate_average_tsnr_in_pam50()

        nii_tsnr_baseline_avg = nib.load(self.fname_tsnr_baseline_avg)
        nii_tsnr_slicewise_avg = nib.load(self.fname_tsnr_slicewise_avg)

        fname_tsnr_baseline_avg_smooth = os.path.join(self.path_fig_tsnr, "data",
                                                      "tsnr_baseline_avg_smooth_in_PAM50.nii.gz")
        nii_tsnr_baseline_avg_smooth = smooth_img(nii_tsnr_baseline_avg, fwhm=[2, 2, 4])
        nib.save(nii_tsnr_baseline_avg_smooth, fname_tsnr_baseline_avg_smooth)

        fname_tsnr_slicewise_avg_smooth = os.path.join(self.path_fig_tsnr, "data",
                                                       "tsnr_slicewise_avg_smooth_in_PAM50.nii.gz")
        nii_tsnr_slicewise_avg_smooth = smooth_img(nii_tsnr_slicewise_avg, fwhm=[2, 2, 4])
        nib.save(nii_tsnr_slicewise_avg_smooth, fname_tsnr_slicewise_avg_smooth)

        nii_tsnr_baseline_avg_smooth = nib.load(fname_tsnr_baseline_avg_smooth)
        nii_tsnr_slicewise_avg_smooth = nib.load(fname_tsnr_slicewise_avg_smooth)

        fname_template_seg = os.path.join(path_code, 'template', self.config["PAM50_cord"])
        fname_template_seg_dil = os.path.join(self.path_fig_tsnr, "data", "template_cord_dil.nii.gz")
        if not os.path.exists(fname_template_seg_dil) or self.redo:
            cmd = f"sct_maths -i {fname_template_seg} -dilate 1 -o {fname_template_seg_dil}"
            os.system(cmd)
        fname_template_seg_outline = os.path.join(self.path_fig_tsnr, "data", "template_cord_outline.nii.gz")
        if not os.path.exists(fname_template_seg_outline) or self.redo:
            cmd = f"sct_maths -i {fname_template_seg_dil} -sub {fname_template_seg} -o {fname_template_seg_outline}"
            os.system(cmd)
        nii_outline = nib.load(fname_template_seg_outline)

        fname_template = os.path.join(self.config["code_dir"], "template", self.config["PAM50_t2"])
        nii_template = nib.load(fname_template)
        create_tsnr_template_plot(fig, gs_main[1], nii_tsnr_baseline_avg_smooth, nii_tsnr_slicewise_avg_smooth,
                                  nii_template, nii_outline)
        fig.suptitle(f"A) Average tSNR map (n={len(self.IDs)})              B) tSNR accross participants", fontsize=12,
                     fontweight='bold',
                     x=0.52)
        fname_output = os.path.join(self.path_fig_tsnr, "tsnr_plot.png")
        fig.savefig(fname_output, dpi=1000)

        print("=== Figure EPI comparison ===", flush=True)

    def _generate_average_tsnr_in_pam50(self):
        print("=== Generate tSNR in PAM50 figure ===", flush=True)

        fname_template = os.path.join(self.config["code_dir"], "template", self.config["PAM50_t2"])
        nii_template = nib.load(fname_template)
        data_tsnr_baseline = np.zeros_like(nii_template.get_fdata(), dtype=float)
        data_tsnr_slicewise = np.zeros_like(nii_template.get_fdata(), dtype=float)
        data_count_subjects_baseline = None
        data_count_subjects_slicewise = None

        fname_tsnr_baseline_avg = os.path.join(self.path_fig_tsnr, "data", "tsnr_baseline_avg_in_PAM50.nii.gz")
        fname_tsnr_slicewise_avg = os.path.join(self.path_fig_tsnr, "data", "tsnr_slicewise_avg_in_PAM50.nii.gz")

        name_baseline = [a for a in self.config["design_exp"]["acq_names"] if a.find("Base") != -1][0]
        name_slicewise = [a for a in self.config["design_exp"]["acq_names"] if a.find("Slice") != -1][0]

        for ID in self.IDs:
            path_task_baseline = os.path.join(
                self.path_fig_data,
                f"sub-{ID}",
                f"task-rest_acq-{name_baseline}")

            if os.path.exists(path_task_baseline):
                task = 'rest'
            else:
                task = 'motor'
                path_task_baseline = os.path.join(
                    self.path_fig_data,
                    f"sub-{ID}",
                    f"task-motor_acq-{name_baseline}")

            fname_tsnr_in_template_baseline = os.path.join(
                path_task_baseline,
                f"sub-{ID}_task-{task}_acq-{name_baseline}*_bold_moco_tsnr_in_PAM50.nii.gz"
            )
            if len(glob.glob(fname_tsnr_in_template_baseline)) != 1:
                raise RuntimeError(
                    f"0 or more than 1 tSNR in template files found: {glob.glob(fname_tsnr_in_template_baseline)}")
            fname_tsnr_in_template_baseline = glob.glob(fname_tsnr_in_template_baseline)[0]

            path_task_slicewise = os.path.join(
                self.path_fig_data,
                f"sub-{ID}",
                f"task-{task}_acq-{name_slicewise}")
            fname_tsnr_in_template_slicewise = os.path.join(
                path_task_slicewise,
                f"sub-{ID}_task-{task}_acq-{name_slicewise}*_bold_moco_tsnr_in_PAM50.nii.gz"
            )
            if len(glob.glob(fname_tsnr_in_template_slicewise)) != 1:
                raise RuntimeError(
                    f"0 or more than 1 tSNR in template files found: {glob.glob(fname_tsnr_in_template_slicewise)}")
            fname_tsnr_in_template_slicewise = glob.glob(fname_tsnr_in_template_slicewise)[0]

            fname_seg_b, fname_warp_from_func_to_template, _ = get_fname_seg_and_warps(ID, task, name_baseline, self.config)
            nii_roi_baseline = count_roi_in_template(os.path.join(self.path_fig_data, f"sub-{ID}", f"task-{task}_acq-{name_baseline}"),
                                                     ID, task, name_baseline,
                                                     fname_seg_b, fname_warp_from_func_to_template, fname_template,
                                                     self.redo)
            fname_seg_s, fname_warp_from_func_to_template, _ = get_fname_seg_and_warps(ID, task, name_slicewise, self.config)
            nii_roi_slicewise = count_roi_in_template(os.path.join(self.path_fig_data, f"sub-{ID}", f"task-{task}_acq-{name_slicewise}"),
                                                     ID, task, name_slicewise,
                                                     fname_seg_s, fname_warp_from_func_to_template, fname_template,
                                                     self.redo)

            nii_baseline = nib.load(fname_tsnr_in_template_baseline)
            data_tsnr_baseline += nii_baseline.get_fdata()
            nii_slicewise = nib.load(fname_tsnr_in_template_slicewise)
            data_tsnr_slicewise += nii_slicewise.get_fdata()
            if data_count_subjects_baseline is None:
                data_count_subjects_baseline = nii_roi_baseline.get_fdata()
                data_count_subjects_slicewise = nii_roi_slicewise.get_fdata()
            else:
                data_count_subjects_baseline += nii_roi_baseline.get_fdata()
                data_count_subjects_slicewise += nii_roi_slicewise.get_fdata()

        # Average
        data_tsnr_baseline_avg = np.divide(data_tsnr_baseline, data_count_subjects_baseline, out=np.zeros_like(data_tsnr_baseline), where=data_count_subjects_baseline != 0)
        data_tsnr_slicewise_avg = np.divide(data_tsnr_slicewise, data_count_subjects_slicewise, out=np.zeros_like(data_tsnr_slicewise), where=data_count_subjects_slicewise != 0)

        nii_tsnr_baseline_avg = nib.Nifti1Image(data_tsnr_baseline_avg, affine=nii_baseline.affine,
                                                header=nii_baseline.header)
        nib.save(nii_tsnr_baseline_avg, fname_tsnr_baseline_avg)

        nii_tsnr_slicewise_avg = nib.Nifti1Image(data_tsnr_slicewise_avg, affine=nii_slicewise.affine,
                                                 header=nii_slicewise.header)
        nib.save(nii_tsnr_slicewise_avg, fname_tsnr_slicewise_avg)


def create_tsnr_template_plot(fig, gs, nii_baseline, nii_slicewise, nii_template, nii_cord_outline):
    axial_slice = 230
    axial_left_bound = 47
    axial_right_bound = 95
    axial_bot_bound = 53
    axial_top_bound = 89

    axial_outline = nii_cord_outline.get_fdata()[axial_left_bound:axial_right_bound, axial_bot_bound:axial_top_bound,
                    axial_slice]
    axial_cord_outline_nan = np.full_like(axial_outline, np.nan)
    axial_cord_outline_nan[axial_outline > 0.5] = 1

    cor_slice = 72
    cor_left_bound = 47
    cor_right_bound = 95
    cor_bot_bound = 158
    cor_top_bound = 322
    cor_templ_bot = 125
    cor_templ_top = 360
    hline = cor_templ_top - axial_slice

    cor_template_slice = nii_template.get_fdata()[cor_left_bound:cor_right_bound, cor_slice, :]
    cor_baseline_slice = np.full_like(cor_template_slice, np.nan)
    cor_baseline_slice[:, cor_bot_bound:cor_top_bound] = nii_baseline.get_fdata()[cor_left_bound:cor_right_bound,
                                                         cor_slice, cor_bot_bound:cor_top_bound]
    cor_baseline_slice = cor_baseline_slice[:, cor_templ_bot:cor_templ_top]

    cor_slicewise_slice = np.full_like(cor_template_slice, np.nan)
    cor_slicewise_slice[:, cor_bot_bound:cor_top_bound] = nii_slicewise.get_fdata()[cor_left_bound:cor_right_bound,
                                                          cor_slice, cor_bot_bound:cor_top_bound]
    cor_slicewise_slice = cor_slicewise_slice[:, cor_templ_bot:cor_templ_top]

    cor_template_slice = cor_template_slice[:, cor_templ_bot:cor_templ_top]

    spinal_levels = {5: range(300, 333),  # C5
                     6: range(269, 300),  # C6
                     7: range(238, 269),  # C7
                     8: range(206, 238),  # C8
                     9: range(172, 206),  # T1
                     10: range(135, 172)}  # T2
    data_spinal_levels = np.full_like(cor_template_slice, 0, dtype=int)
    for level, range_ in spinal_levels.items():
        for r in range_:
            data_spinal_levels[:, r - cor_templ_bot] = level

    # Create a figure showing the average tSNR map in PAM50 space for baseline and slicewise shim
    vmin = 5
    vmax = 14
    gs_1 = gs.subgridspec(ncols=2, nrows=2, width_ratios=[1, 1], height_ratios=[4, 1])
    axs = gs_1.subplots()
    axs[0, 0].imshow(np.rot90(cor_template_slice), cmap='gray')
    im1 = axs[0, 0].imshow(np.rot90(cor_baseline_slice), vmin=vmin, vmax=vmax, cmap='turbo')
    axs[0, 0].hlines(hline, xmin=0, xmax=cor_right_bound - cor_left_bound - 1, color='black', linestyle='--',
                     linewidth=1)
    axs[0, 0].axis('off')
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].set_title('Baseline\n2nd order shim', fontsize=10)
    axs[0, 1].imshow(np.rot90(cor_template_slice), cmap='gray')
    axs[0, 1].imshow(np.rot90(cor_slicewise_slice), vmin=vmin, vmax=vmax, cmap='turbo')
    axs[0, 1].hlines(hline, xmin=0, xmax=cor_right_bound - cor_left_bound - 1, color='black', linestyle='--',
                     linewidth=1)
    axs[0, 1].axis('off')
    axs[0, 1].set_aspect('equal', adjustable='box')
    axs[0, 1].set_title('Slice-wise\nf0xyz shim', fontsize=10)
    axs[1, 0].imshow(np.rot90(
        nii_baseline.get_fdata()[axial_left_bound:axial_right_bound, axial_bot_bound:axial_top_bound, axial_slice]),
        vmin=vmin, vmax=vmax, cmap='turbo')
    axs[1, 0].imshow(np.rot90(axial_cord_outline_nan), vmin=0, vmax=1, cmap='grey', alpha=0.75)
    axs[1, 0].axis('off')
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 1].imshow(np.rot90(
        nii_slicewise.get_fdata()[axial_left_bound:axial_right_bound, axial_bot_bound:axial_top_bound, axial_slice]),
        vmin=vmin, vmax=vmax, cmap='turbo')
    axs[1, 1].imshow(np.rot90(axial_cord_outline_nan), vmin=0, vmax=1, cmap='grey', alpha=0.75)
    axs[1, 1].axis('off')
    axs[1, 1].set_aspect('equal', adjustable='box')

    # Add a colorbar under the 4 subplots
    cbar = fig.colorbar(im1, ax=axs, orientation='horizontal', shrink=0.4, ticks=[vmin, vmax])
    cbar.set_label('tSNR', fontsize=10)

    # Add orientation labels on the left axial slice and on the left coronal
    text_fontsize = 10
    axs[1, 0].text(0.08, 0.5, 'L', color='white', fontsize=text_fontsize, fontweight='bold', ha='center', va='center',
                   transform=axs[1, 0].transAxes)
    axs[1, 0].text(0.92, 0.5, 'R', color='white', fontsize=text_fontsize, fontweight='bold', ha='center', va='center',
                   transform=axs[1, 0].transAxes)
    axs[1, 0].text(0.5, 0.1, 'A', color='white', fontsize=text_fontsize, fontweight='bold', ha='center', va='center',
                   transform=axs[1, 0].transAxes)
    axs[1, 0].text(0.5, 0.9, 'P', color='white', fontsize=text_fontsize, fontweight='bold', ha='center', va='center',
                   transform=axs[1, 0].transAxes)

    axs[0, 0].text(-0.1, 0, 'L', color='black', fontsize=text_fontsize, fontweight='bold', ha='center', va='bottom',
                   transform=axs[0, 0].transAxes)
    axs[0, 0].text(1.1, 0, 'R', color='black', fontsize=text_fontsize, fontweight='bold', ha='center', va='bottom',
                   transform=axs[0, 0].transAxes)

    # Add spinal level labels on the left of the coronal slices
    l, b, w, h = axs[0, 0].get_position(False).bounds
    width = 0.015
    left = l - w - 0.02 - 0.08
    bottom = b - 0.128
    height = h + 0.122
    ax_spl_coords = (left, bottom, width, height)
    ax_spl = fig.add_axes(ax_spl_coords, anchor='SW')
    data_spinal_alpha = np.zeros_like(data_spinal_levels, dtype=float)
    data_spinal_alpha[data_spinal_levels > 0] = 1
    data_spinal_levels_2 = np.copy(data_spinal_levels).astype(float)
    data_spinal_levels_2[data_spinal_levels % 2 == 0] = 0.5
    data_spinal_levels_2[data_spinal_levels % 2 == 1] = 0.75
    ax_spl.imshow(np.rot90(data_spinal_levels_2), vmin=0, vmax=1, cmap='gray', alpha=np.rot90(data_spinal_alpha))
    ax_spl.set_xticks([])
    ax_spl.spines['top'].set_visible(False)
    ax_spl.spines['bottom'].set_visible(False)
    ax_spl.spines['left'].set_visible(False)
    ax_spl.spines['right'].set_visible(False)
    ax_spl.set_yticks([cor_template_slice.shape[1] - (np.mean(spinal_levels[i]) - cor_templ_bot) for i in range(5, 11)])
    ax_spl.set_yticklabels(['C5', 'C6', 'C7', 'C8', 'T1', 'T2'])
    ax_spl.tick_params(axis='y', which='both', bottom=False, top=False, left=False, right=False, labelcolor='grey',
                       labelsize=8)
    ax_spl.set_ylabel('Spinal levels', fontsize=8, color='grey')
    ax_spl.set_aspect('auto', adjustable='datalim')
    ax_spl.set_label('Spinal levels')


def create_tsnr_violin_plot(fig, gs, snr_baseline: list, snr_shim: list):
    # Create a violin plot with matplotolib showing the distribution of SNR values for baseline and shim
    # Link corresponding values for each subject with a line

    gs1 = gs.subgridspec(ncols=1, nrows=2, height_ratios=[9, 0.6])
    axs = gs1.subplots()
    axs[1].axis('off')

    ax = axs[0]
    data = [snr_baseline, snr_shim]
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline\n2nd order shim', 'Slice-wise\nf0xyz shim'])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.set_ylabel('Mean tSNR for each participant')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Link corresponding values for each subject with a line
    for i in range(len(snr_baseline)):
        ax.plot([1, 2], [snr_baseline[i], snr_shim[i]], color='gray', linestyle='--', marker='o', alpha=0.7)


def find_moco_for_tsnr_calculation(config, ID, task, acq_name):
    files = glob.glob(os.path.join(
        config["raw_dir"],
        config["preprocess_dir"]["main_dir"].format(ID),
        "func",
        f"task-{task}_acq-{acq_name}",
        "sct_fmri_moco",
        f"sub-{ID}_task-{task}_acq-{acq_name}*_bold_moco.nii.gz"
    ))
    if len(files) == 0:
        return None
    elif len(files) == 1:
        selected_file = files[0]
    else:
        max_volumes = 0
        selected_file = None
        for f in files:
            img = nib.load(f)
            n_volumes = img.shape[3]
            if n_volumes > max_volumes:
                max_volumes = n_volumes
                selected_file = f
    return selected_file


class FigureEpiComparison:
    def __init__(self, config, IDs, redo):
        self.IDs = IDs
        self.config = config
        self.redo = redo

        self.path_main_fig = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"])
        self.path_fig_epi_comparison = os.path.join(self.path_main_fig, "epi_comparison")
        if not os.path.exists(self.path_fig_epi_comparison):
            os.makedirs(self.path_fig_epi_comparison)
        self.path_fig_data = os.path.join(self.path_fig_epi_comparison, "data")
        if not os.path.exists(self.path_fig_data):
            os.makedirs(self.path_fig_data)

    def create_figure(self, show_avg=False):
        print("=== Create EPI comparison figure ===", flush=True)

        for ID in self.IDs:
            # Figure looks better if we use more volumes (task has more volumes than rest)
            # try:
            #     create_mocomean_same_vols(ID, "rest", self.config, self.path_fig_data, self.redo)
            # except RuntimeError as e:
            #     create_mocomean_same_vols(ID, "motor", self.config, self.path_fig_data, self.redo)
            create_mocomean_same_vols(ID, "motor", self.config, self.path_fig_data, self.redo)

        ### Create 1 figure per subject, showing moco mean in native space between baseline and slicewise shim
        if show_avg:
            name_baseline = [a for a in self.config["design_exp"]["acq_names"] if "Base" in a][0]
            name_slicewise = [a for a in self.config["design_exp"]["acq_names"] if "Slice" in a][0]
            fname_avg_baseline = self._create_avg_moco_mean_in_pam50(self.IDs, name_baseline)
            fname_avg_slicewise = self._create_avg_moco_mean_in_pam50(self.IDs, name_slicewise)
        else:
            fname_avg_baseline = None
            fname_avg_slicewise = None

        self._create_fullcomp_figure(fname_avg_baseline, fname_avg_slicewise, show_avg=False)
        for ID in self.IDs:
            self._create_comp_figure(ID, fname_avg_baseline, fname_avg_slicewise, False)
        # Todo: Create a gif that switches between baseline and slicewise

    def _select_moco_mean_same_vols(self, ID, acq_name):
        fname_moco_mean = os.path.join(self.path_fig_data, f"sub-{ID}", f"sub-{ID}_task-rest_acq-{acq_name}_bold_moco_mean_samevols.nii.gz")
        if not os.path.exists(fname_moco_mean):
            fname_moco_mean = os.path.join(self.path_fig_data, f"sub-{ID}", f"sub-{ID}_task-motor_acq-{acq_name}_bold_moco_mean_samevols.nii.gz")
        if not os.path.exists(fname_moco_mean):
            raise RuntimeError(f"No moco mean same vols found for sub-{ID} acq-{acq_name}")
        return fname_moco_mean

    def _get_task_of_moco_mean_same_vols(self, ID, acq_name):
        fname_samevols = self._select_moco_mean_same_vols(ID, acq_name)
        if os.path.basename(fname_samevols).find("rest") != -1:
            task = "rest"
        elif os.path.basename(fname_samevols).find("motor") != -1:
            task = "motor"
        else:
            raise RuntimeError(f"Cannot find task in filename: {fname_samevols}")
        return task

    def _create_avg_moco_mean_in_pam50(self, IDs, acq_name):

        fname_template = os.path.join(self.config["code_dir"], "template", self.config["PAM50_t2"])
        data_sum = None
        roi_sum = None
        task = None
        for ID in IDs:
            task = self._get_task_of_moco_mean_same_vols(ID, acq_name)

            _, fname_warp_from_func_to_template, _, = get_fname_seg_and_warps(ID, task, acq_name, self.config)
            fname_moco_mean = self._select_moco_mean_same_vols(ID, acq_name)

            if not os.path.exists(os.path.join(self.path_fig_data, f"sub-{ID}")):
                os.makedirs(os.path.join(self.path_fig_data, f"sub-{ID}"))

            nii_roi = count_roi_in_template(os.path.join(self.path_fig_data, f"sub-{ID}"),
                                            ID, task, acq_name,
                                            fname_moco_mean, fname_warp_from_func_to_template, fname_template, self.redo)

            fname_moco_in_template = os.path.join(self.path_fig_data, f"sub-{ID}",
                                                  f"sub-{ID}_task-{task}_acq-{acq_name}_bold_moco_mean_in_PAM50.nii.gz")
            if not os.path.exists(fname_moco_in_template) or self.redo:
                cmd_coreg = f"sct_apply_transfo -i {fname_moco_mean} -d {fname_template} -w {fname_warp_from_func_to_template} -o {fname_moco_in_template}"
                os.system(cmd_coreg)

            nii = nib.load(fname_moco_in_template)
            if data_sum is None:
                data_sum = nii.get_fdata()
                roi_sum = nii_roi.get_fdata()
            else:
                data_sum += nii.get_fdata()
                roi_sum += nii_roi.get_fdata()

        data_avg = np.divide(data_sum, roi_sum, out=np.zeros_like(data_sum), where=roi_sum != 0)
        fname_avg = os.path.join(self.path_fig_data, f"avg_task-{task}_acq-{acq_name}_bold_moco_mean_in_PAM50.nii.gz")
        nii_avg = nib.Nifti1Image(data_avg, affine=nii.affine, header=nii.header)
        nib.save(nii_avg, fname_avg)
        return fname_avg

    def _create_fullcomp_figure(self, fname_avg_baseline=None, fname_avg_slicewise=None, show_avg=False, show_slice_factor=2):
        # Create figure that shows moco mean in native space between baseline and slicewise shim
        # Highlight specific slices

        # highlight = {'090': [1, 3, 7, 9, 25, 27],
        #              '094': [],
        #              '095': [1, 3, 11, 15, 21, 23, 27],
        #              '100': [1, 5, 7, 9],
        #              '101': [1, 5, 13, 27],
        #              '106': [1, 11, 15, 19, 21],
        #              'avg': [168]}

        highlight = {'090': {1: 'sigtot', 3: 'sigtot', 7: 'sigtot', 9: 'geo'},
                     '094': {1: 'sigtot', 5: 'geo', 27: 'sigtot'},
                     '095': {1: 'sigvert', 3: 'sigvert', 21: 'sigtot'},
                     '101': {1: 'geo', 5: 'sigtot', 27: 'sigvert'},
                     '106': {1: 'sigtot', 11: 'geo', 15: 'geo', 19: 'geo'}}

        color = {'sigtot': '#2ca02c', 'sigvert': '#d62728', 'geo': '#9467bd'}

        name_baseline = [a for a in self.config["design_exp"]["acq_names"] if "Base" in a][0]
        name_slicewise = [a for a in self.config["design_exp"]["acq_names"] if "Slice" in a][0]

        n_part = len(self.IDs)
        if n_part > 5:
            n_part = 5

        ids_to_show = []
        # 094, 096, '100'
        chose_if_available = ('090', '094', '095', '101', '106')

        for i_part in range(n_part):
            ID = chose_if_available[i_part] if chose_if_available[i_part] in self.IDs else self.IDs[i_part]
            ids_to_show.append(ID)

        n_max_slices = 0
        for i_id, ID in enumerate(ids_to_show):
            # Paths for baseline and slicewise shim images
            fname_baseline = self._select_moco_mean_same_vols(ID, name_baseline)
            nii_tmp = nib.load(fname_baseline)
            if n_max_slices < nii_tmp.shape[2]:
                n_max_slices = nii_tmp.shape[2]

        width_ratios = []
        [width_ratios.extend([1, 1, 0.09]) for _ in range(n_part)]

        if show_avg:
            width_ratios.extend([1, 1])
            avg_baseline = nib.load(fname_avg_baseline).get_fdata()
            avg_slicewise = nib.load(fname_avg_slicewise).get_fdata()
            fig = plt.figure(figsize=(2.1 * (n_part + 1), n_max_slices // show_slice_factor))
            gs_main = gridspec.GridSpec(2, ((n_part + 1) * 2) + n_part, figure=fig, hspace=0, wspace=0, width_ratios=width_ratios, height_ratios=[0.001,1])
        else:
            width_ratios = width_ratios[:-1]  # remove the last 0.1
            fig = plt.figure(figsize=(2.1 * n_part, n_max_slices // show_slice_factor))
            gs_main = gridspec.GridSpec(2, (n_part * 2) + (n_part -1), figure=fig, hspace=0, wspace=0, width_ratios=width_ratios, height_ratios=[0.001,1])

        print("IDs to show in the figure:", ids_to_show, flush=True)
        for i_id, ID in enumerate(ids_to_show):

            gs_title = gs_main[0, i_id * 3:(i_id + 1) * 3 - 1].subgridspec(1, 1)
            ax_title = gs_title.subplots()
            ax_title.axis('off')
            ax_title.set_title(f"ID {ID}", fontsize=12, fontweight='bold')

            task = self._get_task_of_moco_mean_same_vols(ID, name_baseline)
            # Paths for baseline and slicewise shim images
            fname_baseline = self._select_moco_mean_same_vols(ID, name_baseline)
            fname_seg_baseline, _, fname_warp_from_pam50_to_func_baseline = get_fname_seg_and_warps(ID, task, name_baseline, self.config)
            fname_slicewise = self._select_moco_mean_same_vols(ID, name_slicewise)
            fname_seg_slicewise, _, fname_warp_from_pam50_to_func_slicewise = get_fname_seg_and_warps(ID, task, name_slicewise, self.config)

            # Load images and masks
            img_baseline = nib.load(fname_baseline).get_fdata()
            img_slicewise = nib.load(fname_slicewise).get_fdata()
            mask_baseline = nib.load(fname_seg_baseline).get_fdata()
            mask_slicewise = nib.load(fname_seg_slicewise).get_fdata()

            bound_lr = 16  # left-right bound
            bound_ud = 16  # up-down bound
            vmin, vmax = calc_vmin_vmax(mask_baseline, mask_slicewise, img_baseline, img_slicewise, False,
                                        avg_baseline=None, avg_slicewise=None, show_slice_factor=2,
                                        bound_lr=bound_lr, bound_ud=bound_ud)

            delta = vmax - vmin
            # vmin = vmin + 0.1 * delta
            vmax = vmax - 0.1 * delta

            gs_baseline = gs_main[1, i_id * 3].subgridspec(round(n_max_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
            gs_slicewise = gs_main[1, i_id * 3 + 1].subgridspec(round(n_max_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
            axs_baseline = gs_baseline.subplots()
            axs_slicewise = gs_slicewise.subplots()

            range_slices = range(n_max_slices - 1, -1, -show_slice_factor)
            for idx, slice_idx in enumerate(range_slices):
                com_baseline = center_of_mass(mask_baseline[:, :, slice_idx])
                com_slicewise = center_of_mass(mask_slicewise[:, :, slice_idx])

                # Define cropping bounds
                crop_x_baseline = slice(max(0, int(com_baseline[0] - bound_lr)),
                                        min(mask_baseline.shape[0], int(com_baseline[0] + bound_lr)))
                crop_y_baseline = slice(max(0, int(com_baseline[1] - bound_ud)),
                                        min(mask_baseline.shape[1], int(com_baseline[1] + bound_ud)))

                crop_x_slicewise = slice(max(0, int(com_slicewise[0] - bound_lr)),
                                         min(mask_slicewise.shape[0], int(com_slicewise[0] + bound_lr)))
                crop_y_slicewise = slice(max(0, int(com_slicewise[1] - bound_ud)),
                                         min(mask_slicewise.shape[1], int(com_slicewise[1] + bound_ud)))

                # Crop the images
                cropped_baseline = img_baseline[crop_x_baseline, crop_y_baseline, slice_idx]
                cropped_slicewise = img_slicewise[crop_x_slicewise, crop_y_slicewise, slice_idx]

                axs_baseline[idx].imshow(cropped_baseline.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                axs_baseline[idx].set_aspect('equal', adjustable='box')
                axs_slicewise[idx].imshow(cropped_slicewise.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                axs_slicewise[idx].set_aspect('equal', adjustable='box')

                color_baseline = '#1f77b4'
                color_slicewise = '#ff7f0e'
                if idx == len(range_slices) - 1 and i_id == 0:
                    legend_fontsize = 10
                    legend_elements = [Patch(facecolor='white', edgecolor=color_baseline, label='Baseline',
                                             linewidth=1.5),
                                       Patch(facecolor='white', edgecolor=color_slicewise, label='Slicewise',
                                             linewidth=1.5)]
                    axs_baseline[idx].legend(handles=legend_elements, loc=(0, -0.75), fontsize=legend_fontsize)

                    legend_elements = [Arrow(0, 0, 0.2, 0, color=color['sigtot'], label='Improvement in signal loss from overall inhomogeneity', width=0.1, lw=2),
                                       Arrow(0, 0, 0.2, 0, color=color['sigvert'], label='Improvement in signal loss from inter-vertebral discs', width=0.1, lw=2),
                                       Arrow(0, 0, 0.2, 0, color=color['geo'], label='Improvement in geometric distortions', width=0.1, lw=2)]
                    axs_slicewise[idx].legend(handles=legend_elements, loc=(1.2, -1), fontsize=legend_fontsize, handler_map={mpatches.Arrow : HandlerPatch(patch_func=make_legend_arrow)})

                # Color the borders of the baseline images in yellow to differentiate them from the slicewise images
                spine_thickness = 1.25
                axs_baseline[idx].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False, bottom=False, left=False)
                # Example: Change spine border thickness
                for spine in axs_baseline[idx].spines.values():
                    spine.set_linewidth(spine_thickness)  # Set the border thickness to 2
                axs_baseline[idx].spines['left'].set_edgecolor(color_baseline)
                axs_baseline[idx].spines['right'].set_edgecolor(color_baseline)
                if idx == 0:
                    axs_baseline[idx].spines['top'].set_edgecolor(color_baseline)
                    axs_baseline[idx].spines['bottom'].set_visible(False)
                elif idx == len(range_slices) - 1:
                    axs_baseline[idx].spines['top'].set_visible(False)
                    axs_baseline[idx].spines['bottom'].set_edgecolor(color_baseline)
                else:
                    axs_baseline[idx].spines['top'].set_visible(False)
                    axs_baseline[idx].spines['top'].set_visible(False)


                axs_slicewise[idx].tick_params(axis='both', which='both', length=0, labelbottom=False,
                                              labelleft=False, bottom=False, left=False)
                # Example: Change spine border thickness
                for spine in axs_slicewise[idx].spines.values():
                    spine.set_linewidth(spine_thickness)  # Set the border thickness to 2
                axs_slicewise[idx].spines['left'].set_edgecolor(color_slicewise)
                axs_slicewise[idx].spines['right'].set_edgecolor(color_slicewise)
                if idx == 0:
                    axs_slicewise[idx].spines['top'].set_edgecolor(color_slicewise)
                    axs_slicewise[idx].spines['bottom'].set_visible(False)
                elif idx == len(range_slices) - 1:
                    axs_slicewise[idx].spines['top'].set_visible(False)
                    axs_slicewise[idx].spines['bottom'].set_edgecolor(color_slicewise)
                else:
                    axs_slicewise[idx].spines['top'].set_visible(False)
                    axs_slicewise[idx].spines['top'].set_visible(False)

                template_slice_idx = self.func_slice_to_template_slice(slice_idx, com_baseline[0], com_baseline[1], fname_warp_from_pam50_to_func_baseline, fname_baseline, ID, task, name_baseline)
                spinal_level = template_slice_to_spinal_level(template_slice_idx)[1]
                axs_baseline[idx].text(0.1, 0.9, spinal_level, color='white', fontsize=5, fontweight='bold', ha='center', va='center', transform=axs_baseline[idx].transAxes)

                if ID in highlight and slice_idx in highlight[ID].keys():
                    # Add an arrow pointing to a specific point
                    axs_slicewise[idx].annotate(
                        '',  # Empty text
                        xy=(0.3, 0.3),  # Arrowhead location
                        xytext=(0.05, 0.05),  # Arrow tail location
                        xycoords='axes fraction',
                        arrowprops=dict(arrowstyle="->", color=color[highlight[ID][slice_idx]], lw=2),
                    )

        if show_avg:
            gs_title = gs_main[0, (i_id + 1) * 3:(i_id + 2) * 3 - 1].subgridspec(1, 1)
            ax_title = gs_title.subplots()
            ax_title.axis('off')
            ax_title.set_title(f"Average", fontsize=12, fontweight='bold')

            gs_baseline = gs_main[(i_id + 1) * 3].subgridspec(round(n_max_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
            gs_slicewise = gs_main[(i_id + 1) * 3 + 1].subgridspec(round(n_max_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
            axs_baseline_avg = gs_baseline.subplots()
            axs_slicewise_avg = gs_slicewise.subplots()

            bound_lr = 24  # left-right bound
            bound_ud = 24  # up-down bound

            fname_seg_template = os.path.join(self.config["code_dir"], "template", self.config["PAM50_cord"])
            mask_template_seg = nib.load(fname_seg_template).get_fdata()

            vmin, vmax = calc_vmin_vmax(mask_template_seg, mask_template_seg, avg_baseline, avg_slicewise, False,
                                        avg_baseline=None, avg_slicewise=None, show_slice_factor=2,
                                        bound_lr=bound_lr, bound_ud=bound_ud)

            delta = vmax - vmin
            # vmin = vmin + 0.1 * delta
            vmax = vmax - 0.5 * delta

            for idx, slice_idx in enumerate(range(n_max_slices - 1, -1, -show_slice_factor)):
                # Use last baseline for reference func
                slice_idx = self.func_slice_to_template_slice(slice_idx, com_baseline[0], com_baseline[1], fname_warp_from_pam50_to_func_baseline, fname_baseline, ID, task, name_baseline)

                com = center_of_mass(mask_template_seg[:, :, slice_idx])

                # Define cropping bounds
                crop_x = slice(max(0, int(com[0] - bound_lr)),
                               min(mask_template_seg.shape[0], int(com[0] + bound_lr)))
                crop_y = slice(max(0, int(com[1] - bound_ud)),
                               min(mask_template_seg.shape[1], int(com[1] + bound_ud)))

                cropped_baseline_avg = avg_baseline[crop_x, crop_y, slice_idx]
                cropped_slicewise_avg = avg_slicewise[crop_x, crop_y, slice_idx]

                axs_baseline_avg[idx].imshow(cropped_baseline_avg.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                axs_baseline_avg[idx].axis('off')
                axs_baseline_avg[idx].set_aspect('equal', adjustable='box')
                axs_slicewise_avg[idx].imshow(cropped_slicewise_avg.T, cmap='gray', origin='lower', vmin=vmin,
                                              vmax=vmax)
                axs_slicewise_avg[idx].axis('off')
                axs_slicewise_avg[idx].set_aspect('equal', adjustable='box')

                spinal_level = template_slice_to_spinal_level(slice_idx)[1]
                axs_baseline_avg[idx].text(0.1, 0.9, spinal_level, color='white', fontsize=5, fontweight='bold',
                                           ha='center', va='center', transform=axs_baseline_avg[idx].transAxes)

                if 'avg' in highlight and slice_idx in highlight['avg']:
                    # Add an arrow pointing to a specific point
                    axs_slicewise_avg[idx].annotate(
                        '',  # Empty text
                        xy=(0.3, 0.3),  # Arrowhead location
                        xytext=(0.05, 0.05),  # Arrow tail location
                        xycoords='axes fraction',
                        arrowprops=dict(arrowstyle="->", color='red', lw=2),
                    )

        self.fname_fig_epi_comparison = os.path.join(self.path_fig_epi_comparison, f"epi_comparison.png")
        fig.savefig(self.fname_fig_epi_comparison, dpi=2000)

    def _create_comp_figure(self, ID, fname_avg_baseline, fname_avg_slicewise, show_avg=False, show_slice_factor=2):
        # Todo: Subject 93 has unknown slice ID, issue #101
        # Create figure that shows moco mean in native space between baseline and slicewise shim
        name_baseline = [a for a in self.config["design_exp"]["acq_names"] if "Base" in a][0]
        name_slicewise = [a for a in self.config["design_exp"]["acq_names"] if "Slice" in a][0]

        task = self._get_task_of_moco_mean_same_vols(ID, name_baseline)
        # Paths for baseline and slicewise shim images
        fname_baseline = self._select_moco_mean_same_vols(ID, name_baseline)
        fname_seg_baseline, _, fname_warp_from_pam50_to_func_baseline = get_fname_seg_and_warps(ID, task, name_baseline, self.config)
        fname_slicewise = self._select_moco_mean_same_vols(ID, name_slicewise)
        fname_seg_slicewise, _, fname_warp_from_pam50_to_func_slicewise = get_fname_seg_and_warps(ID, task, name_slicewise, self.config)

        # Load images and masks
        img_baseline = nib.load(fname_baseline).get_fdata()
        img_slicewise = nib.load(fname_slicewise).get_fdata()
        mask_baseline = nib.load(fname_seg_baseline).get_fdata()
        mask_slicewise = nib.load(fname_seg_slicewise).get_fdata()

        if show_avg:
            # Compute the average moco mean in native space by warping the average in PAM50 back to native space
            fname_avg_in_func_baseline = os.path.join(self.path_fig_data, f"sub-{ID}",
                                                      f"sub-{ID}_task-{task}_acq-{name_baseline}_bold_moco_mean_avg_in_func.nii.gz")
            cmd = f"sct_apply_transfo -i {fname_avg_baseline} -d {fname_baseline} -w {fname_warp_from_pam50_to_func_baseline} -o {fname_avg_in_func_baseline}"
            os.system(cmd)
            fname_avg_in_func_slicewise = os.path.join(self.path_fig_data, f"sub-{ID}",
                                                       f"sub-{ID}_task-{task}_acq-{name_slicewise}_bold_moco_mean_avg_in_func.nii.gz")
            cmd = f"sct_apply_transfo -i {fname_avg_slicewise} -d {fname_slicewise} -w {fname_warp_from_pam50_to_func_slicewise} -o {fname_avg_in_func_slicewise}"
            os.system(cmd)

            # Load average images
            avg_baseline = nib.load(fname_avg_in_func_baseline).get_fdata()
            avg_slicewise = nib.load(fname_avg_in_func_slicewise).get_fdata()

        n_slices = img_baseline.shape[2]

        if show_avg:
            fig = plt.figure(figsize=(3.99, round(n_slices / show_slice_factor + 0.1)))
            gs_main = gridspec.GridSpec(1, 4, figure=fig, hspace=0, wspace=0)
        else:
            fig = plt.figure(figsize=(1.99, round(n_slices / show_slice_factor + 0.1)))
            gs_main = gridspec.GridSpec(1, 2, figure=fig, hspace=0, wspace=0)

        title_fontsize = 5
        # Python rounds 12.5 to 12 instead of 13, so we add 0.1 to make sure we have enough space for all slices
        gs_baseline = gs_main[0].subgridspec(round(n_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
        gs_slicewise = gs_main[1].subgridspec(round(n_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
        axs_baseline = gs_baseline.subplots()
        axs_baseline[0].set_title(f"ID {ID} baseline\n2nd order shim", fontsize=title_fontsize)
        axs_slicewise = gs_slicewise.subplots()
        axs_slicewise[0].set_title(f"ID {ID} slice-wise\nf0xyz shim", fontsize=title_fontsize)

        if show_avg:
            gs_baseline_avg = gs_main[2].subgridspec(round(n_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
            gs_slicewise_avg = gs_main[3].subgridspec(round(n_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
            axs_baseline_avg = gs_baseline_avg.subplots()
            axs_baseline_avg[0].set_title(f"Average baseline\n2nd order shim", fontsize=title_fontsize)
            axs_slicewise_avg = gs_slicewise_avg.subplots()
            axs_slicewise_avg[0].set_title(f"Average slice-wise\nf0xyz shim", fontsize=title_fontsize)

        bound_lr = 16  # left-right bound
        bound_ud = 16  # up-down bound
        vmin, vmax = calc_vmin_vmax(mask_baseline, mask_slicewise, img_baseline, img_slicewise, show_avg,
                                   avg_baseline=None, avg_slicewise=None, show_slice_factor=2,
                                   bound_lr=bound_lr, bound_ud=bound_ud)
        delta = vmax - vmin
        # vmin = vmin + 0.1 * delta
        vmax = vmax - 0.1 * delta

        for idx, slice_idx in enumerate(range(n_slices - 1, -1, -show_slice_factor)):
            com_baseline = center_of_mass(mask_baseline[:, :, slice_idx])
            com_slicewise = center_of_mass(mask_slicewise[:, :, slice_idx])

            # Define cropping bounds
            crop_x_baseline = slice(max(0, int(com_baseline[0] - bound_lr)),
                                    min(mask_baseline.shape[0], int(com_baseline[0] + bound_lr)))
            crop_y_baseline = slice(max(0, int(com_baseline[1] - bound_ud)),
                                    min(mask_baseline.shape[1], int(com_baseline[1] + bound_ud)))

            crop_x_slicewise = slice(max(0, int(com_slicewise[0] - bound_lr)),
                                     min(mask_slicewise.shape[0], int(com_slicewise[0] + bound_lr)))
            crop_y_slicewise = slice(max(0, int(com_slicewise[1] - bound_ud)),
                                     min(mask_slicewise.shape[1], int(com_slicewise[1] + bound_ud)))

            # Crop the images
            cropped_baseline = img_baseline[crop_x_baseline, crop_y_baseline, slice_idx]
            cropped_slicewise = img_slicewise[crop_x_slicewise, crop_y_slicewise, slice_idx]

            if show_avg:
                cropped_baseline_avg = avg_baseline[crop_x_baseline, crop_y_baseline, slice_idx]
                cropped_slicewise_avg = avg_slicewise[crop_x_slicewise, crop_y_slicewise, slice_idx]

                axs_baseline_avg[idx].imshow(cropped_baseline_avg.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                axs_baseline_avg[idx].axis('off')
                axs_baseline_avg[idx].set_aspect('equal', adjustable='box')
                axs_slicewise_avg[idx].imshow(cropped_slicewise_avg.T, cmap='gray', origin='lower', vmin=vmin,
                                              vmax=vmax)
                axs_slicewise_avg[idx].axis('off')
                axs_slicewise_avg[idx].set_aspect('equal', adjustable='box')

            axs_baseline[idx].imshow(cropped_baseline.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axs_baseline[idx].axis('off')
            axs_baseline[idx].set_aspect('equal', adjustable='box')
            axs_slicewise[idx].imshow(cropped_slicewise.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            axs_slicewise[idx].axis('off')
            axs_slicewise[idx].set_aspect('equal', adjustable='box')

            template_slice_idx = self.func_slice_to_template_slice(slice_idx, com_baseline[0], com_baseline[1], fname_warp_from_pam50_to_func_baseline, fname_baseline, ID, task, name_baseline)
            spinal_level = template_slice_to_spinal_level(template_slice_idx)[1]
            axs_baseline[idx].text(0.1, 0.9, spinal_level, color='white', fontsize=5, fontweight='bold', ha='center', va='center', transform=axs_baseline[idx].transAxes)

        self.fname_fig_epi_comparison = os.path.join(self.path_fig_epi_comparison, f"sub-{ID}_epi_comparison.png")
        fig.savefig(self.fname_fig_epi_comparison, dpi=2000)

    def func_slice_to_template_slice(self, func_slice, com1, com2, fname_warp_template_to_func, fname_moco_mean, ID, task, acq_name):
        name = f"sub-{ID}_task-{task}_acq-{acq_name}"

        fname_slice_temp = os.path.join(self.path_fig_data, "template_slice.nii.gz")

        if not os.path.exists(fname_slice_temp) or self.redo:
            # Take the template, overwrite it with the slice number, and warp it to func space.
            fname_template = os.path.join(self.config["code_dir"], "template", self.config["PAM50_t2"])
            nii_template = nib.load(fname_template)
            data = nii_template.get_fdata()
            for i_slice in range(nii_template.shape[2]):
                data_template_slice = np.full_like(data[..., 0], i_slice)
                data[:, :, i_slice] = data_template_slice

            nii_slice_temp = nib.Nifti1Image(data.astype(np.int16), affine=nii_template.affine, header=nii_template.header)
            nib.save(nii_slice_temp, fname_slice_temp)

        fname_template_slice_in_func = os.path.join(self.path_fig_data, f"sub-{ID}", f"{name}_template_slice_in_func.nii.gz")
        if not os.path.exists(fname_template_slice_in_func) or self.redo:
            cmd_coreg = f"sct_apply_transfo -i {fname_slice_temp} -d {fname_moco_mean} -w {fname_warp_template_to_func} -o {fname_template_slice_in_func}"
            os.system(cmd_coreg)

        nii_slice_func = nib.load(fname_template_slice_in_func)
        slice_temp = int(nii_slice_func.get_fdata()[int(com1), int(com2), func_slice])
        return slice_temp


def template_slice_to_spinal_level(template_slice):
    spinal_levels = {
        0: range(435, 440),
        1: range(420, 435),  # C1...
        2: range(399, 420),
        3: range(366, 399),
        4: range(333, 366),
        5: range(300, 333),
        6: range(269, 300),
        7: range(238, 269),
        8: range(206, 238),
        9: range(172, 206),
        10: range(135, 172),
        11: range(94, 135),
        12: range(47, 94),
        13: range(420, 47)
    }
    spinal_levels_to_label = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'C8', 9: 'T1', 10: 'T2', 11: 'T3', 12: 'T4', 13: 'T5'}

    data_spinal_levels = np.zeros((440,), dtype=int)
    for level, range_ in spinal_levels.items():
        for r in range_:
            data_spinal_levels[r] = level

    return data_spinal_levels[template_slice], spinal_levels_to_label.get(data_spinal_levels[template_slice], 'Unknown')


def template_slice_to_vert_level(template_slice):
    vert_levels = {
        0: range(413, 440),
        1: range(388, 413),  # C1
        2: range(357, 388),  # C2
        3: range(322, 357),  # ...
        4: range(285, 322),
        5: range(250, 285),
        6: range(221, 250),
        7: range(187, 221),
        8: range(142, 187),
        9: range(96, 142),
        10: range(52, 96),
        11: range(3, 52),
        12: range(0, 3)
    }

    vert_levels_to_label = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3',
                            11: 'T4', 12: 'T5'}

    data_vert_levels = np.zeros((440,), dtype=int)
    for level, range_ in vert_levels.items():
        for r in range_:
            data_vert_levels[r] = level

    return data_vert_levels[template_slice], vert_levels_to_label.get(data_vert_levels[template_slice], 'Unknown')


def calc_vmin_vmax(mask_baseline, mask_slicewise, img_baseline, img_slicewise, show_avg, avg_baseline=None, avg_slicewise=None, show_slice_factor=2, bound_lr=16, bound_ud=16):
    n_slices = img_baseline.shape[2]
    vmin = None
    vmax = None
    for idx, slice_idx in enumerate(range(n_slices - 1, -1, -show_slice_factor)):
        com_baseline = center_of_mass(mask_baseline[:, :, slice_idx])
        com_slicewise = center_of_mass(mask_slicewise[:, :, slice_idx])

        # Define cropping bounds
        crop_x_baseline = slice(max(0, int(com_baseline[0] - bound_lr)),
                                min(mask_baseline.shape[0], int(com_baseline[0] + bound_lr)))
        crop_y_baseline = slice(max(0, int(com_baseline[1] - bound_ud)),
                                min(mask_baseline.shape[1], int(com_baseline[1] + bound_ud)))

        crop_x_slicewise = slice(max(0, int(com_slicewise[0] - bound_lr)),
                                 min(mask_slicewise.shape[0], int(com_slicewise[0] + bound_lr)))
        crop_y_slicewise = slice(max(0, int(com_slicewise[1] - bound_ud)),
                                 min(mask_slicewise.shape[1], int(com_slicewise[1] + bound_ud)))

        # Crop the images
        cropped_baseline = img_baseline[crop_x_baseline, crop_y_baseline, slice_idx]
        cropped_slicewise = img_slicewise[crop_x_slicewise, crop_y_slicewise, slice_idx]
        if vmin is None:
            vmin = min(cropped_baseline.min(), cropped_slicewise.min())
            vmax = max(cropped_baseline.max(), cropped_slicewise.max())
        else:
            vmin = min(vmin, cropped_baseline.min(), cropped_slicewise.min())
            vmax = max(vmax, cropped_baseline.max(), cropped_slicewise.max())

        if show_avg:
            cropped_baseline_avg = avg_baseline[crop_x_baseline, crop_y_baseline, slice_idx]
            cropped_slicewise_avg = avg_slicewise[crop_x_slicewise, crop_y_slicewise, slice_idx]

            vmin = min(vmin, cropped_baseline_avg.min(), cropped_slicewise_avg.min())
            vmax = max(vmax, cropped_baseline_avg.max(), cropped_slicewise_avg.max())

    if vmin is None or vmax is None:
        raise RuntimeError("Could not compute vmin and vmax for the images")

    return vmin, vmax


def get_fname_with_max_volumes(ID, task, acq_name, config, mod_to_return="moco_mean"):
    # Find the acquisition with the most volumes
    fname_acq_list = sorted(glob.glob(os.path.join(
        config["raw_dir"],
        f"sub-{ID}",
        "func",
        f"sub-{ID}_task-{task}_acq-{acq_name}*_bold.nii.gz"
    )))

    if len(fname_acq_list) == 0:
        raise RuntimeError(f"No file found for sub-{ID} task-{task} acq-{acq_name}")

    # take the one with more volumes
    vols = 0
    idx = -1
    for i, fname in enumerate(fname_acq_list):
        img = nib.load(fname)
        n_vols = img.shape[3]
        if n_vols > vols:
            vols = n_vols
            idx = i

    print(f"sub-{ID} task-{task} acq-{acq_name}: {fname} with {n_vols} volumes", flush=True)

    if mod_to_return == "moco_mean":
        fname_list = sorted(glob.glob(os.path.join(
            config["raw_dir"],
            config["preprocess_dir"]["main_dir"].format(ID),
            "func",
            f"task-{task}_acq-{acq_name}",
            "sct_fmri_moco",
            f"sub-{ID}_task-{task}_acq-{acq_name}*_bold_moco_mean.nii.gz"
        )))
    elif mod_to_return == "moco":
        fname_list = sorted(glob.glob(os.path.join(
            config["raw_dir"],
            config["preprocess_dir"]["main_dir"].format(ID),
            "func",
            f"task-{task}_acq-{acq_name}",
            "sct_fmri_moco",
            f"sub-{ID}_task-{task}_acq-{acq_name}*_bold_moco.nii.gz"
        )))
    else:
        raise NotImplementedError(f"mod_to_return {mod_to_return} not implemented")

    if len(fname_list) != len(fname_acq_list):
        raise RuntimeError(
            f"Number of moco mean files does not match number of acq files for sub-{ID} task-{task} acq-{acq_name}")

    return fname_list[idx], vols


def create_mocomean_same_vols(ID, task, config, path_output, redo=False):
    # Output
    path_sub = os.path.join(path_output, f"sub-{ID}")
    fname_fig_shimbase = os.path.join(path_sub, f"sub-{ID}_task-{task}_acq-shimBase+3mm_bold_moco_mean_samevols.nii.gz")
    fname_fig_shimslice = os.path.join(path_sub, f"sub-{ID}_task-{task}_acq-shimSlice+3mm_bold_moco_mean_samevols.nii.gz")

    if not os.path.exists(fname_fig_shimbase) or not os.path.exists(fname_fig_shimslice) or redo:
        fname_moco_shimbase, vols_shimbase = get_fname_with_max_volumes(ID, task, "shimBase+3mm", config,
                                                                        mod_to_return="moco")
        fname_moco_shimslice, vols_shimslice = get_fname_with_max_volumes(ID, task, "shimSlice+3mm", config,
                                                                          mod_to_return="moco")

        vols = min(vols_shimbase, vols_shimslice)

        print(f"sub-{ID} task-{task}: shimBase+3mm has {vols_shimbase} volumes, shimSlice+3mm has {vols_shimslice} volumes", flush=True)
        print(f"Creating moco mean with same number of volumes for sub-{ID} task-{task}: {vols} volumes", flush=True)

        nii_shimbase = nib.load(fname_moco_shimbase)
        nii_shimslice = nib.load(fname_moco_shimslice)

        data_shimbase = np.mean(nii_shimbase.get_fdata()[:, :, :, :vols], axis=3)
        data_shimslice = np.mean(nii_shimslice.get_fdata()[:, :, :, :vols], axis=3)

        if not os.path.exists(path_sub):
            os.makedirs(path_sub)

        nib.save(nib.Nifti1Image(data_shimbase, affine=nii_shimbase.affine, header=nii_shimbase.header),
                 fname_fig_shimbase)
        nib.save(nib.Nifti1Image(data_shimslice, affine=nii_shimslice.affine, header=nii_shimslice.header),
                 fname_fig_shimslice)


def get_fname_seg_and_warps(ID, task, acq_name, config):

    task_name = f"task-{task}_acq-{acq_name}"

    # Segmentation
    fname_seg = os.path.join(
        config["raw_dir"],
        config["preprocess_dir"]["main_dir"].format(ID),
        "func",
        task_name,
        f"sub-{ID}_{task_name}_bold_moco_mean_seg.nii.gz"
    )
    if not os.path.exists(fname_seg):
        raise RuntimeError(f"Could not find a segmentation")

    # Get warp from func to PAM50
    fname_warp_func_to_pam50 = os.path.join(
        config["raw_dir"],
        config["preprocess_dir"]["main_dir"].format(ID),
        "func",
        task_name,
        f"sub-{ID}_{task_name}_from-func_to_PAM50_mode-image_xfm.nii.gz"
    )
    if not os.path.exists(fname_warp_func_to_pam50):
        raise RuntimeError(f"Could not find a func_to_PAM50 for sub-{ID} task-{task} acq-{acq_name}")

    # Get warp from PAM50 to func
    fname_warp_pam50_to_func = os.path.join(
        config["raw_dir"],
        config["preprocess_dir"]["main_dir"].format(ID),
        "func",
        task_name,
        f"sub-{ID}_{task_name}_from-PAM50_to_func_mode-image_xfm.nii.gz"
    )

    if not os.path.exists(fname_warp_pam50_to_func):
        raise RuntimeError(f"Could not find a segmentation")

    return fname_seg, fname_warp_func_to_pam50, fname_warp_pam50_to_func


def count_roi_in_template(path_output, ID, task, acq_name, fname_func, fname_warp_from_func_to_template,
                          fname_template, redo):
    fname_ones_in_func = os.path.join(path_output, f"sub-{ID}_task-{task}_acq-{acq_name}_ones.nii.gz")
    fname_ones_in_template = os.path.join(path_output, f"sub-{ID}_task-{task}_acq-{acq_name}_ones_in_PAM50.nii.gz")
    if not os.path.exists(fname_ones_in_func) or redo:
        nii_tmp = nib.load(fname_func)
        data_ones = np.ones_like(nii_tmp.get_fdata())
        nii_ones = nib.Nifti1Image(data_ones, affine=nii_tmp.affine, header=nii_tmp.header)
        nib.save(nii_ones, fname_ones_in_func)

    if not os.path.exists(fname_ones_in_template) or redo:
        cmd_coreg = f"sct_apply_transfo -i {fname_ones_in_func} -d {fname_template} -w {fname_warp_from_func_to_template} -o {fname_ones_in_template}"
        os.system(cmd_coreg)
    nii_roi = nib.load(fname_ones_in_template)
    return nii_roi


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p