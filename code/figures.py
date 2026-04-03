import os
import glob
import json
import numpy as np
import pandas as pd
import nibabel as nib

#matplotlib
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


# nilearn
from nilearn.plotting import plot_design_matrix
from nilearn.image import resample_to_img
from nilearn.image import smooth_img
from nibabel.processing import resample_from_to

#mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns



#####################################################
class Figures_main:
    '''
    The Postprocess_main class is used to setup the Post-processing path and execute the Post-processing steps.

    Attributes
    ----------
    config : dict
        Defining all the parameters of the analysis including the path to the raw data, the participants to analyze, the design of the experiment, and the preprocessing parameters
    IDs : list
        List of participant IDs to process (e.g., ['A001', 'A002'])
    verbose : bool
        Whether to print information during the each step (default: True)
    '''

    def __init__(self, config, IDs=None,verbose=True):
        if IDs==None:
            raise ValueError("Please provide the participant ID (e.g., _.stc(ID='A001')).")
        
        # Class attributes -------------------------------------------------------------------------------------
        self.config = config # load config info
        self.participant_IDs= IDs # list of the participants to analyze
        self.raw_dir = os.path.join(self.config["raw_dir"])  # directory of the raw data
        self.derivatives_dir = os.path.join(self.config["raw_dir"], self.config["derivatives_dir"])  # directory of the derivatives data
        self.first_level_dir = os.path.join(self.config["raw_dir"], self.config["first_level"]["dir"])  # directory of the derivatives data
        self.second_level_dir = os.path.join(self.config["raw_dir"], self.config["second_level"]["dir"])  # directory of the second-level analysis data
        self.manual_dir = os.path.join(self.config["raw_dir"], self.config["manual_dir"])  # directory of the manual corrections
        self.first_level_fig=os.path.join(self.config["raw_dir"], self.config["figures_dir"]["main_dir"],"first_level") 
        self.second_level_fig=os.path.join(self.config["raw_dir"], self.config["figures_dir"]["main_dir"],"second_level")

        os.makedirs(self.first_level_fig,exist_ok=True)
        os.makedirs(self.second_level_fig,exist_ok=True)
     
    def plot_first_level_maps(self, i_fnames=None, output_fname=None,titles=["shimBase","shimSlice"],cmap="autumn",stat_min=1.6, stat_max=4,background_fname=None,mask_fname=None, underlay_fname=None,task_name=None,plot_mip=True, verbose=True, redo=False,n_cols=5):
        """
        Plot first-level statistical maps for multiple participants and contrasts in a grid layout.

        To do: add spinal levels in the coronal view 
        """
        if output_fname is None:
            output_fname = os.path.join(self.first_level_dir.split("sub-")[0], f"first_level_maps_n{len(i_fnames)}_all.png")
        if i_fnames is None or len(i_fnames) == 0:
            raise ValueError("i_fnames_pair is empty")

        if not os.path.exists(output_fname) or redo:
            n_subjects = len(i_fnames)
            n_participant_rows = (n_subjects + n_cols - 1) // n_cols  # number of participant rows
            n_rows = n_participant_rows * 3  # coronal, axial, gap
            n_actual_cols = min(n_subjects, n_cols)
            total_cols = (n_cols * 4) - 1  # 2 maps + 1 spacer per participant expect for the 5th one

            # --- Load template, mask, and underlay ---
            template_img = nib.as_closest_canonical(nib.load(background_fname))
            template_data = template_img.get_fdata()
            mask_data = None
            if mask_fname is not None:
                mask_img = nib.load(mask_fname)
                mask_data = nib.as_closest_canonical(mask_img).get_fdata()

            underlay_data = None
            if underlay_fname is not None:
                underlay_data = nib.as_closest_canonical(nib.load(underlay_fname)).get_fdata()

            # --- Figure and gridspec ---
            # Figure size scales with number of participant rows
            fig_height = n_participant_rows *2
            fig_width = 7 #max paper width is 7 inches
            fig = plt.figure(figsize=(fig_width, fig_height))
            fig.subplots_adjust(left=0.01,right=0.99,top=0.94,bottom=0.01)

            height_ratios = []
            for _ in range(n_participant_rows):
                height_ratios += [6.5, 2.7, 3]  # coronal, axial, gap
            
            width_ratios = []
            for i in range(n_cols):
                width_ratios += [1, 1, 1]  # three map columns
                if i != n_cols - 1:     # add spacer except after last participant
                    width_ratios += [0.2]  # spacer column smaller

            gs = fig.add_gridspec(nrows=len(height_ratios), ncols=total_cols,
                            height_ratios=height_ratios, 
                            width_ratios=width_ratios,
                            hspace=0.01,wspace=0.1)

            for subj_idx, maps in enumerate(i_fnames):
                if len(maps) == 2:
                    maps=maps+ [None]

                col_idx = subj_idx % n_cols
                row_participant = subj_idx // n_cols 
                row_start = (subj_idx // n_cols) * 3
                col_start = (subj_idx % n_cols) * 4   # 3 for maps, 1 for spacer (subj_idx % n_cols) * 3   

                for map_idx, i_fname in enumerate(maps):
                    if map_idx == 0:
                        cmap = "winter"
                    else:
                        cmap = "autumn"
                    if i_fname is None:
                        ax = fig.add_subplot(gs[row_start, col_start + map_idx])
                        ax.axis("off")   # empty panel
                        continue

                    x_min, x_max = 35, 105
                    z_min, z_max = 130, 350
                    statmap_img = nib.as_closest_canonical(nib.load(i_fname))
                    statmap_data = statmap_img.get_fdata()
                    if mask_data is not None:
                        mask_resampled = resample_from_to(mask_img, statmap_img, order=0)  # nearest-neighbor for mask
                        mask_data = mask_resampled.get_fdata() > 0  # boolean
                        statmap_data = np.where(mask_data, statmap_data, 0)
            
                    stat_thresh = np.where(statmap_data > stat_min, statmap_data, 0)

                    # --- Coronal (top row) ---
                    if plot_mip:
                        y_slice = statmap_data.shape[1] // 2
                        mip_cor = np.max(stat_thresh, axis=1)
                        mip_cor = mip_cor[x_min:x_max,z_min:z_max]
                    else:
                        y_slice = 69
                        mip_cor = stat_thresh
                        mip_cor = mip_cor[x_min:x_max,y_slice, z_min:z_max]
                    mip_cor = np.where(mip_cor > stat_min, mip_cor, np.nan)
                    mip_cor=mip_cor.T
                    template_cor = template_data[x_min:x_max, y_slice, z_min:z_max].T

                    ax_cor = fig.add_subplot(gs[row_start, col_start + map_idx])
                    ax_cor.imshow(template_cor, cmap="gray", origin="lower",aspect='auto')
                    if underlay_data is not None:
                        ax_cor.imshow(underlay_data[x_min:x_max, y_slice, z_min:z_max].T, cmap="gray", origin="lower",aspect='auto')
                    
                    ax_cor.imshow(mip_cor, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max,aspect='auto')
                    ax_cor.axvline(x=(x_max-x_min)/2, color="white", linestyle="--", linewidth=0.5, alpha=0.6)
                    ax_cor.axis("off")

                    if map_idx == 0:
                        x_center = 1.7 
                        y_top = 1.2   
                        ax_cor.text(x_center, y_top, f"ID #{subj_idx + 1}", ha='center', va='bottom', fontsize=8, fontweight='black', transform=ax_cor.transAxes, fontname="Arial")
                        line_y = 1.2
                        ax_cor.hlines(y=line_y, xmin=0.15, xmax=3, colors='black', linewidth=0.8, transform=ax_cor.transAxes, clip_on=False)
        
                        ax_cor.set_title(titles[0], color="black",  fontsize=6, fontname="Arial")
                    if map_idx == 1:
                        ax_cor.set_title(f"{titles[1]}\nrun-01", color="black",  fontsize=6, fontname="Arial",y=0.94)
                    if map_idx == 2 and i_fname != None:
                        ax_cor.set_title(f"{titles[2]}\nrun-02", color="black",  fontsize=6, fontname="Arial",y=0.94)
  
                        

                    # Orientation labels only for first participant
                    if subj_idx == 0 and map_idx == 0:
                        ax_cor.text(0.05, 0.05, "L", transform=ax_cor.transAxes, color="white", fontsize=5, ha="left", va="bottom")
                        ax_cor.text(0.95, 0.05, "R", transform=ax_cor.transAxes, color="white", fontsize=5, ha="right", va="bottom")

                    # --- Axial (bottom row) ---
                    row_axi = row_start + 1
                    if plot_mip:
                        z_slice = statmap_data.shape[2] // 2
                    else:
                        z_slice = 260

                    # Crop for smaller axial view
                    crop_x = 30
                    crop_y = 30
                    x0 = statmap_data.shape[0] // 2
                    y0 = statmap_data.shape[1] // 2
                    x_min, x_max = x0 - crop_x, x0 + crop_x
                    y_min, y_max = y0 - crop_y, y0 + crop_y
                    template_axi = template_data[x_min:x_max, y_min:y_max, z_slice].T
                    
                    if plot_mip:
                        stat_crop = stat_thresh[x_min:x_max, y_min:y_max, :]
                        mip_axi = np.max(stat_crop, axis=2).T
                    else:
                        stat_crop = stat_thresh[x_min:x_max, y_min:y_max, z_slice]
                        mip_axi=stat_crop.T
                    mip_axi = np.where(mip_axi > stat_min, mip_axi, np.nan)

                    ax_axi = fig.add_subplot(gs[row_start + 1, col_start + map_idx])
                    ax_axi.imshow(template_axi, cmap="gray", origin="lower",aspect='auto')
                    if underlay_data is not None:
                        ax_axi.imshow(underlay_data[x_min:x_max, y_min:y_max, z_slice].T,
                                    cmap="gray", alpha=0.3, origin="lower")
                    ax_axi.imshow(mip_axi, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max,aspect='auto')
                    ax_axi.axis("off")

                    if subj_idx == 0 and map_idx == 0:
                        ax_axi.text(0.02, 0.5, "L", transform=ax_axi.transAxes, color="white", fontsize=5, ha="left", va="center")
                        ax_axi.text(0.98, 0.5, "R", transform=ax_axi.transAxes, color="white", fontsize=5, ha="right", va="center")
                        ax_axi.text(0.5, 0.98, "A", transform=ax_axi.transAxes, color="white", fontsize=5, ha="center", va="top")
                        ax_axi.text(0.5, 0.02, "P", transform=ax_axi.transAxes, color="white", fontsize=5, ha="center", va="bottom")
                    
                    # ---- Add colorbar only for the first participant and first map -----
                    gap_col_idx = 2
                    row_for_cbar = 0
                    cbar_ax = fig.add_subplot(gs[row_for_cbar, gap_col_idx])
                    cbar_ax.axis("off")

                    # positions of the two colorbars
                    pos_winter = [14.45, -3.8, 0.3, 0.8]
                    pos_autumn = [14.80, -3.8, 0.3, 0.8]

                    ax_winter = cbar_ax.inset_axes(pos_winter)
                    ax_autumn = cbar_ax.inset_axes(pos_autumn)

                    norm = plt.Normalize(vmin=stat_min, vmax=stat_max)

                    sm_winter = plt.cm.ScalarMappable(cmap="winter", norm=norm)
                    sm_winter.set_array([])

                    sm_autumn = plt.cm.ScalarMappable(cmap="autumn", norm=norm)
                    sm_autumn.set_array([])

                    cbar_winter = fig.colorbar(sm_winter, cax=ax_winter)
                    cbar_autumn = fig.colorbar(sm_autumn, cax=ax_autumn)

                    for cbar in [cbar_winter, cbar_autumn]:
                        cbar.ax.set_yticks([])
                        cbar.ax.set_frame_on(False)

                    cbar_winter.ax.text(-1.55, 0.5, f"z-score (uncorr)",rotation=90, fontsize=6,va="center", ha="right", transform=cbar.ax.transAxes)
                    cbar_winter.ax.text(0.5, -0.1, f"{stat_min:.1f}", fontsize=6,va="center", ha="right", transform=cbar.ax.transAxes)
                    cbar_winter.ax.text(0.5, 1.1, f"{stat_max:.1f}", fontsize=6, va="center", ha="right", transform=cbar.ax.transAxes)
                    

            # --- Save figure ---
            fig.savefig(output_fname, dpi=300)
            plt.close(fig)
        
        else:
            print("First level figure already exists, put redo=True to regenerate the figure")
        
        return output_fname
    
    def plot_fmri_maps(self, i_fnames=None, output_fname=None, stat_min=2.3, stat_max=5,titles = ["shimBase", "shimSlice"],
                  background_fname=None, cbar_label='t-value', cmap="autumn",z_slices=[None,None],
                  mask_fname=None, underlay_fname=None, task_name=None, verbose=True, redo=False):

        if output_fname is None:
            raise ValueError("output_dir is empty")
        if i_fnames is None or len(i_fnames) == 0:
            raise ValueError("i_fnames_pair is empty")
        if background_fname is None:
            raise ValueError("Please provide PAM50 template filename")

        assert len(i_fnames) in [1, 2], f"Expected 1 or 2 maps, got {len(i_fnames)}"
        n_maps = len(i_fnames)

        # --- Figure and gridspec ---
        fig = plt.figure(figsize=(n_maps, 3.5))  # width scales with number of maps
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)

        height_ratios = [6.5, 2.3]
        gs = fig.add_gridspec(nrows=2, ncols=2 + n_maps,
                            height_ratios=height_ratios,
                            width_ratios=[0.2, 0.1] + [1] * n_maps,
                            hspace=0.01, wspace=0.05)

        # --- Load template, mask, and underlay ---
        template_img = nib.load(background_fname)
        template_data = nib.as_closest_canonical(template_img).get_fdata()

        if underlay_fname is not None:
            underlay_data = nib.as_closest_canonical(nib.load(underlay_fname)).get_fdata()

        # --- Plotting ---
        num_voxels_list = []
        values_list = []

        for i, fname in enumerate(i_fnames):
            stat_img = nib.as_closest_canonical(nib.load(fname))
            statmap_data = stat_img.get_fdata()

            num_voxels_list.append(np.nansum(statmap_data > stat_min))
            values_list.append(statmap_data.flatten())

            # --- Coronal slice ---
            x_min, x_max = 35, 105
            z_min, z_max = 200, 333
            y_slice = 72
            cor_slice = statmap_data[x_min:x_max, y_slice, z_min:z_max]
            cor_slice = np.where(cor_slice > stat_min, cor_slice, np.nan)
            cor_slice = cor_slice.T

            ax_cor = fig.add_subplot(gs[0, i+2])
            template_cor = template_data[x_min:x_max, y_slice, z_min:z_max].T
            ax_cor.imshow(template_cor, cmap="gray", origin="lower", aspect="auto")

            # if there are only nan or 0 values, skip plotting the statmap to avoid showing a blank colorbar
            if np.nansum(cor_slice) == 0:
                print(f"warning: no suprathreshold voxels found for {titles[i]} (y={y_slice} coronal slice), skipping statmap overlay")
            else:
               im_cor = ax_cor.imshow(cor_slice, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max, aspect="auto")
            ax_cor.text(0.5, 0.01, f"y={y_slice}", color="white", fontsize=5,
                        ha="center", va="bottom", transform=ax_cor.transAxes)
            ax_cor.axis("off")
            ax_cor.set_title(titles[i], color="black", fontweight='bold', fontsize=7, fontname="Arial")

            # --- Axial slice ---
            crop_x = 30
            crop_y = 30
            x0 = statmap_data.shape[0] // 2
            y0 = statmap_data.shape[1] // 2
            x_min_axi, x_max_axi = x0 - crop_x, x0 + crop_x
            y_min_axi, y_max_axi = y0 - crop_y, y0 + crop_y

            crop_data = statmap_data[x_min_axi:x_max_axi, y_min_axi:y_max_axi, :]
            if z_slices[i] is None:
               z_slice = np.argmax(np.nanmax(crop_data, axis=(0, 1)))
               if np.nansum(cor_slice) == 0:
                   z_slice=258
            else:
                z_slice = z_slices[i]

            axi_slice = crop_data[:, :, z_slice]
            axi_slice = np.where(axi_slice > stat_min, axi_slice, np.nan)
            axi_slice = axi_slice.T

            ax_axi = fig.add_subplot(gs[1, i+2])
            template_axi = template_data[x_min_axi:x_max_axi, y_min_axi:y_max_axi, z_slice].T
            ax_axi.imshow(template_axi, cmap="gray", origin="lower", aspect="auto")

            if underlay_fname:
                underlay_axi = underlay_data[x_min_axi:x_max_axi, y_min_axi:y_max_axi, z_slice].T
                ax_axi.imshow(underlay_axi, cmap="gray", origin="lower", aspect="auto", alpha=0.1)

            im_axi = ax_axi.imshow(axi_slice, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max, aspect="auto")
            ax_axi.axis("off")
            ax_axi.text(0.5, 0.01, f"z={z_slice}", color="white", fontsize=5,
                        ha="center", va="bottom", transform=ax_axi.transAxes)
            
            if np.nansum(cor_slice) != 0:
                ax_cor.axhline(y=z_slice - z_min, color='white', linestyle='--', linewidth=0.8, alpha=0.7)

            # orientation labels only on first map
            if i == 0:
                ax_cor.text(0.05, 0.05, "L", transform=ax_cor.transAxes, color="white", fontsize=7, ha="left", va="bottom")
                ax_cor.text(0.95, 0.05, "R", transform=ax_cor.transAxes, color="white", fontsize=7, ha="right", va="bottom")
                ax_axi.text(0.02, 0.5, "L", transform=ax_axi.transAxes, color="white", fontsize=7, ha="left", va="center")
                ax_axi.text(0.98, 0.5, "R", transform=ax_axi.transAxes, color="white", fontsize=7, ha="right", va="center")
                ax_axi.text(0.5, 0.90, "A", transform=ax_axi.transAxes, color="white", fontsize=7, ha="center", va="top")
                ax_axi.text(0.5, 0.12, "P", transform=ax_axi.transAxes, color="white", fontsize=7, ha="center", va="bottom")

        # -- Shared colorbar
        cbar = self.plot_colorbar(
            fig=fig,
            stat_min=stat_min,
            stat_max=stat_max,
            cmap=cmap,
            label=cbar_label,
            left=0.05 if n_maps == 2 else 0.11 ,
            bottom=0.05, width=0.04, height=0.15 
        )

        # -- Spinal levels
        ax_levels, ax_levels_txt = self.plot_spinal_levels(
            fig=fig,
            gs=gs,
            ax_cor=ax_cor,
            cor_slice_shape=cor_slice.shape,
            z_min=z_min,
            z_max=z_max,
            n_maps=n_maps
        )

        plt.savefig(output_fname, transparent=True, dpi=300)
        plt.close(fig)

        return output_fname

    def plot_spinal_levels(self, fig, gs, ax_cor, cor_slice_shape, z_min, z_max,n_maps):
        """
        Plot spinal level color bands and segmental labels on a figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        gs : matplotlib.gridspec.GridSpec
        ax_cor : matplotlib.axes.Axes
            Coronal axis used as reference for text transforms
        cor_slice_shape : tuple
            Shape of the coronal slice (height, width) — used to init data array
        z_min : int
            Minimum z index of the coronal crop
        z_max : int
            Maximum z index of the coronal crop
        """

        spinal_levels = {
            5: range(300, 333),  # C5
            6: range(269, 300),  # C6
            7: range(238, 269),  # C7
            8: range(206, 238),  # C8
            9: range(172, 206)   # T1
        }

        data_spinal_levels = np.zeros((cor_slice_shape[0], z_max - z_min))

        for level, z_range in spinal_levels.items():
            z_start = max(z_range.start, z_min)
            z_end = min(z_range.stop, z_max)
            if z_start >= z_end:
                continue
            z_inds = np.arange(z_start, z_end) - z_min
            data_spinal_levels[:, z_inds] = level

        data_spinal_alpha = np.zeros_like(data_spinal_levels, dtype=float)
        data_spinal_alpha[data_spinal_levels > 0] = 1

        data_spinal_levels_2 = np.copy(data_spinal_levels).astype(float)
        data_spinal_levels_2[data_spinal_levels % 2 == 0] = 0.5
        data_spinal_levels_2[data_spinal_levels % 2 == 1] = 0.75

        # --- Color bands
        ax_levels = fig.add_subplot(gs[0, 1])
        ax_levels.axis("off")
        ax_levels.imshow(data_spinal_levels_2.T, cmap="gray", vmin=0, vmax=1,
                        alpha=data_spinal_alpha.T, origin='lower', aspect='auto')

        # --- Segmental labels
        ax_levels_txt = fig.add_subplot(gs[0, 0])
        ax_levels_txt.axis("off")
        x_pos = -1.3 if n_maps == 2 else -0.25

        labels = [("C5", 0.86), ("C6", 0.63), ("C7", 0.4), ("C8", 0.18), ("", 0.1)]
        for label, y_pos in labels:
            ax_levels_txt.text(x_pos, y_pos, label, transform=ax_cor.transAxes,
                            color="black", fontsize=6, ha="center", va="center",
                            fontweight='bold', fontname="Arial")

        return ax_levels, ax_levels_txt

    def plot_colorbar(self, fig, stat_min, stat_max, cmap='autumn', 
                  left=0.03, bottom=0.05, width=0.02, height=0.15,
                  label='t-value', fontsize=6):
        """
        Plot a shared colorbar on a figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        stat_min : float
            Minimum value of the colorbar
        stat_max : float
            Maximum value of the colorbar
        cmap : str
            Colormap name (default: 'autumn')
        left : float
            Left position of the colorbar axes (default: 0.03)
        bottom : float
            Bottom position of the colorbar axes (default: 0.05)
        width : float
            Width of the colorbar axes (default: 0.02)
        height : float
            Height of the colorbar axes (default: 0.15)
        label : str
            Label of the colorbar (default: 't-score')
        fontsize : int
            Font size for label and tick text (default: 6)

        Returns
        -------
        cbar : matplotlib.colorbar.Colorbar
        """

        cbar_ax = fig.add_axes([left, bottom, width, height])
        norm = plt.Normalize(vmin=stat_min, vmax=stat_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(label, fontsize=fontsize, labelpad=1.5, fontweight='bold', fontname="Arial")
        cbar.ax.set_yticks([])
        cbar.ax.text(1.5, 1.1, f"{stat_max:.1f}", fontsize=fontsize, va='center', ha='right',
                    color='black', transform=cbar.ax.transAxes)
        cbar.ax.text(1.5, -0.12, f"{stat_min:.1f}", fontsize=fontsize, va='center', ha='right',
                    color='black', transform=cbar.ax.transAxes)
        cbar.ax.set_frame_on(False)

        return cbar
    
    def bar_plot(self,csv_pair=None,metric="nonzero_voxels",output_fname=None, colors = None, maps_name=None, figsize=(1.8, 2.5),width=0.5, alpha=0.8):
        """
        Plot a bar chart of metrics loaded from a pair of CSV files.

        Parameters
        ----------
        csv_pair : list
            List of two CSV filenames (output of extract_metrics)
        output_fname : str, optional
            Path to save the figure. If None, figure is returned without saving.
        colors : list, optional
            Bar colors (default: ["#43BA8C", "#F5AD27"])
        maps_name : list, optional
            X-tick labels (default: ["shimBase", "shimSlice"])
        figsize : tuple
            Figure size (default: (1.5, 2))
        width : float
            Bar width (default: 0.5)
        alpha : float
            Bar transparency (default: 0.7)
        metric : str
            Column name to plot from the CSV (default: "nonzero_voxels")

        """
        if not os.path.exists(output_fname):
            if csv_pair is None:
                raise ValueError("Please provide a list of two CSV filenames.")
        
            if colors is None:
                colors = ["#ADA8A8","#ED263F"]
            if maps_name is None:
                maps_name = ["shimBase", "shimSlice"]

            # --- Load metric from each CSV ---
            values = [pd.read_csv(f)[metric].values[0] for f in csv_pair]

            # --- Plot ---
            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.25)

            ax.bar(range(len(values)), values, color=colors, width=0.5, alpha=alpha)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(
                [maps_name[i] for i in range(len(values))],
                rotation=45, fontsize=8, fontweight='bold', fontname="Arial", ha='right')
            ax.set_ylabel("# significant voxels (GLM)", fontsize=8, fontweight='bold', fontname="Arial")
            ax.tick_params(axis='y', labelsize=7)
            #ax.yaxis.set_label_coords(-0.9, 0.5)
            ax.tick_params(axis='y', which='both', pad=2)
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            plt.savefig(output_fname,transparent=True, dpi=300)
            plt.close(fig)
        
        return output_fname

    def plot_dist(self, csv_pair=None,output_fname=None,colors=None, maps_name=None,bins=100,figsize=(1.8, 2.3),width=0.5, alpha=0.8, redo=False):
        """
        Plot a bar chart of suprathreshold voxel counts as a standalone figure.

        Parameters
        ----------
        csv_pair : list
            List of two CSV filenames 
        output_fname : str, optional
            Path to save the figure. If None, figure is returned without saving.
        colors : list, optional
            Bar colors (default: ["#43BA8C", "#F5AD27"])
        maps_name : list, optional
            X-tick labels (default: ["baseShim", "SliceShim"])
        figsize : tuple
            Figure size (default: (1.5, 2))
        width : float
            Bar width (default: 0.5)
        alpha : float
            Bar transparency (default: 0.7)

        Returns
        -------
        output filename
        """

        if not os.path.exists(output_fname):
            if csv_pair is None:
                raise ValueError("Please provide a list of two CSV filenames.")
        
            if colors is None:
                colors = ["#ADA8A8","#D61532"]
            if maps_name is None:
                maps_name = ["shimBase", "shimSlice"]

            values_list = [pd.read_csv(f)["voxels_values"].values for f in csv_pair]
            # --- Plot ---
            fig, ax = plt.subplots(figsize=figsize)

            for i, values in enumerate(values_list):
                if i==1:
                    alpha=0.9
                values_clean = values[values != 0]
                ax.hist(values_clean, bins=bins, color=colors[i], alpha=alpha,
                        label=maps_name[i], density=False)

            ax.set_xlabel("t-value", fontsize=8, fontweight='bold', fontname="Arial")
            ax.set_ylabel("# significant voxels (GLM)", fontsize=8, fontweight='bold', fontname="Arial")
            ax.tick_params(axis='both', labelsize=6)
            ax.legend(fontsize=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            plt.savefig(output_fname,transparent=True, dpi=300)
            plt.close(fig)
        
        return output_fname

    def boxplots(self, csv_file=None,df=None,output_fname=None, stats_file=None,x_data=None, x_order=None, y_data=None, hue=None, hue_order=None,specify_y_label=None,output_dir=None, color=None, indiv_values=False,indiv_hue=None, indiv_color=None, plot_legend=True, output_tag='', ymin=6, ymax=17,height=2.5,aspect=0.6, invers_axes=False,indiv=False, group=False, redo=False):
        '''
        Create matrix of correlation boxplots with matching box outline and whisker colors.
        '''

        if not os.path.exists(output_fname) or redo:
            if csv_file:
                df = pd.read_csv(csv_file)
            

            # Set style and default palette if not provided
            if color is None:
                color = ["#ADA8A8","#ED263F"]
            if hue is None:
                hue = x_data
                hue_order = x_order
            
            if invers_axes:
                x_data_f=y_data
                y_data_f=x_data
            else:
                x_data_f=x_data
                y_data_f=y_data


            #--- Create the boxplot
            g = sns.catplot(
                    x=x_data_f, 
                    y=y_data_f, 
                    data=df,  
                    kind="box",  
                    linewidth=2, 
                    #color=color,  # Use the provided palette
                    medianprops=dict(color="white",alpha=0.5),  # Set median line color to white
                    #hue=None,
                    order=x_order, 
                    #hue_order=None,
                    fliersize=0,  # Remove outliers' markers
                    height=height,
                    aspect=aspect,
                    legend=plot_legend
                )

            # Apply custom outline and whisker colors to match the palette
            for ax in g.axes.flat:
                # Add a horizontal line at y=0
                if invers_axes:
                    ax.axvline(0, color='grey', linestyle='--', linewidth=1)
                else:
                    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
                
                # Change whisker colors
                for i, box in enumerate(ax.patches):  # Access the box patches
                    category = df[x_data_f].unique()[i % len(df[x_data_f].unique())]  # Use modulus to loop over categories
                    color_index = list(df[x_data_f].unique()).index(category)  # Get the index of the category in the unique list
                    
                    # Set the box color and alpha
                    box.set_color(color[color_index])  # Set box color
                    box.set_alpha(0.4)  # Set alpha for the box
                    
                    whisker_lines = ax.lines[i * 6:i * 6 + 2]  # Whiskers are the first two lines for each box
                    for whisker in whisker_lines:
                        whisker.set_color(color[color_index])  # Set the whisker color
                        whisker.set_alpha(0.4)  # Set alpha for whiskers

                    cap_lines = ax.lines[i * 6 + 2:i * 6 + 4]  # Caps are the next two lines for each box
                    for cap in cap_lines:
                        cap.set_color(color[color_index])  # Set the cap color
                        cap.set_alpha(0.4)


                    # Loop through each box and set outline color
                    # Get the current category for the box
                    category = df[x_data_f].unique()[i % len(df[x_data_f].unique())]  # Use modulus to loop over categories
                    color_index = list(df[x_data_f].unique()).index(category)  # Get the index of the category in the unique list

                    # Set the box color
                    box.set_color(color[color_index])  # Set box color
                    
                    # Get the bounding box by extracting the vertices of the path
                    vertices = box.get_path().vertices
                    x_pos = vertices[:, 0].min()  # Minimum x value
                    y_pos = vertices[:, 1].min()  # Minimum y value
                    box_width = vertices[:, 0].max() - x_pos  # Width
                    box_height = vertices[:, 1].max() - y_pos  # Height

                    # Create a new outline with lower alpha for the edge
                    outline = plt.Rectangle(
                        (x_pos, y_pos),  # Position as a tuple
                        box_width,  # Width
                        box_height,  # Height
                        fill=False,  # No fill for the outline
                        edgecolor=color[color_index],  # Same color as the box
                        lw=0,  # Line width
                        alpha=0.4  # Set alpha for transparency of the outline
                    )
                    ax.add_patch(outline)  # Add the outline to the axis

            # ------- Add individual points if requested
            if indiv_values:
                sns.stripplot(
                    x=x_data_f, 
                    y=y_data_f, 
                    data=df, 
                    hue=hue, 
                    hue_order=hue_order,
                    size=5,
                    palette=indiv_color if indiv_color else color,
                    #palette=palette, 
                    linewidth=0, 
                    alpha=0.7,
                    edgecolor='white',
                    jitter=False #set 0.25 to add jitter between individual points
                )

                # Draw lines between points from the same individual
                ax = g.axes.flat[0]

                x_positions = {}
                collections = [c for c in ax.collections if isinstance(c, plt.matplotlib.collections.PathCollection)]  # Get the jittered x positions from the stripplot collections

                for coll_idx, collection in enumerate(collections):
                    offsets = collection.get_offsets()
                    category = x_order[coll_idx] if x_order else df[x_data_f].unique()[coll_idx]
                    for x_pos, y_pos in offsets:
                        # Match y value back to the ID
                        matched = df[(df[x_data_f] == category) & (np.isclose(df[y_data_f], y_pos))]
                        if not matched.empty:
                            ind_id = matched.iloc[0]['IDs']
                            if ind_id not in x_positions:
                                x_positions[ind_id] = {}
                            x_positions[ind_id][category] = (x_pos, y_pos)

                # Draw lines using the recovered jittered positions
                for ind_id, coords in x_positions.items():
                    ordered_cats = [c for c in (x_order if x_order else df[x_data_f].unique()) if c in coords]
                    xs = [coords[c][0] for c in ordered_cats]
                    ys = [coords[c][1] for c in ordered_cats]
                    ax.plot(xs, ys, color='grey', alpha=1,linestyle='--', linewidth=1, zorder=1) #linestyle='--',
            
            # ------- Add significance annotation if stats_file provided
            if stats_file is not None:
                stats_df = pd.read_csv(stats_file)
                ax = g.axes.flat[0]

                # Get x positions of the two conditions
                if x_order:
                    x1 = x_order.index(stats_df['cond1'].values[0])
                    x2 = x_order.index(stats_df['cond2'].values[0])
                else:
                    cats = list(df[x_data_f].unique())
                    x1 = cats.index(stats_df['cond1'].values[0])
                    x2 = cats.index(stats_df['cond2'].values[0])

                stars = stats_df['significance'].values[0]

                # Draw bracket
                y_bracket = ymax * 0.97  # just below the top
                y_tip     = y_bracket - (ymax - ymin) * 0.02
                bracket_color = 'black'

                ax.plot([x1, x1, x2, x2], 
                        [y_tip, y_bracket, y_bracket, y_tip],
                        color=bracket_color, linewidth=1)
                ax.text((x1 + x2) / 2, y_bracket + (ymax - ymin) * 0.01,
                        stars,fontname="Arial",
                        ha='center', va='bottom',
                        fontsize=7, color=bracket_color)
            
            ax.set_xlabel('')
            y_label=specify_y_label if specify_y_label else y_data

            ax.set_ylabel(y_label, fontsize=8, fontname="Arial",fontweight='bold')
            ax.tick_params(axis='y', labelsize=7)
            

            if output_tag:
                g.set(title=output_tag)

            if invers_axes:
                g.set(xlim=(ymin, ymax))
            else:
                g.set(ylim=(ymin, ymax))
            sns.despine(offset=5, trim=True)
            if plot_legend:
                g.add_legend()
            else:
                plt.legend([],[], frameon=False)
            
            ax.set_xticks(range(len(df[x_data_f].unique())))
            ax.set_xticklabels(x_order if x_order else df[x_data_f].unique(), 
                   rotation=45, fontsize=8, fontweight='bold', fontname="Arial", ha='right')
            
            # Save the figure if requested
            plt.tight_layout()
            plt.savefig(output_fname, dpi=300, transparent=True)
            plt.close()
        
        return output_fname

    def combine_plots(self, output_fname, map_files, graph_files,
                  map_titles=["SNR", "GLM"], graph_titles=None,
                  figsize=(3.15, 10), graph_width_scale=0.9, redo=False):

        assert len(map_files) == 2, f"Expected 2 map_files, got {len(map_files)}"
        assert len(graph_files) == 4, f"Expected 4 graph_files, got {len(graph_files)}"

        if not os.path.exists(output_fname) or redo:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, 4,
                                width_ratios=[1.4, 1.4, 1, 1],
                                hspace=0.05, wspace=0.01)

            label_y = 1.02  # same y for all labels

            # --- Col 1 & 2: maps spanning both rows ---
            for i, fname in enumerate(map_files):
                ax = fig.add_subplot(gs[:, i])
                img = mpimg.imread(fname)
                h, w = img.shape[:2]
                ax.imshow(img, aspect='auto')
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.axis('off')

                ax.text(0.0, label_y, f"{chr(65+i)}.",
                        ha='left', va='bottom',
                        fontsize=8, fontweight='bold', fontname="Arial",
                        transform=ax.transAxes,
                        clip_on=False)

                if map_titles:
                    ax.text(0.5, label_y,
                            map_titles[i],
                            ha='center', va='bottom',
                            fontsize=7, fontweight='bold', fontname="Arial",
                            transform=ax.transAxes,
                            clip_on=False)

            # --- Col 3 & 4, Row 1 & 2: graphs in 2x2 grid ---
            for i, fname in enumerate(graph_files):
                row = i // 2       # 0, 0, 1, 1
                col = 2 + (i % 2)  # 2, 3, 2, 3

                ax = fig.add_subplot(gs[row, col])
                img = mpimg.imread(fname)
                margin = (1 - graph_width_scale) / 2
                ax_inner = ax.inset_axes([margin, 0, graph_width_scale, 1])
                ax_inner.imshow(img, aspect='auto')
                ax_inner.axis('off')
                ax.axis('off')

                if graph_titles:
                    ax_inner.set_title(graph_titles[i], fontsize=7,
                                    fontweight='bold', fontname="Arial")

                # only label top row graphs (i=0,1) to align with map labels
                if row == 0:
                    ax.text(0.0, label_y, f"{chr(65+len(map_files)+i)}.",  # C, D
                            ha='left', va='bottom',
                            fontsize=8, fontweight='bold', fontname="Arial",
                            transform=ax.transAxes,
                            clip_on=False)

            fig.subplots_adjust(wspace=0.05, hspace=0.05,
                                left=0.01, right=0.99, top=0.93, bottom=0.01)
            plt.savefig(output_fname, dpi=300, transparent=True, bbox_inches='tight')
            plt.close()
            
       