import os
import glob
import json
import numpy as np
import pandas as pd
import nibabel as nib

#matplotlib
import matplotlib.pyplot as plt
import matplotlib

# nilearn
from nilearn.plotting import plot_design_matrix
from nilearn.image import resample_to_img
from nilearn.image import smooth_img
from nibabel.processing import resample_from_to

#mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
     
    def plot_first_level_maps(self, i_fnames=None, output_fname=None,titles=["shimBase","shimSlice",""],cmap="autumn",stat_min=1.6, stat_max=4,background_fname=None,mask_fname=None, underlay_fname=None,task_name=None,plot_mip=True, verbose=True, redo=False,n_cols=5):
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
    
    def plot_two_maps(self, i_fnames_pair=None, output_fname=None,stat_min=2.3, stat_max=5,background_fname=None,cbar_label='t-value',cmap="autumn",mask_fname=None, underlay_fname=None,task_name=None, verbose=True, redo=False):
        """
        Plot second-level statistical maps for two maps.


        To do: 
        - plot GM
        - add spinal levels in the coronal view 

        """
        if output_fname is None:
            raise ValueError("output_dir is empty")
        if i_fnames_pair is None or len(i_fnames_pair) == 0:
            raise ValueError("i_fnames_pair is empty")
        if background_fname is None :
            raise ValueError("Please provide PAM50 template filename")

        # --- Figure and gridspec ---
        fig = plt.figure(figsize=(2, 3.5))
        fig.subplots_adjust(left=0.01,right=0.99,top=0.95,bottom=0.01)
        
        height_ratios = [6.5, 2.3]  # coronal, axial
        
        gs = fig.add_gridspec(nrows=2, ncols=4, 
                              height_ratios=height_ratios,
                               width_ratios=[0.2,0.1,1,1], hspace=0.01, wspace=0.05)


        # --- Load template, mask, and underlay ---
        template_img = nib.load(background_fname)
        template_data = nib.as_closest_canonical(template_img).get_fdata()
        
        if underlay_fname is not None:
            underlay_data = nib.as_closest_canonical(nib.load(underlay_fname)).get_fdata()
        
        # --- Plotting ---
        num_voxels_list=[];values_list=[]

        for i, fname in enumerate(i_fnames_pair):
            stat_img = nib.as_closest_canonical(nib.load(fname))
            statmap_data = stat_img.get_fdata()

            # Count suprathreshold voxels
            num_voxels_list.append(np.nansum(statmap_data > stat_min))
            values_list.append(statmap_data.flatten()) 

            # --- Coronal slice ---
            x_min, x_max = 35, 105
            z_min, z_max = 172, 333
            y_slice = np.unravel_index(np.nanargmax(statmap_data), statmap_data.shape)[1]
            

            # Find y_slice along y-axis with maximum intensity
            crop_data = statmap_data[x_min:x_max, :, z_min:z_max]
            y_slice = np.argmax(np.nanmax(crop_data, axis=(0, 2)))  # max over x and z, returns y index
            cor_slice = statmap_data[x_min:x_max,y_slice,z_min:z_max]
            cor_slice = np.where(cor_slice > stat_min, cor_slice, np.nan)
            cor_slice=cor_slice.T

            ax_cor = fig.add_subplot(gs[0, i+2])
            template_cor = template_data[x_min:x_max, y_slice, z_min:z_max].T
            ax_cor.imshow(template_cor, cmap="gray", origin="lower", aspect="auto")
            im_cor = ax_cor.imshow(cor_slice, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max, aspect="auto")
            ax_cor.text(0.5, 0.01, f"y={y_slice}", color="white", fontsize=5,ha="center", va="bottom", transform=ax_cor.transAxes)
            
            ax_cor.axis("off")

            # --- Axial slice ---
            crop_x = 30
            crop_y = 30
            x0 = statmap_data.shape[0] // 2
            y0 = statmap_data.shape[1] // 2
            x_min, x_max = x0 - crop_x, x0 + crop_x
            y_min, y_max = y0 - crop_y, y0 + crop_y
            
            crop_data = statmap_data[x_min:x_max, y_min:y_max, :]
            z_slice = np.argmax(np.nanmax(crop_data, axis=(0, 1)))  # max over x and z, returns y index
            axi_slice = crop_data[:, :, z_slice]
            axi_slice = np.where(axi_slice > stat_min, axi_slice, np.nan)
            axi_slice=axi_slice.T

            ax_axi = fig.add_subplot(gs[1, i+2])
            template_axi = template_data[x_min:x_max, y_min:y_max, z_slice].T
            ax_axi.imshow(template_axi, cmap="gray", origin="lower", aspect="auto")

            if underlay_fname:
                underlay_axi = underlay_data[x_min:x_max, y_min:y_max, z_slice].T
                ax_axi.imshow(underlay_axi, cmap="gray", origin="lower", aspect="auto",alpha=0.1)

            im_axi = ax_axi.imshow(axi_slice, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max, aspect="auto")
            ax_axi.axis("off")
            ax_axi.text(0.5, 0.01, f"z={z_slice}", color="white", fontsize=5,ha="center", va="bottom", transform=ax_axi.transAxes)
            ax_cor.axhline(y=z_slice - z_min, color='white', linestyle='--', linewidth=0.8, alpha=0.7)

            if i==0:
                ax_cor.set_title(f"shimBase", color="black", fontweight='bold', fontsize=7, fontname="Arial")
                ax_cor.text(0.05, 0.05, "L", transform=ax_cor.transAxes, color="white", fontsize=7, ha="left", va="bottom")
                ax_cor.text(0.95, 0.05, "R", transform=ax_cor.transAxes, color="white", fontsize=7, ha="right", va="bottom")
                ax_axi.text(0.02, 0.5, "L", transform=ax_axi.transAxes, color="white", fontsize=7, ha="left", va="center")
                ax_axi.text(0.98, 0.5, "R", transform=ax_axi.transAxes, color="white", fontsize=7, ha="right", va="center")
                ax_axi.text(0.5, 0.90, "A", transform=ax_axi.transAxes, color="white", fontsize=7, ha="center", va="top")
                ax_axi.text(0.5, 0.12, "P", transform=ax_axi.transAxes, color="white", fontsize=7, ha="center", va="bottom")

            else:
                ax_cor.set_title(f"shimSlice", color="black", fontweight='bold', fontsize=7, fontname="Arial")

        # -- Shared colorbar
        cbar = self.plot_colorbar(
            fig=fig,
            stat_min=stat_min,
            stat_max=stat_max,
            cmap=cmap,
            label=cbar_label,
            left=0.05, bottom=0.05, width=0.04, height=0.15
        )

        # -- plot spinal levels at the very left side
        ax_levels, ax_levels_txt = self.plot_spinal_levels(
        fig=fig,
        gs=gs,
        ax_cor=ax_cor,
        cor_slice_shape=cor_slice.shape,
        z_min=172,
        z_max=333
        )
        
        out_file=os.path.join(output_fname)
        plt.savefig(out_file, dpi=300)
        plt.close(fig)


    def plot_ICC_maps(self, i_fname=None, output_fname=None,stat_min=0.5, stat_max=0.9,background_fname=None,cmap="autumn",mask_fname=None, underlay_fname=None,task_name=None, verbose=True, redo=False):
        """
        Plot second-level statistical maps for two maps.

        """
        if output_fname is None:
            raise ValueError("output_fname is empty")
        if i_fname is None or len(i_fname) == 0:
            raise ValueError("i_fnames_pair is empty")
        if background_fname is None :
            raise ValueError("Please provide PAM50 template filename")

        # --- Figure and gridspec ---
        fig = plt.figure(figsize=(1, 3))
        fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
        
        height_ratios = [6.5, 2.5]  # coronal, axial
        
        gs = fig.add_gridspec(nrows=2, ncols=3, 
                              height_ratios=height_ratios,
                               width_ratios=[0.2,0.1,1], hspace=0.01, wspace=0.05)


        # --- Load template, mask, and underlay ---
        template_img = nib.load(background_fname)
        template_data = nib.as_closest_canonical(template_img).get_fdata()
        
        if underlay_fname is not None:
            underlay_data = nib.as_closest_canonical(nib.load(underlay_fname)).get_fdata()
        
        # --- Plotting ---
        num_voxels_list=[];values_list=[]

        stat_img = nib.as_closest_canonical(nib.load(i_fname))
        statmap_data = stat_img.get_fdata()

        # --- Coronal slice ---
        x_min, x_max = 35, 105
        z_min, z_max = 172, 333

        # Find y_slice along y-axis with maximum intensity
        crop_data = statmap_data[x_min:x_max, :, z_min:z_max]
        y_slice = 72#np.argmax(np.nanmax(crop_data, axis=(0, 2)))  # max over x and z, returns y index
        cor_slice = statmap_data[x_min:x_max,y_slice,z_min:z_max]
        cor_slice = np.where(cor_slice > stat_min, cor_slice, np.nan)
        cor_slice=cor_slice.T

        ax_cor = fig.add_subplot(gs[0, 2])
        template_cor = template_data[x_min:x_max, y_slice, z_min:z_max].T
        ax_cor.imshow(template_cor, cmap="gray", origin="lower", aspect="auto")
        im_cor = ax_cor.imshow(cor_slice, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max, aspect="auto")
        ax_cor.text(0.5, 0.01, f"y={y_slice}", color="white", fontsize=5,ha="center", va="bottom", transform=ax_cor.transAxes)
            
        ax_cor.axis("off")


        # --- Axial slice ---
        crop_x = 30
        crop_y = 30
        x0 = statmap_data.shape[0] // 2
        y0 = statmap_data.shape[1] // 2
        x_min, x_max = x0 - crop_x, x0 + crop_x
        y_min, y_max = y0 - crop_y, y0 + crop_y
            
        crop_data = statmap_data[x_min:x_max, y_min:y_max, :]
        z_slice = 310#np.argmax(np.nanmax(crop_data, axis=(0, 1)))  # max over x and z, returns y index
        axi_slice = crop_data[:, :, z_slice]
        axi_slice = np.where(axi_slice > stat_min, axi_slice, np.nan)
        axi_slice=axi_slice.T

        ax_axi = fig.add_subplot(gs[1, 2])
        template_axi = template_data[x_min:x_max, y_min:y_max, z_slice].T
        underlay_axi = underlay_data[x_min:x_max, y_min:y_max, z_slice].T
        ax_axi.imshow(template_axi, cmap="gray", origin="lower", aspect="auto")
        ax_axi.imshow(underlay_axi, cmap="gray", origin="lower", aspect="auto",alpha=0.1)
            
        im_axi = ax_axi.imshow(axi_slice, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max, aspect="auto")
        ax_axi.axis("off")
        ax_axi.text(0.5, 0.01, f"z={z_slice}", color="white", fontsize=5,ha="center", va="bottom", transform=ax_axi.transAxes)
        ax_cor.axhline(y=z_slice - z_min, color='white', linestyle='--', linewidth=0.8, alpha=0.7)

        ax_cor.text(0.05, 0.05, "L", transform=ax_cor.transAxes, color="white", fontsize=7, ha="left", va="bottom")
        ax_cor.text(0.95, 0.05, "R", transform=ax_cor.transAxes, color="white", fontsize=7, ha="right", va="bottom")
        ax_axi.text(0.02, 0.5, "L", transform=ax_axi.transAxes, color="white", fontsize=7, ha="left", va="center")
        ax_axi.text(0.98, 0.5, "R", transform=ax_axi.transAxes, color="white", fontsize=7, ha="right", va="center")
        ax_axi.text(0.5, 0.90, "A", transform=ax_axi.transAxes, color="white", fontsize=7, ha="center", va="top")
        ax_axi.text(0.5, 0.12, "P", transform=ax_axi.transAxes, color="white", fontsize=7, ha="center", va="bottom")



        # -- Shared colorbar
        
        cbar_ax = fig.add_axes([0.08, 0.05, 0.08, 0.15])  # left, bottom, width, height
        norm = plt.Normalize(vmin=stat_min, vmax=stat_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('icc', fontsize=6, labelpad=1.5,fontweight='bold',fontname="Arial")
        cbar.ax.set_yticks([])
        cbar.ax.text(1.35, 1.1, f"{stat_max:.1f}", fontsize=6, va='center', ha='right', color='black', transform=cbar.ax.transAxes)
        cbar.ax.text(1.35, -0.12, f"{stat_min:.1f}", fontsize=6, va='center', ha='right', color='black', transform=cbar.ax.transAxes)
        cbar.ax.set_frame_on(False)

        # -- plot spinal levels at the very left side
        ax_levels = fig.add_subplot(gs[0, 1])
        ax_levels.axis("off") 
        spinal_levels = {5: range(300, 333),  # C5
                     6: range(269, 300),  # C6
                     7: range(238, 269),  # C7
                     8: range(206, 238),  # C8
                     9: range(172, 206)  # T1
                     } 
        data_spinal_levels = np.zeros((cor_slice.shape[0], z_max - z_min))  # height x width
        print(data_spinal_levels.shape)
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

        ax_levels.imshow(data_spinal_levels_2.T, cmap="gray", vmin=0, vmax=1, alpha=data_spinal_alpha.T, origin='lower', aspect='auto')

        # --- Add text for the segmental labels
        ax_levels_txt = fig.add_subplot(gs[0, 0])
        ax_levels_txt.axis("off")  # we only want labels and lines

        ax_levels_txt.text(-0.24, 0.9, "C5", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")
        ax_levels_txt.text(-0.24, 0.68, "C6", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")
        ax_levels_txt.text(-0.24, 0.49, "C7", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")
        ax_levels_txt.text(-0.24, 0.3, "C8", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")
        ax_levels_txt.text(-0.24, 0.1, "T1", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")

        out_file=os.path.join(output_fname)
        plt.savefig(out_file, dpi=300)
        plt.close(fig)

    def plot_voxel_count_bar(self, num_voxels_list,output_fname=None,colors=None, maps_name=None,figsize=(1.5, 2),width=0.5, alpha=0.7):
        """
        Plot a bar chart of suprathreshold voxel counts as a standalone figure.

        Parameters
        ----------
        num_voxels_list : list
            List of suprathreshold voxel counts per map
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
        fig : matplotlib.figure.Figure
        """
        if colors is None:
            colors = ["#43BA8C", "#F5AD27"]
        if maps_name is None:
            maps_name = ["baseShim", "SliceShim"]

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(range(len(num_voxels_list)), num_voxels_list,
            color=colors, width=width, alpha=alpha)
        ax.set_xticks(range(len(num_voxels_list)))
        ax.set_xticklabels(
            [maps_name[i] for i in range(len(num_voxels_list))],
            rotation=45, fontsize=6, fontweight='bold', fontname="Arial")
        ax.set_ylabel("# voxels", fontsize=6, fontweight='bold', fontname="Arial")
        ax.tick_params(axis='y', labelsize=6)
        ax.yaxis.set_label_coords(-0.9, 0.5)
        ax.tick_params(axis='y', which='both', pad=2)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if output_fname is not None:
            plt.savefig(output_fname, dpi=300)
            plt.close(fig)
        
        return fig

    def plot_spinal_levels(self, fig, gs, ax_cor, cor_slice_shape, z_min, z_max):
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

        labels = [("C5", 0.9), ("C6", 0.68), ("C7", 0.49), ("C8", 0.3), ("T1", 0.1)]
        for label, y_pos in labels:
            ax_levels_txt.text(-1.3, y_pos, label, transform=ax_cor.transAxes,
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