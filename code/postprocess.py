import os
import glob
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

# nilearn
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import non_parametric_inference
from nilearn.image import resample_to_img
from nilearn.image import smooth_img

from mpl_toolkits.axes_grid1 import make_axes_locatable
from preprocess import Preprocess_main, Preprocess_Sc
from nibabel.processing import resample_from_to
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import pingouin as pg


#####################################################
class Postprocess_main:
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

        # Create directories -------------------------------------------------------------------------------------
        for ID in self.participant_IDs:
            ID_first_level_dir=self.first_level_dir.format(ID)
            os.makedirs(ID_first_level_dir, exist_ok=True)

            # Create a folder for each task in participant folder
            if "design_exp" in self.config.keys():
                for ses_name in self.config["design_exp"]['ses_names']:
                    ses_dir=ses_name if int(self.config["design_exp"]["ses_nb"])>1 else ""
                    if "acq_names" in self.config["design_exp"].keys():
                        for task_name in self.config["design_exp"]['task_names']:
                            for acq_name in self.config["design_exp"]['acq_names']:
                                tag="task-" + task_name + "_acq-" + acq_name
                                os.makedirs(ID_first_level_dir + tag ,exist_ok=True)
        



    def run_first_level_glm(self, ID=None, i_fname=None,events_file=None,mask_file=None,task_name=None,run_name=None,contrasts = ["trial_RH-rest", "trial_RH", "rest"],smoothing_fwhm=1.5,verbose=True,redo=False):
        """
        Run first-level GLM for a specific subject and task.

        Parameters
        ----------
        ID : str
            Participant ID (e.g., "093")
        i_fname : str
            Filename of the input fMRI image (4D NIfTI file)
        events_file : str
            Filename of the events TSV file
        mask_file : str
            Filename of the mask NIfTI file where to restrict the analysis
        task_name : str
            Task name (e.g., "motor_acq-shimBase+3mm")
        contrasts : list of str, optional
            List of contrasts to compute (default is ["trial_RH-rest", "trial_RH", "rest"])
        smoothing_fwhm : float, optional
            Full-width at half-maximum for spatial smoothing (default is 1.5 mm)
        verbose : bool, optional
            Whether to print information during processing (default is True)
        redo : bool, optional
            Whether to redo the analysis even if results already exist (default is False)

        Returns
        -------
        None
        """
        # --- Input validation -------------------------------------------------------------
        if ID is None:
            raise ValueError("Please provide the participant ID (e.g., _.stc(ID='A001')).")
        if i_fname is None:
            raise ValueError("Please provide the filename of the input image.")
        if events_file is None:
            raise ValueError("Please provide the filename of the events TSV file.")
        if run_name is None or run_name=="":
            run_tag=""
        else:
            run_tag="_" + run_name
        # --- Define directories and load files -----------------------------------------------------------
        first_level_dir = self.first_level_dir.format(ID) + task_name + "/"
        os.makedirs(first_level_dir, exist_ok=True)

        df_events = pd.read_csv(events_file, sep="\t") # Load event file
        df_events=df_events#.iloc[1:-1] #remove the first raw
        df_events["trial_type"] = df_events["trial_type"].replace({"start": "rest"}) # start is equivalent to rest

        # Load json file
        json_file = os.path.join(self.raw_dir, f"sub-{ID}/func/sub-{ID}_{task_name}{run_tag}_bold.json")
        with open(json_file, "r") as f:
            json_data = json.load(f)
        tr = json_data.get("RepetitionTime")

        # Load fMRI image
        img = nib.load(i_fname)
        n_scans = img.shape[3]
        frame_times = np.arange(n_scans) * tr

        # --- Fit first-level model -----------------------------------------------------------
        design_mat_file = os.path.join(first_level_dir, f"sub-{ID}_{task_name}{run_tag}_design_matrix.png")
        if not os.path.exists(design_mat_file) or redo:
            model = FirstLevelModel(
                t_r=tr,
                noise_model="ar1",
                min_onset=0,
                standardize=False,
                hrf_model="spm + derivative + dispersion",
                drift_model=None,
                signal_scaling=0,
                high_pass=None,
                smoothing_fwhm=smoothing_fwhm,
                mask_img=mask_file
            )

            fmri_glm = model.fit(i_fname, events=df_events)

            # Plot design matrix 
            design_mat = fmri_glm.design_matrices_[0]
            
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
            plot_design_matrix(design_mat, ax=ax1)
            ax1.set_title(f"Design Matrix: sub-{ID}, {task_name}", fontsize=12)
            plt.savefig(design_mat_file)

        else:
            if verbose:
                print(f"First-level results already exist for sub-{ID} {task_name} {run_name}. Skipping computation.")
        
        # --- Compute contrasts and save -----------------------------------------------------------
        stat_maps=[]

        for i, contrast in enumerate(contrasts):
            if smoothing_fwhm is not None:
                tag="_s"
            else:
                tag=""
            stat_maps.append(os.path.join(first_level_dir, f"sub-{ID}_{task_name}{run_tag}_{contrast}{tag}.nii.gz"))
            
            if not os.path.exists(stat_maps[i]) or redo:
                results = fmri_glm.compute_contrast(contrast, output_type="stat")
                results.to_filename(stat_maps[i])
        
        return stat_maps
    
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

    def run_icc(self, IDs=None, i_fnames=None, o_dir=None, mask_file=None, threshold=0, fwhm=[1,1,1],redo=False):
        
        if IDs==None:
                raise ValueError('Please provide IDs labels (IDs=["sub-01","sub-02"])')
        if i_fnames==None:
                raise ValueError('Please provide filenames i_fnames=[["sub-01-run-01.nii.gz", "sub-01-run-02.nii.gz"],["sub-02-run-01.nii.gz", "sub-02-run-02.nii.gz"]]')
        
        if o_dir==None:
            o_dir=self.second_level_dir.format("icc_analysis")
        os.makedirs(o_dir,exist_ok=True)
        all_maps=[]

        o_fname=os.path.join(o_dir, 'group_voxelwise_ICC')
        if not os.path.exists(o_fname + '.nii.gz') or redo:
            for i, ID in enumerate(IDs):
                if len(i_fnames[i]) != 2:
                    raise ValueError("Need exactly 2 files per individual")

                # --- Load mask ---
                if mask_file:
                    mask_img = nib.load(mask_file)
                else:
                    mask_img = None

                run_data = []
                for f in i_fnames[i]:
                    img = nib.load(f)
                    data = img.get_fdata()

                    # --- resample mask to functional space ---
                    if mask_img:
                        mask_resampled = resample_to_img(mask_img, img, interpolation='nearest').get_fdata() > 0
                    else:
                        mask_resampled = data != 0  # fallback

                    # --- threshold ---
                    if threshold > 0:
                        mask_resampled &= data > threshold
                    
                    run_data.append(data[mask_resampled].ravel())
                if len(run_data) != 2:
                    raise ValueError(f"Expected 2 runs but got {len(run_data)}")
                run_data = np.stack(run_data, axis=1)
                all_maps.append(run_data)
            
            # --- Convert to array: subjects × runs × voxels ---
            all_maps_array = np.array([maps.T for maps in all_maps])  # subjects × runs × voxels

            n_subjects, n_runs, n_voxels = all_maps_array.shape
            icc_map = np.zeros(n_voxels)

            # --- Compute voxelwise ICC(3,1) ---
            for v in range(n_voxels):
                voxel_data = all_maps_array[:, :, v]  # subjects × runs
                df = pd.DataFrame({
                    'ID': np.repeat(np.arange(n_subjects), n_runs),
                    'run': np.tile(np.arange(n_runs), n_subjects),
                    'value': voxel_data.ravel()
                })
                icc_result = pg.intraclass_corr(data=df, targets='ID', raters='run', ratings='value')
                icc_map[v] = icc_result.loc[icc_result['Type'] == 'ICC3', 'ICC'].values[0]

            # --- Save as NIfTI ---
            icc_nii = np.zeros(mask_resampled.shape)
            icc_nii[mask_resampled] = icc_map
            icc_img = nib.Nifti1Image(icc_nii, affine=img.affine)
            nib.save(icc_img, o_fname + ".nii.gz")

            # apply smoothing for visual purpose
            if fwhm:
                icc_img_s=smooth_img(o_fname + ".nii.gz",fwhm=fwhm)
                icc_img_s.to_filename(o_fname + "_s.nii.gz")

        return o_fname + ".nii.gz",  o_fname + "_s.nii.gz"
    
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
        z_slice = 296#np.argmax(np.nanmax(crop_data, axis=(0, 1)))  # max over x and z, returns y index
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


        
    def run_second_level_glm(self,i_fnames=None,design_matrix=None,mask_fname=None,smoothing_fwhm=None,parametric=False,n_perm=10000,vox_thr=0.01,task_name=None,run_name=None,verbose=True,redo=False):
        '''
        Run second-level GLM for a specific task.
        # ongoing test nilearn: https://nilearn.github.io/stable/modules/generated/nilearn.glm.second_level.SecondLevelModel.html

        Parameters
        ----------
        i_fnames : list of str
            List of filenames of the input contrast images in the same space (e.g., ["sub-A001_task-motor_contrast-trial_RH-rest_inTemplate.nii.gz", "sub-A002_task-motor_contrast-trial_RH-rest_inTemplate.nii.gz"])
        design_matrix : pandas DataFrame, optional
            Design matrix for the second-level analysis (default is None, which will create a design matrix
            with an intercept only)
        mask_fname : str, optional
            Filename of the mask NIfTI file where to restrict the analysis (default is None)
        smoothing_fwhm : float, optional
            Full-width at half-maximum for spatial smoothing (default is None, which means no smoothing)
        parametric: bool
            Set True for parametric statistics or False for non-parametric
        n_perm: int
            Used for non-parametric testing, choose the number of permutation. 
        vox_thr:
            Cluster-forming threshold in p-scale: Uncorrected voxel threshold before cluster inerence (for non-parametric testing). 
        task_name : str
            Task name (e.g., "motor_acq-shimBase+3mm")
        run_name : str, optional
            Run name (e.g., "run-1") (default is None, which means no run name will be added to the output filename)
        verbose : bool, optional
            Whether to print information during processing (default is True)
        redo : bool, optional
            Whether to redo the analysis even if results already exist (default is False)
        Returns
        -------
        z_map_file : str
            Filename of the output t-map NIfTI file (e.g., "n20_motor_acq-shimBase+3mm_intercept_z_map.nii.gz")
        '''
        


        # --- Input validation -------------------------------------------------------------
        if i_fnames is None:
            raise ValueError("Please provide the list of filenames of the input contrast images.")
        
        # --- Define directories  -----------------------------------------------------------
        second_level_dir = self.second_level_dir.format(task_name) + "/"
        os.makedirs(second_level_dir, exist_ok=True)

        # Load design matrix file if provided, otherwise create a default design matrix with an intercept only
        if design_matrix is None:
            design_matrix = pd.DataFrame([1] * len(i_fnames),columns=["intercept"])

        if parametric ==True:
            stat_map_file = os.path.join(second_level_dir, f"n{len(i_fnames)}_{task_name}_intercept_z_map.nii.gz")
            if not os.path.exists(stat_map_file) or redo:
                print(f"Computing parametric second-level analysis for task {task_name}.")
                # --- Estimate and Fit second-level model -----------------------------------------------------------
                second_level_model = SecondLevelModel(mask_img=mask_fname,smoothing_fwhm=smoothing_fwhm, n_jobs=2, verbose=1) # define the model to the contrast images and the design matrix
                second_level_model.fit(i_fnames, design_matrix=design_matrix)  # fit the model to the contrast images and the design matrix
                
                # --- Compute contrasts and save -----------------------------------------------------------
                z_map = second_level_model.compute_contrast(second_level_contrast="intercept",output_type="z_score")
                z_map.to_filename(stat_map_file)
        
        else:
            stat_map_file = os.path.join(second_level_dir, f"n{len(i_fnames)}_{task_name}_")
            if not os.path.exists(stat_map_file + 'logp_max_t.nii.gz') or redo:
                print(f"Computing non-parametric second-level analysis for task {task_name} with {n_perm} permutations.")

                out_dict = non_parametric_inference(
                    i_fnames,
                    design_matrix=design_matrix,
                    mask=mask_fname,
                    model_intercept=True,
                    n_perm=n_perm, 
                    two_sided_test=False,
                    smoothing_fwhm=None,
                    n_jobs=2,
                    threshold=vox_thr, # voxel level threshold for cluster definition (uncorrected p-value)
                    #tfce=True,
                    verbose=1,
                    )
                
                out_dict["t"].to_filename(stat_map_file+ 't.nii.gz')
                out_dict["logp_max_t"].to_filename(stat_map_file+ 'logp_max_t.nii.gz')
                out_dict["logp_max_size"].to_filename(stat_map_file+ 'logp_max_size.nii.gz')
                out_dict["logp_max_mass"].to_filename(stat_map_file+ 'logp_max_mass.nii.gz')
                #out_dict["tfce"].to_filename(stat_map_file+ 'tfce.nii.gz')
                #out_dict["logp_max_tfce"].to_filename(stat_map_file+ 'logp_max_tfce.nii.gz')

                #mask the t-map with the significant cluster in the logp_max_size map
                logp_max_size_img = nib.load(stat_map_file+ 'logp_max_size.nii.gz')
                logp_max_size_data = logp_max_size_img.get_fdata()
                #threshold the logp_max_size map at p<0.05
                logp_max_size_data_thresholded = logp_max_size_data > -np.log10(0.05)
                #mask the t-map with the thresholded logp_max_size map
                t_img = nib.load(stat_map_file+ 't.nii.gz')
                t_data = t_img.get_fdata()
                t_data_masked = t_data * logp_max_size_data_thresholded
                t_masked_img = nib.Nifti1Image(t_data_masked, t_img.affine, t_img.header)
                t_masked_img.to_filename(stat_map_file+ 't_clustercorrected.nii.gz')

        
        return stat_map_file
    
    def plot_second_level_maps(self, i_fnames_pair=None, output_dir=None,stat_min=2.3, stat_max=5,background_fname=None,cmap="autumn",mask_fname=None, underlay_fname=None,task_name=None, verbose=True, redo=False):
        """
        Plot second-level statistical maps for two maps.


        To do: 
        - plot GM
        - add spinal levels in the coronal view 

        """
        if output_dir is None:
            raise ValueError("output_dir is empty")
        if i_fnames_pair is None or len(i_fnames_pair) == 0:
            raise ValueError("i_fnames_pair is empty")
        if background_fname is None :
            raise ValueError("Please provide PAM50 template filename")

        # --- Figure and gridspec ---
        fig = plt.figure(figsize=(3.5, 3.5))
        fig.subplots_adjust(left=0.01,right=0.99,top=0.95,bottom=0.01)
        
        height_ratios = [6.5, 3]  # coronal, axial
        
        gs = fig.add_gridspec(nrows=2, ncols=5, 
                              height_ratios=height_ratios,
                               width_ratios=[0.2,0.1,1,1,1.5], hspace=0.01, wspace=0.05)


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
            underlay_axi = underlay_data[x_min:x_max, y_min:y_max, z_slice].T
            ax_axi.imshow(template_axi, cmap="gray", origin="lower", aspect="auto")
            ax_axi.imshow(underlay_axi, cmap="gray", origin="lower", aspect="auto",alpha=0.1)
            print(template_axi.shape)
            im_axi = ax_axi.imshow(axi_slice, cmap=cmap, origin="lower", vmin=stat_min, vmax=stat_max, aspect="auto")
            ax_axi.axis("off")
            ax_axi.text(0.5, 0.01, f"z={z_slice}", color="white", fontsize=5,ha="center", va="bottom", transform=ax_axi.transAxes)
            ax_cor.axhline(y=z_slice - z_min, color='white', linestyle='--', linewidth=0.8, alpha=0.7)

            if i==0:
                ax_cor.set_title(f"baseShim", color="black", fontweight='bold', fontsize=7, fontname="Arial")
                ax_cor.text(0.05, 0.05, "L", transform=ax_cor.transAxes, color="white", fontsize=7, ha="left", va="bottom")
                ax_cor.text(0.95, 0.05, "R", transform=ax_cor.transAxes, color="white", fontsize=7, ha="right", va="bottom")
                ax_axi.text(0.02, 0.5, "L", transform=ax_axi.transAxes, color="white", fontsize=7, ha="left", va="center")
                ax_axi.text(0.98, 0.5, "R", transform=ax_axi.transAxes, color="white", fontsize=7, ha="right", va="center")
                ax_axi.text(0.5, 0.90, "A", transform=ax_axi.transAxes, color="white", fontsize=7, ha="center", va="top")
                ax_axi.text(0.5, 0.12, "P", transform=ax_axi.transAxes, color="white", fontsize=7, ha="center", va="bottom")

            else:
                ax_cor.set_title(f"sliceShim", color="black", fontweight='bold', fontsize=7, fontname="Arial")

        # -- Shared colorbar
        
        cbar_ax = fig.add_axes([0.03, 0.05, 0.02, 0.15])  # left, bottom, width, height
        norm = plt.Normalize(vmin=stat_min, vmax=stat_max)
        sm = plt.cm.ScalarMappable(cmap='autumn', norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('t-score', fontsize=6, labelpad=1.5,fontweight='bold',fontname="Arial")
        cbar.ax.set_yticks([])
        cbar.ax.text(1.35, 1.1, f"{stat_min:.1f}", fontsize=6, va='center', ha='right', color='black', transform=cbar.ax.transAxes)
        cbar.ax.text(1.35, -0.12, f"{stat_max:.1f}", fontsize=6, va='center', ha='right', color='black', transform=cbar.ax.transAxes)
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

        ax_levels_txt.text(-1.3, 0.9, "C5", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")
        ax_levels_txt.text(-1.3, 0.68, "C6", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")
        ax_levels_txt.text(-1.3, 0.49, "C7", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")
        ax_levels_txt.text(-1.3, 0.3, "C8", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")
        ax_levels_txt.text(-1.3, 0.1, "T1", transform=ax_cor.transAxes, color="black", fontsize=6, ha="center", va="center",fontweight='bold',fontname="Arial")

        # --- Add bar plot column for number of voxels ---
        colors=["#43BA8C","#F5AD27"]
        maps_name=["baseShim","SliceShim"]
        ax_bar_container = fig.add_subplot(gs[:, 4])
        ax_bar_container.axis("off")  # hide container axis
        ax_bar_top = inset_axes(
        ax_bar_container,
        width="40%",     # full width of column 3
        height="25%",     # 50% of its height
        loc="upper right",
        bbox_to_anchor=(-0.1, 0.01, 0.9, 0.9),
        bbox_transform=ax_bar_container.transAxes
        )

        ax_bar_top.bar(range(len(num_voxels_list)), num_voxels_list, color=colors, width=0.5, alpha=0.7)
        ax_bar_top.set_xticks(range(len(num_voxels_list)))
        ax_bar_top.set_xticklabels(
            [ maps_name[i] for i in range(len(num_voxels_list))],
            rotation=45,fontsize=6,fontweight='bold',fontname="Arial")
        ax_bar_top.set_ylabel("# voxels", fontsize=6,fontweight='bold',fontname="Arial")
        ax_bar_top.tick_params(axis='y', labelsize=6)
        ax_bar_top.yaxis.set_label_coords(-0.9, 0.5)
        ax_bar_top.tick_params(axis='y', which='both', pad=2)  # reduce padding if needed

        ax_bar_top.spines['left'].set_position(('outward', 10))  # 10 points outward
        ax_bar_top.spines['top'].set_visible(False)
        ax_bar_top.spines['right'].set_visible(False)

        # --- Add bar plot column for t-value distribution ---
        maps_name=["baseShim","SliceShim"]
        ax_hist = inset_axes(
        ax_bar_container,
        width="50%",     # full width of column 3
        height="25%",     # 50% of its height
        loc="center right",
        bbox_to_anchor=(-0.1, -0.1, 0.9, 0.9),
        bbox_transform=ax_bar_container.transAxes
        )

        for i, vals in enumerate(values_list):
            vals = np.asarray(vals)  # ensure it’s a NumPy array
            
        all_values = np.concatenate(values_list)
        bins = np.linspace(stat_min, np.nanmax(all_values), 30)
        
        for i, values in enumerate(values_list):
            ax_hist.hist(
                values,
                bins=bins,
                color=colors[i],
                density=False,      # normalize
                alpha=0.5,         # transparency
                label=maps_name[i]
            )

        ax_hist.set_xlabel("t-score", fontsize=6, fontweight='bold',fontname="Arial")
        ax_hist.set_ylabel("# voxels", fontsize=6, fontweight='bold',fontname="Arial")
        ax_hist.tick_params(axis='both', labelsize=6)

        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)

        ax_hist.legend(fontsize=6, frameon=False)
        ax_hist.legend(
            fontsize=5,
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(0.4, 1)   # x slightly outside axes
        )
        ax_hist.yaxis.set_label_coords(-0.5, 0.5)



        out_file=os.path.join(output_dir, f"second_level_maps.png")
        plt.savefig(out_file, dpi=300)
        plt.close(fig)



            