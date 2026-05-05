import os
import glob
import json
import numpy as np
import pandas as pd
import nibabel as nib
import pingouin as pg
import warnings

# nilearn
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import non_parametric_inference
from nilearn.image import resample_to_img
from nilearn.image import smooth_img

#stats
from scipy import stats
from scipy.ndimage import center_of_mass

#plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Arrow
from matplotlib.legend_handler import HandlerPatch
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from utils import compute_tsnr_map, compute_SNR, extract_mean_within_mask
#####################################################
class GLM_main:
    """
    The GLM_main class is used to setup the GLM path and execute the GLM steps.

    Attributes
    ----------
    config : dict
        Defining all the parameters of the analysis including the path to the raw data, the participants to analyze, the design of the experiment, and the preprocessing parameters
    IDs : list
        List of participant IDs to process (e.g., ['A001', 'A002'])
    verbose : bool
        Whether to print information during the each step (default: True)
    """

    def __init__(self, config, IDs=None,verbose=True):
        if IDs is None:
            raise ValueError("Please provide the participant ID (e.g., _.stc(ID='A001')).")
        
        # Class attributes -------------------------------------------------------------------------------------
        self.config = config # load config info
        self.participant_IDs= IDs # list of the participants to analyze
        self.raw_dir = self.config["raw_dir"]  # directory of the raw data
        self.derivatives_dir = os.path.join(self.config["raw_dir"], self.config["derivatives_dir"])  # directory of the derivatives data
        self.first_level_dir = os.path.join(self.config["raw_dir"], self.config["first_level"]["dir"])  # directory of the derivatives data
        self.second_level_dir = os.path.join(self.config["raw_dir"], self.config["second_level"]["dir"])  # directory of the second-level analysis data
        self.manual_dir = os.path.join(self.config["raw_dir"], self.config["manual_dir"])  # directory of the manual corrections

        # Create directories -------------------------------------------------------------------------------------
        for ID in self.participant_IDs:
            ID_glm_dir=self.first_level_dir.format('glm',ID)
            os.makedirs(ID_glm_dir, exist_ok=True)

            # Create a folder for each task in participant folder
            if "design_exp" in self.config.keys():
                for ses_name in self.config["design_exp"]['ses_names']:
                    ses_dir=ses_name if int(self.config["design_exp"]["ses_nb"])>1 else ""
                    if "acq_names" in self.config["design_exp"].keys():
                        for task_name in self.config["design_exp"]['task_names']:
                            for acq_name in self.config["design_exp"]['acq_names']:
                                tag="task-" + task_name + "_acq-" + acq_name
                                os.makedirs(ID_glm_dir + tag ,exist_ok=True)
        



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
        first_level_dir = os.path.join(self.first_level_dir.format("glm",ID), task_name)
        os.makedirs(first_level_dir, exist_ok=True)

        df_events = pd.read_csv(events_file, sep="\t") # Load event file
        df_events=df_events#.iloc[1:-1] #remove the first raw
        df_events["trial_type"] = df_events["trial_type"].replace({"start": "rest"}) # start is equivalent to rest

        # Load json file
        json_file = os.path.join(self.raw_dir, f"sub-{ID}", "func", f"sub-{ID}_{task_name}{run_tag}_bold.json")
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
            plot_design_matrix(design_mat, axes=ax1)
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
    

        files = glob.glob(os.path.join(
            self.raw_dir,
            self.config["preprocess_dir"]["main_dir"].format(ID),
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

    def run_icc(self, IDs=None, i_fnames=None, o_dir=None, mask_file=None, threshold=0, fwhm=[1,1,1],redo=False):
        
        if IDs is None:
                raise ValueError('Please provide IDs labels (IDs=["sub-01","sub-02"])')
        if i_fnames is None:
                raise ValueError('Please provide filenames i_fnames=[["sub-01-run-01.nii.gz", "sub-01-run-02.nii.gz"],["sub-02-run-01.nii.gz", "sub-02-run-02.nii.gz"]]')
        
        if o_dir is None:
            o_dir = self.second_level_dir.format("icc_analysis")
        os.makedirs(o_dir, exist_ok=True)
        all_maps = []

        o_fname = os.path.join(o_dir, 'group_voxelwise_ICC')
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

            # --- Compute voxelwise ICC(C,1) ---
            # Pingouin updated  the intraclass_corr function, the output df Type changed from "ICC3" to "ICC(C,1)". See https://github.com/raphaelvallat/pingouin/pull/501
            for v in range(n_voxels):
                voxel_data = all_maps_array[:, :, v]  # subjects × runs
                df = pd.DataFrame({
                    'ID': np.repeat(np.arange(n_subjects), n_runs),
                    'run': np.tile(np.arange(n_runs), n_subjects),
                    'value': voxel_data.ravel()
                })
                icc_result = pg.intraclass_corr(data=df, targets='ID', raters='run', ratings='value')
                icc_map[v] = icc_result.loc[icc_result['Type'] == 'ICC(C,1)', 'ICC'].values[0]

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
    
    def run_second_level_glm(self,i_fnames=None,design_matrix=None,mask_fname=None,smoothing_fwhm=None,parametric=False,n_perm=10000,vox_thr=0.01,cluster_corr=0.01,task_name=None,n_jobs=2,run_name=None,verbose=True,redo=False):

        """
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
        """
        


        # --- Input validation -------------------------------------------------------------
        if i_fnames is None:
            raise ValueError("Please provide the list of filenames of the input contrast images.")
        
        # --- Define directories  -----------------------------------------------------------
        second_level_dir = os.path.join(self.second_level_dir.format("glm"), f"cluster_p{cluster_corr}_vox{vox_thr}_perm{n_perm}", task_name)
        os.makedirs(second_level_dir, exist_ok=True)

        # Load design matrix file if provided, otherwise create a default design matrix with an intercept only
        if design_matrix is None:
            design_matrix = pd.DataFrame([1] * len(i_fnames),columns=["intercept"])

        if parametric:
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
                    smoothing_fwhm=smoothing_fwhm,
                    n_jobs=n_jobs,
                    threshold=vox_thr, # voxel level threshold for cluster definition (uncorrected p-value)
                    tfce=False,
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
                #threshold the logp_max_size map at cluster_corr
                logp_max_size_data_thresholded = logp_max_size_data > -np.log10(cluster_corr)
                #mask the t-map with the thresholded logp_max_size map
                t_img = nib.load(stat_map_file+ 't.nii.gz')
                t_data = t_img.get_fdata()
                t_data_masked = t_data * logp_max_size_data_thresholded
                t_masked_img = nib.Nifti1Image(t_data_masked, t_img.affine, t_img.header)
                t_masked_img.to_filename(stat_map_file+ 't_clustercorrected.nii.gz')

        return stat_map_file+ 't_clustercorrected.nii.gz'

    def extract_metrics(self,i_fname=None,threshold=0,o_fname=None,redo=False):
        
        if i_fname is None:
            raise ValueError("Please provide the filename of the input image.")
        
        if o_fname==None:
            o_fname=i_fname.split('.nii.gz')[0]
        
        fname_metrics = o_fname + "_metrics.csv"
        fname_values  = o_fname + "_values.csv"

        if not os.path.exists(fname_metrics) or not os.path.exists(fname_values) or redo:

            num_voxels_list=[];values_list=[]

            # --- Load ---
            img = nib.as_closest_canonical(nib.load(i_fname))
            data = img.get_fdata()

            # --- Extract metrics ---
            all_values=data.flatten()
            threshold_values=all_values[all_values > threshold]
            if len(threshold_values)>0:
                df_metrics = pd.DataFrame([{
                    "total_voxels": len(threshold_values),
                    "nonzero_voxels": len(threshold_values),
                    "mean": np.mean(threshold_values),
                    "std": np.std(threshold_values),
                    "min": np.min(threshold_values),
                    "max": np.max(threshold_values),
                }])
            else:
                df_metrics = pd.DataFrame([{
                    "total_voxels": 0,
                    "nonzero_voxels": 0,
                    "mean": 0,
                    "std": 0,
                    "min": 0,
                    "max": 0,
                }])
                

            df_values = pd.DataFrame({"voxels_values": threshold_values})
    
            df_metrics.to_csv(o_fname + "_metrics.csv", index=False)
            df_values.to_csv(o_fname + "_values.csv", index=False)

        return fname_metrics, fname_values

    def extract_FD(self,IDs=None,task_name=None,run_name=None,output_file=None,redo=False):
        """Extract mean framewise displacement (FD) for each participant and save to a CSV file.
        Parameters
        ----------
        IDs : list of str
            List of participant IDs (e.g., ["A001", "A002"])
        task_name : str
            Task name (e.g., "motor_acq-shimBase+3mm")
        run_name : str, optional
            Run name (e.g., "run-1") (default is None, which means no run name will be added to the output filename)
        output : str, optional
            Output filename (default is None, which means the output filename will be automatically generated)
        redo : bool, optional
            If True, recompute the FD even if the output file already exists (default is False)
        Returns
        -------
        str
            Filename of the output CSV file containing mean FD for each participant
        """

        if IDs is None:
            raise ValueError("Please provide the list of participant IDs (e.g., ['A001', 'A002']).")

        FD_values = {"IDs": [], "mean_FD": [],"acq":[]}
        for i, ID in enumerate(IDs):
            preprocess_dir = os.path.join(self.config["raw_dir"], self.config["preprocess_dir"]["main_dir"].format(ID), "func")
            moco_param_file = glob.glob(os.path.join(preprocess_dir, f"{task_name}","sct_fmri_moco", f"moco_params*{run_name[i]}.txt"))[0]
            params_data=pd.read_csv(moco_param_file, delimiter=',', header=None)

            # calulate the mean FD from the moco_params file, which contains the FD for each volume
            diff_XY = np.abs(np.diff(params_data[0])) # Calculate Framewise displacement (abs difference of displacement between each volumes)
            mean_FD=np.mean(diff_XY)

            FD_values["IDs"].append(ID)
            FD_values["mean_FD"].append(mean_FD)
            FD_values["acq"].append(task_name.split("acq-")[1].split("+")[0])
        df_FD = pd.DataFrame(FD_values)

        if output_file is None:
            output_file = os.path.join(self.second_level_dir.format("FD"), f"n{len(IDs)}_mean_FD_{task_name}.csv")

        if not os.path.exists(output_file) or redo:
            df_FD.to_csv(output_file, index=False)

        mean_FD = df_FD["mean_FD"].mean()
        std_FD = df_FD["mean_FD"].std()

        print(f"Mean FD {np.round(mean_FD, 2)} ± {np.round(std_FD, 2)} for task {task_name}")
        return output_file


class TSNR_main:
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
        self.first_level_dir = os.path.join(self.config["raw_dir"], self.config["first_level"]["dir"])  # directory of the derivatives data
        self.second_level_dir= os.path.join(self.config["raw_dir"], self.config["second_level"]["dir"])
        self.path_tsnr = os.path.join(self.first_level_dir.format("snr","").split("sub")[0])
        self.path_tsnr_inTemplate = os.path.join(self.second_level_dir.format("snr"))
        self.fname_metrics = {
            "ssnr": os.path.join(self.path_tsnr, "ssnr_metrics.csv"),
            "tsnr": os.path.join(self.path_tsnr, "tsnr_metrics.csv")
        }

    def generate_tsnr_maps_and_csv(self, space="native", native_gm_mask=None):
        """
        tSNR extraction can be either be done in native or PAM50 space:
        space: str
            choose the option "native" or "PAM50"
        """
        dfs= {
            "tsnr": pd.DataFrame(columns=["IDs", "task", "acq", "tsnr"]),
            "ssnr": pd.DataFrame(columns=["IDs", "task", "acq", "ssnr"])
        }

        print("=== Compute tSNR map on longest moco neighbour run ===", flush=True)
        # Find the minimum number of volumes across all runs to standardize tSNR calculation
        min_vols_for_tsnr = 1000
        for ID in self.IDs:
            for task in self.config["design_exp"]["task_names"]:
                for acq_name in self.config["design_exp"]["acq_names"]:
                    selected_file = self.find_moco_for_tsnr_calculation(ID, task, acq_name)
                    if selected_file is None:
                        continue
                    n_vols = nib.load(selected_file).shape[3]
                    if n_vols < min_vols_for_tsnr:
                        min_vols_for_tsnr = n_vols

        print(f"Minimum number of volumes across all runs: {min_vols_for_tsnr}", flush=True)
        # Minimum number of volumes across all runs: 60 (2026-04-07)

        # Compute_tsnr
        print(f"Compute tSNR for each participant", flush=True)
        for ID in self.IDs:
            print(f"Participant: {ID}")
            for task in self.config["design_exp"]["task_names"]:
                for acq_name in self.config["design_exp"]["acq_names"]:
                    tag = "task-" + task + "_acq-" + acq_name

                    selected_file = self.find_moco_for_tsnr_calculation(ID, task, acq_name)
                    if selected_file is None:
                        continue

                    selected_mean_file = selected_file.split('.')[0] + "_mean.nii.gz"

                    # Compute tSNR map in native space
                    path_tsnr_sub_folder = os.path.join(self.path_tsnr, f"sub-{ID}", tag)
                    fname_tsnr = compute_tsnr_map(selected_file, path_tsnr_sub_folder, self.redo, min_vols_for_tsnr)

                    # Segmentation file in native space
                    fname_mask = os.path.join(self.config["raw_dir"],self.config["preprocess_dir"]["main_dir"].format(ID),"func",tag,f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz")

                    # Warp tSNR in PAM50 space
                    fname_tsnr_in_template = fname_tsnr.replace("tsnr.nii.gz",
                                                                "tsnr_in_PAM50.nii.gz")
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

                    if native_gm_mask:
                        mask_PAM50=os.path.join(self.config["code_dir"], "template",self.config["PAM50_gm"])
                        fname_gm_mask = fname_tsnr.split("tsnr")[0] + "gm.nii.gz"
                        fname_wm_mask = fname_tsnr.split("tsnr")[0] + "wm.nii.gz"
                        fname_warp_from_template_to_func = os.path.join(
                            self.config["raw_dir"],
                            self.config["preprocess_dir"]["main_dir"].format(ID),
                            "func",
                            tag,
                            f"sub-{ID}_{tag}_from-PAM50_to_func_mode-image_xfm.nii.gz")

                        if not os.path.exists(fname_gm_mask) or self.redo:
                            cmd_coreg = f"sct_apply_transfo -i {mask_PAM50} -d {fname_tsnr} -w {fname_warp_from_template_to_func} -o {fname_gm_mask} -x nn"
                            cmd_bin=f"fslmaths {fname_gm_mask} -thr 0.1 -bin {fname_gm_mask}" # binarize the mask
                            os.system(cmd_coreg)
                            os.system(cmd_bin)

                        if not os.path.exists(fname_wm_mask) or self.redo:
                            cmd_bin=f"fslmaths {fname_mask} -sub {fname_gm_mask} -thr 0.1 -bin {fname_wm_mask}" # binarize the mask
                            os.system(cmd_bin)

                    # Extract metrics from native space
                    if space=="native" and fname_tsnr is not None:
                        if native_gm_mask:
                            self.fname_metrics["tsnr"] = os.path.join(self.path_tsnr, "tsnr_ratio_metrics.csv")
                            fname_mask = fname_gm_mask
                            tsnr_mean_gm = extract_mean_within_mask(fname_tsnr, fname_gm_mask)
                            tsnr_mean_wm = extract_mean_within_mask(fname_tsnr, fname_wm_mask)
                            tsnr_mean = tsnr_mean_gm / tsnr_mean_wm
                        else:
                            tsnr_mean = extract_mean_within_mask(fname_tsnr, fname_mask)

                    # Extract metrics from PAM50 space
                    elif space == 'PAM50' and fname_tsnr_in_template is not None:
                        self.fname_metrics["tsnr"] = os.path.join(self.path_tsnr, "tsnr_metrics_PAM50.csv")
                        if fname_mask is None:
                            fname_mask = os.path.join(
                                self.config["code_dir"],
                                "template",
                                "PAM50_cord.nii.gz")

                        if not os.path.exists(fname_mask):
                            raise RuntimeError(f"Mask file not found: {fname_mask}")

                        tsnr_mean = extract_mean_within_mask(fname_tsnr_in_template, fname_mask)

                    # Extract sSNR
                    ssnr = compute_SNR(selected_mean_file, fname_mask)

                    for metric in ["tsnr", "ssnr"]:
                        values = tsnr_mean if metric=="tsnr" else ssnr
                        if len(dfs[metric]) == 0:
                            dfs[metric] = pd.DataFrame([[ID, task, acq_name.split("+")[0], values]], columns=dfs[metric].columns)
                        else:
                            dfs[metric] = pd.concat(
                                [pd.DataFrame([[ID, task, acq_name.split("+")[0], values]], columns=dfs[metric].columns), dfs[metric]], ignore_index=True)

        # Keep only 'rest' rows for IDs that have both 'motor' and 'rest'
        for metric in ["tsnr","ssnr"]:

            if not os.path.exists(self.fname_metrics[metric].split(".csv")[0]+"_reduced.csv") or self.redo:
                ids_with_both = dfs[metric].groupby('IDs')['task'].apply(
                    lambda x: set(['motor', 'rest']).issubset(set(x))
                )
                ids_with_both = ids_with_both[ids_with_both].index
                df_reduced = dfs[metric][~((dfs[metric]['IDs'].isin(ids_with_both)) & (dfs[metric]['task'] == 'motor'))]
                df_reduced.to_csv(self.fname_metrics[metric].split(".csv")[0]+"_reduced.csv", index=False)
            pair_ttest(csv_files=[self.fname_metrics[metric].split(".csv")[0]+"_reduced.csv"], value_col=metric, redo=self.redo)

            if not os.path.exists(self.fname_metrics[metric]) or self.redo:
                dfs[metric].to_csv(self.fname_metrics[metric], index=False)
                pair_ttest(csv_files=[self.fname_metrics[metric]], value_col=metric, redo=self.redo)

    def generate_average_tsnr_in_pam50(self, IDs, acq_name=None,task_name=None,tsnr_fnames=None,seg_fnames=None, warp_fnames=None,fname_mask=None, redo=False):
        
        if IDs is None:
            raise ValueError("Please provide a list of participant IDs (e.g., _.stc(IDs=['A001','A002'])).")
        if tsnr_fnames is None:
            raise ValueError("Please provide a list of the input tSNR filenames.")
        if seg_fnames is None:
            raise ValueError("Please provide a list of the input segmentation filenames.")
        if warp_fnames is None:
            raise ValueError("Please provide a list of the input warping field filenames.")
        
        print("=== Generate average tSNR maps in PAM50  ===", flush=True)
        fname_template = os.path.join(self.config["code_dir"], "template", self.config["PAM50_t2"])
        nii_template = nib.load(fname_template)
        data_tsnr = np.zeros_like(nii_template.get_fdata(), dtype=float)
        data_count_id = None

        fname_tsnr_avg = os.path.join(self.path_tsnr_inTemplate, f"tsnr_n{str(len(tsnr_fnames))}_{acq_name}_avg_in_PAM50.nii.gz")
        os.makedirs(os.path.dirname(fname_tsnr_avg), exist_ok=True)

        if not os.path.exists(fname_tsnr_avg) or redo:
            for i,ID in enumerate(IDs):
                tsnr_basename = os.path.join(os.path.dirname(tsnr_fnames[i]), os.path.basename(tsnr_fnames[i]).split("moco")[0])
                fname_tsnr_in_template = glob.glob(tsnr_basename + "moco_tsnr_in_PAM50.nii.gz")[0]

                nii_roi = count_roi_in_template(os.path.dirname(tsnr_fnames[i]),
                                                     ID, 
                                                     task_name,
                                                     acq_name,
                                                     tsnr_fnames[i],
                                                     warp_fnames[i],
                                                     os.path.join(fname_template),
                                                     redo)

                nii_tsnr = nib.load(fname_tsnr_in_template)
                data_tsnr += nii_tsnr.get_fdata()
            
                if data_count_id is None:
                    data_count_id = nii_roi.get_fdata()
                else:
                    data_count_id += nii_roi.get_fdata()

            # Average
            data_tsnr_avg = np.divide(data_tsnr, data_count_id, out=np.zeros_like(data_tsnr), where=data_count_id != 0)

            # --- Apply mask if provided ---
            if fname_mask is not None:
                nii_mask = nib.load(fname_mask)
                mask_data = nib.as_closest_canonical(nii_mask).get_fdata().astype(bool)
                if mask_data.shape != data_tsnr_avg.shape:
                    raise ValueError(f"Mask shape {mask_data.shape} does not match data shape {data_tsnr_avg.shape}")
                data_tsnr_avg[~mask_data] = 0
            
            nii_tsnr_avg = nib.Nifti1Image(data_tsnr_avg, affine=nii_tsnr.affine, header=nii_tsnr.header)
            nib.save(nii_tsnr_avg, fname_tsnr_avg)

        return fname_tsnr_avg

    def find_moco_for_tsnr_calculation(self, ID, task, acq_name):
        files = sorted(glob.glob(os.path.join(
            self.config["raw_dir"],
            self.config["preprocess_dir"]["main_dir"].format(ID),
            "func",
            f"task-{task}_acq-{acq_name}",
            "sct_fmri_moco",
            f"sub-{ID}_task-{task}_acq-{acq_name}*_bold_moco.nii.gz"
        )))
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

    
class EpiComparison:
    def __init__(self, config, IDs, redo):
        self.IDs = IDs
        self.config = config
        self.redo = redo

        self.path_main_fig = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"])
        self.path_fig_epi_comparison = os.path.join(self.path_main_fig, "epi_comparison")
        os.makedirs(self.path_fig_epi_comparison, exist_ok=True)
        self.path_fig_data = os.path.join(self.path_fig_epi_comparison, "data")
        os.makedirs(self.path_fig_data, exist_ok=True)

    def create_figure(self, show_avg=False):
        print("=== Create EPI comparison figure ===", flush=True)

        for ID in self.IDs:
            # Figure looks better if we use more volumes (task-motor has more volumes than rest)
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
            self._create_gif_comparison(ID, redo=self.redo)

    def _create_gif_comparison(self, ID, redo):
        show_slice_factor = 2
        fname_gif = os.path.join(self.path_fig_epi_comparison, f"sub-{ID}_epi_comparison.gif")
        if not os.path.exists(fname_gif) or redo:
            # Create figure that shows moco mean in native space between baseline and slicewise shim
            name_baseline = [a for a in self.config["design_exp"]["acq_names"] if "Base" in a][0]
            name_slicewise = [a for a in self.config["design_exp"]["acq_names"] if "Slice" in a][0]

            task = self._get_task_of_moco_mean_same_vols(ID, name_baseline)
            # Paths for baseline and slicewise shim images
            fname_baseline = self._select_moco_mean_same_vols(ID, name_baseline)
            fname_seg_baseline, _, fname_warp_from_pam50_to_func_baseline = get_fname_seg_and_warps(ID, task,
                                                                                                    name_baseline,
                                                                                                    self.config)
            fname_slicewise = self._select_moco_mean_same_vols(ID, name_slicewise)
            fname_seg_slicewise, _, fname_warp_from_pam50_to_func_slicewise = get_fname_seg_and_warps(ID, task,
                                                                                                      name_slicewise,
                                                                                                      self.config)

            # Load images and masks
            img_baseline = nib.load(fname_baseline).get_fdata()
            img_slicewise = nib.load(fname_slicewise).get_fdata()
            mask_baseline = nib.load(fname_seg_baseline).get_fdata()
            mask_slicewise = nib.load(fname_seg_slicewise).get_fdata()

            n_slices = img_baseline.shape[2]

            fig = plt.figure(figsize=(round(n_slices / show_slice_factor + 0.1 / 2) / 2, 0.5))
            gs_main = gridspec.GridSpec(1, 1, figure=fig, hspace=0, wspace=0)

            title_fontsize = 5
            # Python rounds 12.5 to 12 instead of 13, so we add 0.1 to make sure we have enough space for all slices
            gs = gs_main[0].subgridspec(1, round(n_slices / show_slice_factor + 0.1), hspace=0, wspace=0)
            # gs_slicewise = gs_main[1].subgridspec(round(n_slices / show_slice_factor + 0.1), 1, hspace=0, wspace=0)
            axs = gs.subplots()
            # axs_slicewise = gs_slicewise.subplots()
            # axs_slicewise[0].set_title(f"ID {ID} slice-wise\nf0xyz shim", fontsize=title_fontsize)

            bound_lr = 16  # left-right bound
            bound_ud = 16  # up-down bound
            vmin, vmax = calc_vmin_vmax(mask_baseline, mask_slicewise, img_baseline, img_slicewise, False,
                                        avg_baseline=None, avg_slicewise=None, show_slice_factor=2,
                                        bound_lr=bound_lr, bound_ud=bound_ud)
            delta = vmax - vmin
            # vmin = vmin + 0.1 * delta
            vmax = vmax - 0.2 * delta

            ims = []
            for idx, slice_idx in enumerate(range(n_slices - 1, -1, -show_slice_factor)):
                com_baseline = center_of_mass(mask_baseline[:, :, slice_idx])

                # Define cropping bounds
                crop_x_baseline = slice(max(0, int(com_baseline[0] - bound_lr)),
                                        min(mask_baseline.shape[0], int(com_baseline[0] + bound_lr)))
                crop_y_baseline = slice(max(0, int(com_baseline[1] - bound_ud)),
                                        min(mask_baseline.shape[1], int(com_baseline[1] + bound_ud)))

                # Crop the images
                cropped_baseline = img_baseline[crop_x_baseline, crop_y_baseline, slice_idx]

                ims.append(axs[idx].imshow(cropped_baseline.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, animated=True))
                axs[idx].axis('off')
                axs[idx].set_aspect('equal', adjustable='box')

                template_slice_idx = self.func_slice_to_template_slice(slice_idx, com_baseline[0], com_baseline[1],
                                                                       fname_warp_from_pam50_to_func_baseline,
                                                                       fname_baseline, ID, task, name_baseline)
                spinal_level = template_slice_to_spinal_level(template_slice_idx)[1]
                axs[idx].text(0.15, 0.85, spinal_level, color='white', fontsize=4, fontweight='bold',
                                       ha='center', va='center', transform=axs[idx].transAxes)

            def update(i):
                for idx, slice_idx in enumerate(range(n_slices - 1, -1, -show_slice_factor)):
                    if i % 2 == 1:
                        com_slicewise = center_of_mass(mask_slicewise[:, :, slice_idx])
                        crop_x_slicewise = slice(max(0, int(com_slicewise[0] - bound_lr)),
                                                 min(mask_slicewise.shape[0], int(com_slicewise[0] + bound_lr)))
                        crop_y_slicewise = slice(max(0, int(com_slicewise[1] - bound_ud)),
                                                 min(mask_slicewise.shape[1], int(com_slicewise[1] + bound_ud)))
                        cropped_slicewise = img_slicewise[crop_x_slicewise, crop_y_slicewise, slice_idx]
                        ims[idx].set_data(cropped_slicewise.T)
                    else:
                        com_baseline = center_of_mass(mask_baseline[:, :, slice_idx])
                        crop_x_baseline = slice(max(0, int(com_baseline[0] - bound_lr)),
                                                min(mask_baseline.shape[0], int(com_baseline[0] + bound_lr)))
                        crop_y_baseline = slice(max(0, int(com_baseline[1] - bound_ud)),
                                                min(mask_baseline.shape[1], int(com_baseline[1] + bound_ud)))
                        cropped_baseline = img_baseline[crop_x_baseline, crop_y_baseline, slice_idx]
                        ims[idx].set_data(cropped_baseline.T)

                return ims
            ani = animation.FuncAnimation(fig, update, frames=2, interval=2000, blit=False, repeat=True)
            ani.save(fname_gif, dpi=2000, writer="ffmpeg")

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

        self.fname_fig_epi_comparison = os.path.join(self.path_fig_epi_comparison, "epi_comparison.png")
        
        if not os.path.exists(self.fname_fig_epi_comparison) or self.redo:
            highlight = {
                # '090': {1: 'sigtot', 3: 'sigtot', 7: 'sigtot', 9: 'geo'},
                '094': {1: 'sigtot', 5: 'geo', 27: 'sigtot'},
                '095': {1: 'sigvert', 3: 'sigvert', 21: 'sigtot'},
                '100': {1: 'sigtot', 5: 'sigtot', 7: 'sigtot', 9: 'sigtot', 19: 'sigvert'},
                '101': {1: 'geo', 5: 'sigtot', 27: 'sigvert'},
                '106': {1: 'sigtot', 11: 'geo', 15: 'geo', 19: 'geo'}}

            color = {'sigtot': '#2ca02c', 'sigvert': '#26ede3', 'geo': '#b996d9'}

            name_baseline = [a for a in self.config["design_exp"]["acq_names"] if "Base" in a][0]
            name_slicewise = [a for a in self.config["design_exp"]["acq_names"] if "Slice" in a][0]

            n_part = len(self.IDs)
            if n_part > 5:
                n_part = 5

            ids_to_show = []
            # 094, 096, '100'
            chose_if_available = ('094', '095', '100', '101', '106')

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
                fig = plt.figure(figsize=(2.15 * (n_part + 1), n_max_slices // show_slice_factor))
                gs_main = gridspec.GridSpec(2, ((n_part + 1) * 2) + n_part, figure=fig, hspace=0, wspace=0, width_ratios=width_ratios, height_ratios=[0.001,1])
            else:
                width_ratios = width_ratios[:-1]  # remove the last 0.1
                fig = plt.figure(figsize=(2.15 * n_part, n_max_slices // show_slice_factor))
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
                vmax = vmax - 0.2 * delta

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

                    color_baseline = '#ADA8A8'
                    color_slicewise = '#ED263F'
                    if idx == len(range_slices) - 1 and i_id == 0:
                        legend_fontsize = 10
                        legend_elements = [Patch(facecolor='white', edgecolor=color_baseline, label='shimBase',
                                                linewidth=1.5),
                                        Patch(facecolor='white', edgecolor=color_slicewise, label='shimSlice',
                                                linewidth=1.5)]
                        axs_baseline[idx].legend(handles=legend_elements, loc=(0, -0.75), fontsize=legend_fontsize)

                        legend_elements = [Arrow(0, 0, 0.2, 0, color=color['sigtot'], label='Improvement in signal loss from overall inhomogeneity', width=0.1, lw=2),
                                        Arrow(0, 0, 0.2, 0, color=color['sigvert'], label='Improvement in signal loss from inter-vertebral discs', width=0.1, lw=2),
                                        Arrow(0, 0, 0.2, 0, color=color['geo'], label='Improvement in geometric distortions', width=0.1, lw=2)]
                        axs_slicewise[idx].legend(handles=legend_elements, loc=(1.2, -1), fontsize=legend_fontsize, handler_map={mpatches.Arrow : HandlerPatch(patch_func=make_legend_arrow)})

                    # Color the borders of the baseline images in yellow to differentiate them from the slicewise images
                    spine_thickness = 2.5
                    axs_baseline[idx].tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False, bottom=False, left=False)
                    # Example: Change spine border thickness
                    for spine in axs_baseline[idx].spines.values():
                        spine.set_linewidth(spine_thickness)  # Set the border thickness
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

        if not os.path.exists(os.path.join(self.path_fig_epi_comparison, f"sub-{ID}_epi_comparison.png")) or self.redo:
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

        os.makedirs(path_sub, exist_ok=True)

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

def pair_ttest(df=None, csv_files=None, output_fname=None,index='IDs', value_col='tSNR', acq_col='acq', cond1='shimSlice', cond2='shimBase',task_filter=None, task_col='task', redo=False):

        if output_fname==None and csv_files:
            output_fname=csv_files[0].split('.csv')[0] + "_stats.csv"

        if not os.path.exists(output_fname) or redo:
            if csv_files:
                if len(csv_files) > 1:
                    df_list = [pd.read_csv(f) for f in csv_files]
                    df = pd.concat(df_list, ignore_index=True) # contactenate all dataframes into one
                else:
                    df = pd.read_csv(csv_files[0])

            # Filter by task if requested
            if task_filter:
                df = df[df[task_col] == task_filter]

            # Pivot to get one column per condition
            df_pivot = df.pivot_table(index=index, columns=acq_col, values=value_col)
            df_pivot = df_pivot.dropna(subset=[cond1, cond2])

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(df_pivot[cond1], df_pivot[cond2])
            degrees_of_freedom = len(df_pivot) - 1

            # Significance stars
            if p_value < 0.001:
                stars = '***'
            elif p_value < 0.01:
                stars = '**'
            elif p_value < 0.05:
                stars = '*'
            else:
                stars = 'ns'

            # Build results dataframe
            results = pd.DataFrame([{
                'cond1'         : cond1,
                'cond2'         : cond2,
                'N_pairs'       : len(df_pivot),
                f'mean_{cond1}'    : df_pivot[cond1].mean(),
                f'std_{cond1}'     : df_pivot[cond1].std(),
                f'mean_{cond2}'    : df_pivot[cond2].mean(),
                f'std_{cond2}'     : df_pivot[cond2].std(),
                't_stat'        : t_stat,
                'df'            : degrees_of_freedom,
                'p_value'       : p_value,
                'significance'  : stars,
            }])
            if task_filter:
                results['task_filter'] = task_filter if task_filter else 'all'

            results.to_csv(output_fname, index=False)
            print(results.to_string(index=False))

        return pd.read_csv(output_fname)