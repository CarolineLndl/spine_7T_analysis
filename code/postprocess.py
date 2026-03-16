import os
import glob
import json
import numpy as np
import pandas as pd
import nibabel as nib
import pingouin as pg

# nilearn
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import non_parametric_inference
from nilearn.image import resample_to_img
from nilearn.image import smooth_img

from scipy import stats

from utils import compute_tsnr_map, extract_mean_within_mask
#####################################################
class GLM_main:
    '''
    The GLM_main class is used to setup the GLM path and execute the GLM steps.

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
        first_level_dir = self.first_level_dir.format("glm",ID) + task_name + "/"
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
    
    def run_second_level_glm(self,i_fnames=None,design_matrix=None,mask_fname=None,smoothing_fwhm=None,parametric=False,n_perm=10000,vox_thr=0.01,task_name=None,n_jobs=2,run_name=None,verbose=True,redo=False):

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
                    smoothing_fwhm=smoothing_fwhm,
                    n_jobs=n_jobs,
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

        
        return stat_map_file+ 't_clustercorrected.nii.gz'

    def extract_metrics(self,i_fname=None,threshold=0,o_fname=None,redo=False):
        
        if i_fname is None:
            raise ValueError("Please provide the filename of the input image.")
        
        if o_fname==None:
            o_fname=i_fname.split('.')[0]
        
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
            

            df_metrics = pd.DataFrame([{
                "total_voxels": len(threshold_values),
                "nonzero_voxels": len(threshold_values),
                "mean": np.mean(threshold_values),
                "std": np.std(threshold_values),
                "min": np.min(threshold_values),
                "max": np.max(threshold_values),
            }])

            df_values = pd.DataFrame({"voxels_values": threshold_values})
    
            df_metrics.to_csv(o_fname + "_metrics.csv", index=False)
            df_values.to_csv(o_fname + "_values.csv", index=False)

        return fname_metrics, fname_values

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
        self.path_tsnr = os.path.join(self.first_level_dir.format("tsnr","").split("sub")[0])
        self.path_tsnr_inTemplate = os.path.join(self.second_level_dir.format("tsnr"))
        #self.fname_tsnr_baseline_avg = os.path.join(self.path_fig_tsnr, "data", "tsnr_baseline_avg_in_PAM50.nii.gz")
        #self.fname_tsnr_slicewise_avg = os.path.join(self.path_fig_tsnr, "data", "tsnr_slicewise_avg_in_PAM50.nii.gz")

        self.fname_tsnr_metrics = os.path.join(self.path_tsnr, "tsnr_metrics.csv")

    def generate_tsnr_maps_and_csv(self):
        df_tsnr = pd.DataFrame(columns=["IDs", "task", "acq", "Mean tSNR"])

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
        # Minimum number of volumes across all runs: 30 (2026-01-28)
        # Compute_tsnr
        for ID in self.IDs:
            for task in self.config["design_exp"]["task_names"]:
                for acq_name in self.config["design_exp"]["acq_names"]:
                    tag = "task-" + task + "_acq-" + acq_name

                    selected_file = self.find_moco_for_tsnr_calculation(ID, task, acq_name)
                    if selected_file is None:
                        continue

                    # Compute tSNR map in native space
                    path_tsnr_sub_folder = os.path.join(self.path_tsnr, f"sub-{ID}", tag)
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
                            df_tsnr = pd.DataFrame([[ID, task, acq_name.split("+")[0], tsnr_mean]], columns=df_tsnr.columns)
                        df_tsnr = pd.concat(
                            [pd.DataFrame([[ID, task, acq_name.split("+")[0], tsnr_mean]], columns=df_tsnr.columns), df_tsnr],
                            ignore_index=True)

        # Keep only 'rest' rows for IDs that have both 'motor' and 'rest'
        if not os.path.exists(self.fname_tsnr_metrics.split(".csv")[0]+"_reduced.csv"):
            ids_with_both = df_tsnr.groupby('IDs')['task'].apply(
                lambda x: set(['motor', 'rest']).issubset(set(x))
            )
            ids_with_both = ids_with_both[ids_with_both].index
            df_reduced = df_tsnr[~((df_tsnr['IDs'].isin(ids_with_both)) & (df_tsnr['task'] == 'motor'))]
            df_reduced.to_csv(self.fname_tsnr_metrics.split(".csv")[0]+"_reduced.csv", index=False)

        if not os.path.exists(self.fname_tsnr_metrics):
            df_tsnr.to_csv(self.fname_tsnr_metrics, index=False)

        self.pair_ttest(csv_file=self.fname_tsnr_metrics,redo=self.redo)
        

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

    def generate_average_tsnr_in_pam50(self, IDs=None,acq_name=None,tsnr_fnames=None,seg_fnames=None, warp_fnames=None,fname_mask=None, redo=False):
        
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
        os.makedirs(os.path.dirname(fname_tsnr_avg),exist_ok=True)

        # --- Load mask if provided ---
        if fname_mask is not None:
            nii_mask = nib.load(fname_mask)
            mask_data = nib.as_closest_canonical(nii_mask).get_fdata().astype(bool)
        else:
            mask_data = None

        if not os.path.exists(fname_tsnr_avg) or redo:
            print(fname_tsnr_avg)
            for i,ID in enumerate(IDs):
                tsnr_path=self.first_level_dir.format("tsnr",ID)
                tsnr_basename=tsnr_fnames[i].split("moco")[0]
                fname_tsnr_in_template=glob.glob(tsnr_basename + "moco_tsnr_in_PAM50.nii.gz")[0]

                nii_roi = self.count_roi_in_template(tsnr_fnames[i],
                                                     ID, 
                                                     acq_name,
                                                     seg_fnames[i],
                                                     warp_fnames[i],
                                                     fname_template,
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
            if mask_data is not None:
                if mask_data.shape != data_tsnr_avg.shape:
                    raise ValueError(f"Mask shape {mask_data.shape} does not match data shape {data_tsnr_avg.shape}")
                data_tsnr_avg[~mask_data] = 0
            
            nii_tsnr_avg = nib.Nifti1Image(data_tsnr_avg, affine=nii_tsnr.affine,
                                                    header=nii_tsnr.header)
                                                    
            nib.save(nii_tsnr_avg, fname_tsnr_avg)

        return fname_tsnr_avg
    
    def count_roi_in_template(self,fname_tnsr, ID, tag, fname_seg, fname_warp_from_func_to_template,
                          fname_template, redo):
        fname_ones_in_func = os.path.join(os.path.dirname(fname_tnsr), f"sub-{ID}_{tag}_ones.nii.gz")
        fname_ones_in_template = os.path.join(os.path.dirname(fname_tnsr), f"sub-{ID}_{tag}_ones_in_PAM50.nii.gz")
        if not os.path.exists(fname_ones_in_func) or redo:
            nii_tmp = nib.load(fname_seg)
            data_ones = np.ones_like(nii_tmp.get_fdata())
            nii_ones = nib.Nifti1Image(data_ones, affine=nii_tmp.affine, header=nii_tmp.header)
            nib.save(nii_ones, fname_ones_in_func)

        if not os.path.exists(fname_ones_in_template) or redo:
            cmd_coreg = f"sct_apply_transfo -i {fname_ones_in_func} -d {fname_template} -w {fname_warp_from_func_to_template} -o {fname_ones_in_template}"
            os.system(cmd_coreg)
        nii_roi = nib.load(fname_ones_in_template)
        return nii_roi
    
    def find_moco_for_tsnr_calculation(self,ID, task, acq_name):
        files = glob.glob(os.path.join(
            self.config["raw_dir"],
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
    
    def pair_ttest(self, df=None, csv_file=None, output_fname=None,index='IDs', value_col='Mean tSNR', acq_col='acq', cond1='shimSlice', cond2='shimBase',task_filter=None, task_col='task', redo=False):

        if output_fname==None and csv_file:
            output_fname=csv_file.split('.csv')[0] + "_stats.csv"

        if not os.path.exists(output_fname) or redo:
            if csv_file:
                df = pd.read_csv(csv_file)

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
                'task_filter'   : task_filter if task_filter else 'all',
                'N_pairs'       : len(df_pivot),
                'mean_cond1'    : df_pivot[cond1].mean(),
                'std_cond1'     : df_pivot[cond1].std(),
                'mean_cond2'    : df_pivot[cond2].mean(),
                'std_cond2'     : df_pivot[cond2].std(),
                't_stat'        : t_stat,
                'df'            : degrees_of_freedom,
                'p_value'       : p_value,
                'significance'  : stars,
            }])

            results.to_csv(output_fname, index=False)
            print(results.to_string(index=False))

        return pd.read_csv(output_fname)

    
