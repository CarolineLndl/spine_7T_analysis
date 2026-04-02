# -*- coding: utf-8 -*-
# General Imports
import os, glob, shutil, re, json, fnmatch
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# plotting imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# neuroimaging imports
import nibabel as nib

#custom imports
import utils


#####################################################
class Preprocess_main:
    """
    The Preprocess_main class is used to setup the preprocessing directories and files according to the config file

    Attributes
    ----------
    config : dict
        Defining all the parameters of the analysis including the path to the raw data, the participants to analyze,
        the structure to preprocess (brain/spinal cord), the design of the experiment, and the preprocessing parameters

    verbose : bool
        Whether to print information during the each step (default: True)
    """

    def __init__(self, config, ana_contrast="T2star",IDs=None,verbose=True):
        if verbose:
            print("All the raw data should be stored in BIDS format")
            print(" ")
        if IDs==None:
            raise ValueError("Please provide the participant ID (e.g., _.stc(ID='A001')).")


        # Class attributes -------------------------------------------------------------------------------------
        self.config = config # load config info
        self.participant_IDs= IDs # list of the participants to analyze
        self.raw_dir = os.path.join(self.config["raw_dir"])  # directory of the raw data
        self.derivatives_dir = os.path.join(self.config["raw_dir"], self.config["derivatives_dir"])  # directory of the derivatives data
        self.manual_dir = os.path.join(self.config["raw_dir"], self.config["manual_dir"])  # directory of the manual corrections
        self.qc_dir = os.path.join(self.config["raw_dir"], self.config["preprocess_dir"]["QC_dir"])  # directory of the QC outputs

        # Create directories -------------------------------------------------------------------------------------
        os.makedirs(self.qc_dir, exist_ok=True)

        # Create participant directories (if not already existed)
        for ID in self.participant_IDs:
            if "preprocess_dir" in self.config.keys():
                ID_preproc_dir = os.path.join(self.config["raw_dir"], os.path.expandvars(self.config["preprocess_dir"]["main_dir"].format(ID))) # directory of the preprocess data

                if not os.path.exists(ID_preproc_dir):
                    os.makedirs(ID_preproc_dir)
                    # create 1 folder per session if there are multiple sessions (exemple multiple days of acquisition)
                    for ses_name in self.config["design_exp"]['ses_names']:
                        ses_dir = ses_name if int(self.config["design_exp"]["ses_nb"])>0 else ""
                        if ses_dir != "":
                            os.makedirs(os.path.join(ID_preproc_dir, ses_dir), exist_ok=True)

                        os.mkdir(os.path.join(ID_preproc_dir, ses_dir, "anat"))  # create anat folder
                        os.mkdir(os.path.join(ID_preproc_dir, ses_dir, "func"))  # create func folder

                        # spinal cord or brain subfolders will be created if two structures are specified in the config file
                        if len(self.config["structures"])>1:
                            for structure in self.config["structures"]:
                               os.mkdir(os.path.join(ID_preproc_dir, ses_dir, "anat", structure))

                    print("New folders in preprocess dir have been created") if verbose==True else None

            #create manual correction folder if not already existing (>>> improve: add session folders)
            if not os.path.exists(os.path.join(self.manual_dir, f"sub-{ID}")):
                ID_manual_dir = os.path.join(self.manual_dir, f"sub-{ID}")
                os.mkdir(ID_manual_dir)
                for ses_name in self.config["design_exp"]['ses_names']:
                    ses_dir = ses_name if int(self.config["design_exp"]["ses_nb"])>0 else ""
                    if ses_dir != "":
                        os.makedirs(os.path.join(ID_manual_dir, ses_dir), exist_ok=True)  # create session folder

                    os.mkdir(os.path.join(ID_manual_dir, ses_dir, "anat"))  # create anat folder
                    os.mkdir(os.path.join(ID_manual_dir, ses_dir, "func"))  # create anat folder


            # Create a folder for each runs in func folder
            if "design_exp" in self.config.keys():
                for ses_name in self.config["design_exp"]['ses_names']:
                    ses_dir=ses_name if int(self.config["design_exp"]["ses_nb"])>1 else ""
                     # if "acq_names" exist in the config file
                    if "acq_names" in self.config["design_exp"].keys():
                        for task_name in self.config["design_exp"]['task_names']:
                            for acq_name in self.config["design_exp"]['acq_names']:
                                tag="task-" + task_name + "_acq-" + acq_name
                                os.makedirs(os.path.join(ID_preproc_dir, ses_dir, "func", tag), exist_ok=True)

                    else:
                        for task_name in self.config["design_exp"]['task_names']:
                            task_dir=task_name if int(self.config["design_exp"]["task_nb"])>1 else ""
                            os.makedirs(os.path.join(ID_preproc_dir, ses_dir, "func", task_dir) ,exist_ok=True)


            # copy raw anatomical file to preprocess folder anat directory ------------------------------------------------------------------
            print(os.path.join(self.raw_dir, f"sub-{ID}", "anat", self.config["preprocess_f"]["anat_raw"].format(ID,"*")))
            raw_anat=glob.glob(os.path.join(self.raw_dir, f"sub-{ID}", "anat", self.config["preprocess_f"]["anat_raw"].format(ID,"*")))[0]

            if not os.path.exists(os.path.join(ID_preproc_dir, "anat", os.path.basename(raw_anat))):
                shutil.copy(raw_anat, os.path.join(ID_preproc_dir, "anat"))


class Preprocess_Sc:
    """
    The Preprocess class is used to compute spinal cord preprocessing
    Motion correction, segmentation, vertebrae labeling, registration to template
    All functions are based on the spinal cord toolbox (SCT v7.2.dev0)


    Attributes
    ----------
    config : dict
        Defining all the parameters of the analysis including the path and the participants to analyze

    """
    def __init__(self, config, IDs=None):

        if IDs==None:
            raise ValueError("Please provide the participant ID (e.g., _.stc(ID='A001')).")

        # Class attributes -------------------------------------------------------------------------------------
        self.config = config # load config info
        self.participant_IDs= IDs # list of the participants to analyze
        self.raw_dir = self.config["raw_dir"]  # directory of the raw data
        self.derivatives_dir = os.path.join(self.config["raw_dir"], self.config["derivatives_dir"])  # directory of the derivatives data
        self.manual_dir = os.path.join(self.config["raw_dir"], self.config["manual_dir"])  # directory of the manual corrections
        self.qc_dir = os.path.join(self.config["raw_dir"], self.config["preprocess_dir"]["QC_dir"])  # directory of the QC outputs
        self.preprocessing_dir = os.path.join(self.config["raw_dir"], os.path.expandvars(self.config["preprocess_dir"]["main_dir"]))  # directory of the preprocess data
        self.code_dir = self.config["code_dir"]  # directory of the code

        # Check the structure type ----------------------------------------------------------------
        if len(self.config["structures"])>1:
            self.structure = "spinalcord" # structure subfolder if two structures are specified
        else:
            self.structure = "" # no structure subfolder if only one structure is specified


    def moco_mask(self,ID=None,i_img=None,o_folder=None, radius_size=15,ses_name='',task_name='', tag='',manual=False,redo_ctrl=False,redo_mask=False,verbose=True):

        """
        This function creates a mask around a spinal cord centerline.

        References:
        -----------
        - https://spinalcordtoolbox.com/user_section/command-line.html#sct-get-centerline
        - https://spinalcordtoolbox.com/user_section/command-line.html#sct-create-mask

        Attributes:
        -----------
        ID : str
            Name of the participant (default: None; an error will be raised if not provided).
        i_img : str
            Input filename of the functional image (default: None; an error will be raised if not provided).
        o_folder : str
            Output folder name (default: None; if not provided, the input folder will be used).
        radius_size : int
            Diameter of the surrounding mask in voxels (default: 15).
        ses_name : str
            Session name, if applicable (should include the 'ses-' prefix in BIDS format).
        task_name : str
            Task name, if applicable (should include the 'task-' prefix in BIDS format).
        manual : bool
            Whether to use manual drawing of the centerline (default: False).
        redo_ctrl : bool
            Whether to redo the centerline creation (default: False).
        redo_mask : bool
            Whether to redo the mask creation (default: False). This step should be repeated if the centerline has been modified.
        verbose : bool
            Whether to display information and generate quality control plots (default: True).

        Outputs:
        --------
        centerline_f : str
            Filename of the created centerline.
        mask_f : str
            Filename of the created mask.
        """


        # --- Input validation -------------------------------------------------------------
        if ID is None:
            raise ValueError("Please provide the participant ID (e.g., _.stc(ID='A001')).")
        if i_img is None:
            raise ValueError("Please provide the filename of the input image.")

        # --- Define directories -----------------------------------------------------------
        preprocess_dir = self.preprocessing_dir.format(ID)

        # --- Define method and output folder ----------------------------------------------
        if manual:
            method = "viewer"
            o_folder = os.path.join(self.manual_dir, f"sub-{ID}", "{ses_name}", "func")
        else:
            method = "optic"
            if o_folder is None : # gave the default folder name if not provided
                o_folder = os.path.join(preprocess_dir, ses_name, self.config["preprocess_dir"]["func_mask"].format(task_name))

        os.makedirs(os.path.join(o_folder, self.structure), exist_ok=True)
        centerline_f = os.path.join(o_folder, self.structure, os.path.basename(i_img).split(".")[0] + "_centerline")  # output centerline filename without extension

        # --- Create mask output folder ----------------------------------------------------
        mask_o_folder = os.path.join(preprocess_dir, ses_name, self.config["preprocess_dir"]["func_mask"].format(task_name))
        os.makedirs(os.path.join(mask_o_folder, self.structure), exist_ok=True)
        mask_f = os.path.join(mask_o_folder, self.structure, os.path.basename(i_img).split(".")[0] + "_mask.nii.gz")  # output mask filename

        # --- Compute centerline -----------------------------------------------------------
        if not os.path.exists(centerline_f + ".nii.gz") or redo_ctrl:
            print(f"Centerline for sub-{ID}")
            cmd_centerline=f"sct_get_centerline -i {i_img} -o {centerline_f} -c t1 -method {method} -centerline-algo bspline -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
            os.system(cmd_centerline)

        # --- Create mask around centerline ------------------------------------------------
        if not os.path.exists(mask_f) or redo_mask:
            print(f"Create a mask for sub-{ID}")
            cmd_mask=f"sct_create_mask -i {i_img} -p centerline,{centerline_f}.nii.gz -size {radius_size} -o {mask_f} -v 0"
            os.system(cmd_mask)

        # --- Validate mask and image dimensions ------------------------------------------
        img_4d = nib.load(i_img) # load the 4D image
        mask_3d = nib.load(mask_f) # load the 3D mask
        if img_4d.shape[:3] != mask_3d.shape[:3]:
            raise ValueError(
            f"Mask and functional image dimensions do not match.\n"
            f"Check with: fsleyes {mask_f} {i_img}\n"
            f"Possible cause: centerline does not start at the first slice."
            )

        # --- QC handling -----------------------------------------------------------------
        manual_file = os.path.join(self.manual_dir, f"sub-{ID}", ses_name, "func", os.path.basename(i_img).split(".")[0] + "_centerline.nii.gz")

        if os.path.exists(manual_file):
            centerline_f = manual_file.split(".nii.gz")[0]
            print(f"⚠ A manual centerline file exists: {manual_file}")
            print("The manual centerline is prioritized. Remove it to use the automatic version.")

            if manual and redo_ctrl:
                print("Running QC for manual centerline...")
                cmd_qc = f"sct_qc -i {i_img} -s {centerline_f}.nii.gz -p sct_get_centerline -qc {self.qc_dir } -qc-subject sub-{ID} -v 0"
                os.system(cmd_qc)

        # --- Generate QC plot -------------------------------------------------------------
        if verbose:
            qc_indiv_path = os.path.join(self.qc_dir, f"sub-{ID}", "func", ses_name, task_name, "sct_get_centerline")
            self._plot_qc(ID=ID, ses_name=ses_name, task_name=task_name, tag="centerline", qc_indiv_path=qc_indiv_path, fig_size=(15,15),alpha=0.8)

            if not os.path.exists(manual_file):
                print("If manual corrections are needed, set:")
                print("manual=True, redo_ctrl=True, redo_mask=True")
                print("⚠ Ensure the centerline starts at the first slice.")

        return centerline_f +'.nii.gz', mask_f

    def moco(self,ID=None,i_img=None,mask_img=None,ref_img=None,o_folder=None,params=None,ses_name='',task_name='',run_name="",redo=False,verbose=True):

        """
        This function performs motion correction on functional (fMRI) images using a mask around the spinal cord.
        It also plots motion parameters for visual inspection.

        Reference:
        ----------
        - https://spinalcordtoolbox.com/user_section/command-line.html#sct-fmri-moco

        Attributes:
        -----------
        ID : str
            Name of the participant (default: None; an error will be raised if not provided).
        i_img : str
            Input filename of the 4D functional images (default: None; an error will be raised if not provided).
        mask_img : str
            Filename of the binary mask used to restrict voxels considered by the registration metric (default: None; an error will be raised if not provided).
        ref_img : str
            Reference image for motion correction (default: None; if not provided, the first volume of the input image is used).
        o_folder : str
            Output folder (default: None; if not provided, the input folder will be used).
        params : str
            Motion correction parameters (default: None).
            Default parameters are predefined in this function but can be modified if necessary.
            See the Spinal Cord Toolbox documentation for more details. These parameters should remain consistent across participants in the same study.
        ses_name : str
            Session name, if applicable (should include the 'ses-' prefix in BIDS format).
        task_name : str
            Task name, if applicable (should include the 'task-' prefix in BIDS format).
        run_name : str
            Run name, if applicable (should include the 'run-' prefix in BIDS format).
        redo : bool
            Whether to redo the motion correction step (default: False).
        verbose : bool
            Whether to display information and generate quality control plots (default: True).

        Outputs:
        --------
        moco_file : str
            Filename of the motion-corrected fMRI volumes: *_sc_moco.nii.gz
        moco_mean_file : str
            Filename of the mean (time-averaged) corrected fMRI volumes: *_sc_moco_mean.nii.gz
        qc_indiv_dir : str
            Directory containing quality control files for motion correction.

        Additional outputs saved in the output folder:
        ----------------------------------------------
        - FD_mean.txt — Text file containing the mean framewise displacement (in mm).
        - moco_params_task-*.tsv — TSV file containing motion correction parameters for all volumes.
        - moco_params_abs_task-*.txt — TXT file containing absolute motion correction parameters for all volumes.
        - moco_params_x_task-*.nii.gz and moco_params_y_task-*.nii.gz — 3D NIfTI files containing motion correction parameters along the X and Y directions, respectively.
        """


        # --- Validate inputs --------------------------------------------------------------
        if ID is None:
            raise ValueError("Please provide a participant ID (e.g., _.stc(ID='A001')).")
        if i_img is None:
            raise ValueError("Please provide the input image filename.")
        if mask_img is None:
            raise ValueError("Please provide the mask image filename.")



        # --- Define main folders ----------------------------------------------------------
        preprocess_dir =self.preprocessing_dir.format(ID)
        if o_folder is None : # gave the default folder name if not provided
            o_folder = os.path.join(preprocess_dir, ses_name, self.config["preprocess_dir"]["func_moco"].format(task_name))

        os.makedirs(o_folder, exist_ok=True)
        os.makedirs(os.path.join(o_folder, self.structure), exist_ok=True)

        run_tag = f"_{run_name}" if run_name else ""
        task_tag = f"_{task_name}" if task_name else ""

        moco_file = os.path.join(o_folder, self.structure, os.path.basename(i_img).split(".")[0] + "_moco.nii.gz")
        moco_mean_file = os.path.join(o_folder, self.structure, os.path.basename(i_img).split(".")[0] + "_moco_mean.nii.gz")

        # --- Define motion correction parameters -----------------------------------------
        if params is None:
            params = 'poly=0,smooth=1,metric=MeanSquares,gradStep=1,sampling=0.2'
            print(f"Using default motion correction parameters: {params}")

        # --- Run motion correction --------------------------------------------------------
        if not os.path.exists(moco_file) or redo:
            print(f">>>>> Running motion correction for sub-{ID}...")
            cmd=f"sct_fmri_moco -i {i_img} -m {mask_img} -param {params} -ofolder {os.path.join(o_folder, self.structure)} -x spline -g 1 -r 1 -qc {self.qc_dir} -qc-subject sub-{ID} -qc-seg {mask_img} -v 0"
            if ref_img is not None:
                cmd += f" -ref {ref_img}"
            os.system(cmd)

            # Rename output parameter files for clarity
            for dim in ["x","y"]:
                os.rename(os.path.join(os.path.dirname(moco_file), f"moco_params_{dim}.nii.gz"), os.path.join(os.path.dirname(moco_file), f"moco_params_{dim}{task_tag}{run_tag}.nii.gz"))

            params_tsv = os.path.join(o_folder, 'moco_params.tsv'.split('.')[0] + task_tag + run_tag + '.tsv')
            os.rename(os.path.join(o_folder, 'moco_params.tsv'), params_tsv)

            # # Re-run moco with nearest neighbor interpolation for tSNR computation
            # Commented out until this is resolved: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/5157
            # o_folder_nn = o_folder[:-1] + self.structure + "_nn"
            # cmd_nn=f"sct_fmri_moco -i {i_img} -m {mask_img} -param {params} -ofolder {o_folder_nn} -x nn -g 1 -r 1 -v 0"
            # os.system(cmd_nn)

        # --- Load and plot motion parameters ----------------------------------------------
        ## Load motion parameters
        params_tsv = os.path.join(o_folder, 'moco_params.tsv'.split('.')[0] + task_tag + run_tag + '.tsv')
        data = pd.read_csv(params_tsv, delimiter='\t')
        params_txt = os.path.splitext(params_tsv)[0] + '.txt'
        if not os.path.exists(params_txt) or redo:
            data.to_csv(params_txt,index=False, header=None)
        params_data = pd.read_csv(params_txt, delimiter=',', header=None)

        ## Plot moco parameters
        fig, axs = plt.subplots(1,1, figsize=(10, 2), facecolor='w', edgecolor='k')
        fig.tight_layout()
        fig.subplots_adjust(hspace = .5, wspace=.001)
        axs.plot(params_data[0]) # add axs[] if more than one run
        axs.set_title(f"{ID} {task_name} {run_name}")
        axs.set_ylabel("Translation (mm)")
        axs.set_xlabel("Volumes")
        if verbose:
            plt.show()

        params_pdf = os.path.splitext(params_txt)[0] + '.pdf'
        if not os.path.exists(params_pdf) or redo:
            plt.savefig(params_pdf) # save the plot
            plt.close()

        # --- Generate QC plot -------------------------------------------------------------
        diff_XY = np.abs(np.diff(params_data[0])) # Calculate Framewise displacement (abs difference of displacement between each volumes)
        meandiff=[np.mean(diff_XY)]
        if not os.path.exists(os.path.join(o_folder, self.structure+'FD_mean.txt')) or redo:
                np.savetxt(os.path.join(o_folder, self.structure, 'FD_mean.txt'), [meandiff])  # save the mean framewise displacement
        if verbose:
            print(f"sub-{ID} Diff_XY: " + str(round(meandiff[0],3)) + " mm")
            qc_indiv_path = os.path.join(self.qc_dir, f"sub-{ID}", "func", ses_name, task_name, "sct_fmri_moco", "sct_fmri_moco")
            qc_indiv_dir=utils.get_latest_dir(base_dir=qc_indiv_path)

        return moco_file, moco_mean_file, qc_indiv_dir if verbose==True else None

    def segmentation(self,ID=None,i_img=None,i_gm_img=None,o_folder=None,mask_qc=None,task_name='',ses_name='',tag='',tissue=None,img_type="anat",contrast_anat="t1",redo=False,redo_qc=False,verbose=True):

        """
        This function segments the spinal cord.
        Visual inspection and manual corrections are required.

        Reference:
        ----------
        - https://spinalcordtoolbox.com/user_section/command-line.html#sct-propseg

        Attributes:
        -----------
        ID : str
            Name of the participant (default: None; an error will be raised if not provided).
        i_img : str
            Input filename of the 3D functional image (default: None; an error will be raised if not provided).
        i_gm_img : str
            Input filename for the gray matter segmentation file (used when img_type="anat" and tissue="wm").
        o_folder : str
            Output folder (default: None; if not provided, the input folder will be used).
        mask_qc : str
            Mask image used for quality control when img_type="func" (default: None).
        task_name : str
            Task name, if applicable (should include the 'task-' prefix in BIDS format).
        ses_name : str
            Session name, if applicable (should include the 'ses-' prefix in BIDS format).
        tag : str
            Additional filename specification (default: '').
        tissue : str
            Type of tissue to segment — "wm" or "gm" for anatomical images, or "csf" for functional images (default: None, which segments the spinal cord).
        img_type : str
            Type of input image, either "func" or "anat" (default: "anat").
        contrast_anat : str
            Contrast type of the anatomical image: "t1", "t2", "t2s", or "t2star" (default: "t1").
        redo : bool
            Whether to redo the segmentation step (default: False).
        redo_qc : bool
            Whether to redo the quality control step for manual segmentation only (default: False).
        verbose : bool
            Whether to display information and generate quality control plots (default: True).

        Outputs:
        --------
        o_img : str
            Filename of the segmentation output file.
        """

        # --- Input checks --------------------------------------------------------------------------------
        if ID is None:
            raise Warning("Please provide the ID of the participant, e.g., _.stc(ID='A001')")
        if i_img is None:
            raise Warning("Please provide the input filename.")
        if img_type == "anat" and tissue == "wm" and i_gm_img is None:
            raise Warning("GM segmentation must be done first; provide i_gm_img='mysegfile.nii.gz'.")
        if img_type == "func" and tissue==None and mask_qc is None:
            raise Warning("Provide a mask around the cord for QC computation: mask_qc='mymask.nii.gz'")


        # --- Define folders and filenames ----------------------------------------------------------------
        preprocess_dir = self.preprocessing_dir.format(ID)

        if o_folder is None : # gave the default folder name if not provided
            if img_type=="func":
                key = f"{img_type}_csf_seg" if tissue == "csf" else f"{img_type}_seg"
                o_folder = os.path.join(preprocess_dir, ses_name, self.config["preprocess_dir"][key].format(task_name, self.structure))

            else:
                o_folder = os.path.join(preprocess_dir, ses_name, self.config["preprocess_dir"][f"{img_type}_seg"])


        os.makedirs(o_folder, exist_ok=True)

        # --- Define output filenames ---------------------------------------------------------------------
        base_name = os.path.basename(i_img)
        if img_type == "func" and tag == '':
            tag = "mean_seg.nii.gz"
            o_img = os.path.join(o_folder, base_name.split('mean.')[0] + tag)
        elif img_type != "func" and tag == '':
            tag = "_seg.nii.gz"
            o_img = os.path.join(o_folder, base_name.split('.')[0] + tag)
        elif tissue == "wm":
            o_img = os.path.join(o_folder, base_name.split('_cord')[0] + tag)
        else:
            o_img = os.path.join(o_folder, base_name.split('.')[0] + tag)

        # --- Manual segmentation paths -------------------------------------------------------------------
        if img_type == "func":
            if tissue == "csf":
                o_manual = os.path.join(self.manual_dir, f"sub-{ID}", "func", base_name.split(".nii.gz")[0] + "_CSF_seg.nii.gz")
            else:
                o_manual = os.path.join(self.manual_dir, f"sub-{ID}", "func", os.path.basename(o_img))
        else:
            o_manual = os.path.join(self.manual_dir, f"sub-{ID}", "anat", os.path.basename(o_img))

        # --- Run segmentation ----------------------------------------------------------------------------
        if not (os.path.exists(o_img) or os.path.exists(o_manual)) or redo:
            print(f">>>>> Segmentation is running for {img_type} image of sub-{ID}...")

            if img_type=="func":
                if tissue is None:
                    cmd=f"sct_deepseg sc_epi -i {i_img} -o {o_img} -qc {self.qc_dir } -qc-subject sub-{ID} -qc-seg {mask_qc} -v 0" #segmentation
                elif tissue=="csf":
                    cmd_propseg=f"sct_propseg -i {i_img} -c {contrast_anat} -CSF -o {o_img}" #segmentation
                    os.system(cmd_propseg) # run propseg
                    csf_mask=glob.glob(os.path.join(os.path.dirname(o_img), "*_CSF_*"))[0] # filename of the CSF segmentation
                    cmd=f"sct_qc -i {i_img} -s {csf_mask} -p sct_propseg -qc {self.qc_dir } -qc-subject sub-{ID} -v 0"

            elif img_type!="func":
                if tissue=="gm":
                    cmd=f"sct_deepseg graymatter -i {i_img} -c {contrast_anat} -thr 0.01 -o {o_img} -qc {self.qc_dir } -qc-subject sub-{ID} -v 0"
                elif tissue=="wm":
                    cmd_fslmaths=f"fslmaths {i_img} -sub {i_gm_img} {o_img}"
                    os.system(cmd_fslmaths) # substract cord and gm segmentation to obtain wm
                    cmd=f"fslmaths {o_img} -thr 0 {o_img}" # threshold the mask at 0
                else:
                    cmd=f"sct_deepseg spinalcord -i {i_img} -c {contrast_anat} -thr 0.01 -o {o_img} -qc {self.qc_dir} -qc-subject sub-{ID} -v 0" # segmentation

            os.system(cmd) # run the process

        # --- Use manual segmentation if available ---------------------------------------------------------

        if os.path.exists(o_manual):
            o_img=o_manual
            print("/!\\ Manual segmentation file detected — using it as output.")
            # Generate QC report
            cmd_qc=f"sct_qc -i {i_img} -s {o_manual} -p sct_deepseg_sc -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
            os.system(cmd_qc)

        # --- Generate QC plot -----------------------------------------------------------------------------
        if verbose:
            if img_type=="func":
                if os.path.exists(o_manual):
                    if redo_qc==True:
                        if tissue==None:
                            cmd_qc=f"sct_qc -i {i_img} -s {o_manual} -p sct_deepseg_sc -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
                        elif tissue=="csf":
                            cmd_qc=f"sct_qc -i {i_img} -s {o_manual} -p sct_propseg -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
                        os.system(cmd_qc)

                    ## QC path
                    if tissue==None:
                        qc_indiv_path = os.path.join(self.qc_dir, f"sub-{ID}", "func", ses_name, task_name, "sct_fmri_moco", "sct_deepseg_sc")
                    else:
                        qc_indiv_path = os.path.join(self.qc_dir, f"sub-{ID}", "func", ses_name, task_name, "sct_fmri_moco", "sct_propseg")
                else:
                    if tissue==None:
                        qc_indiv_path = os.path.join(self.qc_dir, f"sub-{ID}", "func", ses_name, task_name, "sct_fmri_moco", "sct_deepseg")
                    else:
                        qc_indiv_path = os.path.join(self.qc_dir, f"sub-{ID}", "func", ses_name, task_name, "sct_fmri_moco", "sct_propseg")
            else:
                if redo_qc and os.path.exists(o_manual):
                    # rerun QC for anat img_type
                    cmd_qc=f"sct_qc -i {i_img} -s {o_img} -p sct_deepseg_sc -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
                    os.system(cmd_qc)
                    qc_indiv_path = os.path.join(self.qc_dir, self.qc_dir.split("/")[-3], f"sub-{ID}", "anat", "sct_deepseg_sc")  # QC path
                else:
                    qc_indiv_path = os.path.join(self.qc_dir, self.qc_dir.split("/")[-3], f"sub-{ID}", "anat", "sct_deepseg") # QC path

            ## plot qc
            self._plot_qc(ID=ID, ses_name=ses_name, task_name=task_name, tag="segmentation", qc_indiv_path=qc_indiv_path, fig_size=(10,10),alpha=0.8)

        return o_img


    def label_vertebrae(self, ID=None, i_img=None, seg_img=None, c="t2", labels=range(1, 15), auto=True, o_folder=None, ses_name="", task_name="", tag="", labels_to_keep=None, redo=False, verbose=True):
        """
        Labels vertebrae automatically (totalspineseg) or manually (labels provided in derivatives/).

        Outputs:
        --------
        label_file : str
            Labeled vertebrae / disc image filename.
        labels_to_keep : tuple
            Label values to keep
        """

        # --- Input checks ------------------------------------------------------------------
        if ID is None:
            raise Warning("Please provide participant ID, e.g., _.stc(ID='A001')")
        if i_img is None:
            raise Warning("Please provide input filename")

        # --- Define output folder -----------------------------------------------------------
        preprocess_dir = self.preprocessing_dir.format(ID)

        if o_folder is None:
            if auto:
                o_folder = os.path.join(preprocess_dir, "anat", "sct_deepseg_totalspineseg")
            else:
                o_folder = os.path.join(self.manual_dir, f"sub-{ID}", "anat")

        os.makedirs(o_folder, exist_ok=True)

        base_name = os.path.basename(i_img).split(".")[0]

        if auto:
            label_file = os.path.join(
                o_folder, f"{base_name}_totalspineseg_discs.nii.gz"
            )
        else:
            label_file = os.path.join(
                o_folder, f"{base_name}_space-orig_label-ivd_mask.nii.gz"
            )

        # --- Manual segmentation paths -------------------------------------------------------------------

        o_manual = os.path.join(self.manual_dir, f"sub-{ID}, anat", f"{base_name}_space-orig_label-ivd_mask.nii.gz")

        # --- Run labeling -------------------------------------------------------------------
        if not (os.path.exists(label_file) or os.path.exists(o_manual)) or redo:
            if auto:
                print(f">>>>> Running totalspineseg for sub-{ID}...")

                fname_out = os.path.join(o_folder, f"{base_name}.nii.gz")
                cmd = (f"sct_deepseg spine -i {i_img} -o {fname_out} -qc {self.qc_dir} -qc-subject sub-{ID}")

            else:
                nb = labels
                print(
                    f">>>>> Place labels manually at the posterior tip of each inter-vertebral disc "
                    f"for sub-{ID}..."
                )

                cmd = (
                    "sct_label_utils "
                    f"-i {i_img} "
                    f"-o {label_file} "
                    f"-qc {self.qc_dir} "
                    f"-qc-subject sub-{ID} "
                    f"-create-viewer "
                    + ",".join(map(str, nb))
                )

            os.system(cmd)

        labels_file_to_keep = label_file.split(".nii.gz")[0] + "_labels_to_keep.nii.gz"
        if labels_to_keep is not None:
            if not os.path.exists(labels_file_to_keep) or redo:
                cmd = f"sct_label_utils -i {label_file} -o {labels_file_to_keep} -keep {','.join(map(str, labels_to_keep))} -qc {self.qc_dir} -qc-subject sub-{ID}"
                os.system(cmd)

        # --- Use manual segmentation if available ---------------------------------------------------------

        if os.path.exists(o_manual):
            o_img=o_manual
            print("/!\\ Manual segmentation file detected — using it as output.")
            # Generate QC report
            cmd_qc=f"sct_qc -i {i_img} -s {o_manual} -p sct_label_utils -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
            os.system(cmd_qc)

        # --- QC visualization ---------------------------------------------------------------
        if verbose:
            if auto:
                qc_indiv_path = os.path.join(self.qc_dir, self.qc_dir.split("/")[-3], f"sub-{ID}", "anat", "sct_totalspineseg")
                tag = "automatic vertebra labeling (totalspineseg)"
            else:
                qc_indiv_path = os.path.join(self.qc_dir, self.qc_dir.split("/")[-3], f"sub-{ID}", "anat", "sct_label_utils")
                tag = "manual vertebra labeling"

            self._plot_qc(
                ID=ID,
                ses_name=ses_name,
                task_name=task_name,
                tag=tag,
                qc_indiv_path=qc_indiv_path,
                fig_size=(5, 5),
                alpha=0.8,
            )

            print(" ")

        if labels_to_keep is not None:
            return labels_file_to_keep
        else:
            return label_file

    def coreg_anat2PAM50(self,ID=None,i_img=None,o_folder=None,seg_img=None,labels_img=None,img_type="t2",param=None,ses_name='',task_name='',tag='T2w', redo=False,verbose=True):

        """
        Registers anatomical image to the PAM50 template and warps the template into anatomical space.

        References:
        ----------
        - https://spinalcordtoolbox.com/stable/user_section/command-line/sct_register_to_template.html
        - https://spinalcordtoolbox.com/stable/user_section/command-line/sct_warp_template.html

        Attributes:
        -----------
        ID : str
            Participant ID (required).
        i_img : str
            Input anatomical image filename (required).
        o_folder : str
            Output folder (default: None; created based on ID and tag).
        seg_img : str
            Spinal cord segmentation filename (required).
        labels_img : str
            Intervertebral disc label image (required).
        img_type : str
            Image contrast for registration ("t1" or "t2", default: "t2").
        param : str
            Registration parameters (default: centermassrot + syn, see SCT docs).
        ses_name : str
            Session name (include the 'ses-' prefix if BIDS format).
        task_name : str
            Task name (include the 'task-' prefix if BIDS format).
        tag : str
            Tag for output folder (default: "T2w").
        redo : bool
            Redo registration if files exist (default: False).
        verbose : bool
            Display messages and QC plots (default: True).

        Outputs:
        --------
        warp_from_PAM502anat : str
            Filename of warping field from template to anatomical space.
        warp_from_anat2PAM50 : str
            Filename of warping field from anatomical image to template.
        """

        # --- Input checks --------------------------------------------------------------------------------
        if ID is None:
            raise Warning("Please provide participant ID, e.g., _.stc(ID='A001')")
        if i_img is None or seg_img is None or labels_img is None:
            raise Warning("Provide i_img, seg_img, and labels_img")

        if param is None:
            param = "step=1,type=seg,algo=centermassrot:step=2,type=im,algo=syn,iter=5,slicewise=1,metric=CC,smooth=0"

        # --- Define output folder and filenames ----------------------------------------------------------
        preprocess_dir = self.preprocessing_dir.format(ID)

        if o_folder is None:
            o_folder = os.path.join(preprocess_dir, f"{self.config['preprocess_dir'][tag + '_coreg']}")

        os.makedirs(o_folder, exist_ok=True)

        warp_from_anat2PAM50 = os.path.join(o_folder, f"sub-{ID}_from-{tag}_to-PAM50_mode-image_xfm.nii.gz") # warping field form anat to PAM50
        warp_from_PAM502anat = os.path.join(o_folder, f"sub-{ID}_from-PAM50_to-{tag}_mode-image_xfm.nii.gz") # warping field form PAM50 to anat

        # --- Use manual segmentation if available ---------------------------------------------------------
        manual_seg = os.path.join(self.manual_dir, f"sub-{ID}", ses_name, "anat", os.path.basename(seg_img))
        if os.path.exists(manual_seg):
            print("Segmentation file will be the manually corrected file")
            seg_img = manual_seg

         # --- Use manual labels if available ---------------------------------------------------------
        manual_labels = glob.glob(os.path.join(self.manual_dir, f"sub-{ID}", ses_name, "anat", "*label-ivd_mask.nii.gz"))
        if manual_labels:
            print("Labels file will be the manually corrected file")
            labels_img = manual_labels[0]

        # --- Run registration ---------------------------------------------------------------------------
        if not os.path.exists(warp_from_anat2PAM50) or redo:
            cmd_coreg=f"sct_register_to_template -i {i_img} -s {seg_img} -ldisc {labels_img} -c {img_type} -param {param} -ofolder {o_folder} -qc {self.qc_dir} -qc-subject sub-{ID}"
            print(">>>>> Registration step is running for sub-" + ID)
            os.system(cmd_coreg)

            # Rename warping fields for consistency
            os.rename(os.path.join(o_folder, "warp_anat2template.nii.gz"), warp_from_anat2PAM50)
            os.rename(os.path.join(o_folder, "warp_template2anat.nii.gz"), warp_from_PAM502anat)

            # Warp template to anatomical space
            template_folder = os.path.join(o_folder, f"template_in_{tag}")
            cmd_template = f"sct_warp_template -d {i_img} -w {warp_from_PAM502anat} -s 1 -ofolder {template_folder} -a 0 -s 0"
            os.system(cmd_template)

        else:
            print("/!\\ Warping field detected — using it")
            # Generate QC report
            cmd_qc=f"sct_qc -i {i_img} -s {seg_img} -p sct_register_to_template -d {os.path.join(o_folder, 'template2anat.nii.gz')} -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
            os.system(cmd_qc)

        # --- QC visualization ---------------------------------------------------------------------------
        if verbose:
            qc_indiv_path = os.path.join(self.qc_dir, self.qc_dir.split("/")[-3], f"sub-{ID}", "anat", "sct_register_to_template")
            tag="anat2PAM50"
            self._plot_qc(ID=ID, ses_name=ses_name, task_name=task_name, tag=tag, qc_indiv_path=qc_indiv_path, fig_size=(15,15),alpha=0.5)
            print(" ")

        return warp_from_PAM502anat, warp_from_anat2PAM50

    def coreg_img2PAM50(self,ID=None,i_img=None,o_folder=None,i_seg=None,PAM50_cord=None,PAM50_t2=None,img_type="func",coreg_type="slicereg",initwarp=None,initwarpinv=None,param=None,ses_name='',task_name='',run_name="",redo=False,verbose=True):

        """
        Registers a mean functional (or other modality) image to the PAM50 template.

        References:
        ----------
        - https://spinalcordtoolbox.com/user_section/command-line.html#sct-register-multimodal

        Attributes:
        -----------
        ID : str
            Participant ID (required).
        i_img : str
            Input functional image filename (3D, required).
        o_folder : str
            Output folder (default: None; auto-generated if not provided).
        i_seg : str
            Segmentation of the functional image (3D, required).
        PAM50_cord : str
            PAM50 cord image (default self.config["PAM50_cord"]).
        PAM50_t2 : str
            PAM50 T2 template (default self.config["PAM50_t2"]).
        img_type : str
            Type of input image (default: "func").
        coreg_type : str
            Coregistration type: "slicereg" or "centermass" (default: "slicereg").
        initwarp : str
            Initial warping field from previous registration (required).
        initwarpinv : str
            Initial inverse warping field (required).
        param : str
            Parameters for registration (default depends on img_type/coreg_type).
        ses_name : str
            Session name (include the 'ses-' prefix if BIDS format).
        task_name : str
            Task name (include the 'task-' prefix if BIDS format).
        run_name : str
            Run name (include the 'run-' prefix BIDS format).
        redo : bool
            Redo registration if output exists (default: False).
        verbose : bool
            Display messages and QC plots (default: True).

        Outputs:
        --------
        o_folder : str
            Output folder used for registration.
        o_warp_img : str
            Warping field from input image to PAM50.
        o_warpinv_img : str
            Warping field from PAM50 to input image.
        """
        # --- Input checks --------------------------------------------------------------------------------
        if ID is None:
            raise Warning("Please provide participant ID, e.g., _.stc(ID='A001')")
        if i_img is None or i_seg is None or initwarp is None or initwarpinv is None:
            raise Warning("Provide i_img, i_seg, and warping fields (initwarp, initwarpinv)")

        # --- Default template files -----------------------------------------------------------------------
        if PAM50_cord is None:
            PAM50_cord = os.path.join(self.code_dir, "template", self.config["PAM50_cord"])
        if PAM50_t2 is None:
            PAM50_t2 = os.path.join(self.code_dir, "template", self.config["PAM50_t2"])

        # --- Tags for run/task ---------------------------------------------------------------------------
        run_tag = f"_{run_name}" if run_name else ""
        task_tag = f"_{task_name}" if task_name else ""

        # --- Default parameters ---------------------------------------------------------------------------
        if param==None:
            if img_type=="func" and coreg_type=="slicereg":
                param="step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=2:step=2,type=im,algo=syn,metric=CC,iter=3,slicewise=1"

            elif img_type=="func" and coreg_type=="centermass": # this option can be used for very curvate spine
                param="step=1,type=seg,algo=centermass,metric=MeanSquares,smooth=2:step=2,type=im,algo=syn,metric=CC,iter=3,slicewise=1"
            elif img_type=="t2s":
                param="step=1,type=seg,algo=rigid:step=2,type=seg,metric=CC,algo=bsplinesyn,slicewise=1,iter=3:step=3,type=im,metric=CC,algo=syn,slicewise=1,iter=2"
            elif img_type=="mtr":
                param='step=1,type=seg,algo=centermass,metric=MeanSquares:step=2,algo=bsplinesyn,type=seg,slicewise=1,iter=5'
            elif img_type=="dwi":
                param='step=1,type=seg,algo=centermass,metric=MeanSquares:step=2,algo=bsplinesyn,type=seg,slicewise=1,iter=5'

        # --- Define output folder -------------------------------------------------------------------------
        preprocess_dir = self.preprocessing_dir.format(ID)

        if img_type=="func":
            if o_folder is None : # gave the default folder name if not provided
                o_folder = os.path.join(preprocess_dir, ses_name, self.config["preprocess_dir"]["func_coreg"].format(task_name,self.structure))

        os.makedirs(o_folder, exist_ok=True)

        # --- Output filenames -----------------------------------------------------------------------------
        base_name = os.path.basename(i_img).split('.')[0]
        o_img = os.path.join(o_folder, f"{base_name}_coreg_in_PAM50.nii.gz")
        o_warp_img = os.path.join(o_folder, f"sub-{ID}{task_tag}{run_tag}_from-PAM50_to_{img_type}_mode-image_xfm.nii.gz")
        o_warpinv_img = os.path.join(o_folder, f"sub-{ID}{task_tag}{run_tag}_from-{img_type}_to_PAM50_mode-image_xfm.nii.gz")

        # --- Run registration -------------------------------------------------------------------------------
        if not os.path.exists(o_img) or redo:
            print(f">>>>> Registration step running for sub-{ID}...")
            cmd_coreg=f"sct_register_multimodal -i {PAM50_t2} -iseg {PAM50_cord} -d {i_img} -dseg {i_seg} -param {param} -initwarp {initwarp} -initwarpinv {initwarpinv} -owarp {o_warp_img} -owarpinv {o_warpinv_img} -ofolder {o_folder} -x spline -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
            os.system(cmd_coreg)
            os.rename(os.path.join(o_folder, f"{base_name}_reg.nii.gz"),o_img)
            if img_type=="func":
                os.rename(glob.glob(os.path.join(o_folder, "PAM50_t2_*reg.nii.gz"))[0], os.path.join(o_folder, f"PAM50_t2_reg{run_tag}.nii.gz"))

        else:
            print("/!\\ Registration detected — using it")
            # Generate QC report
            cmd_qc=f"sct_qc -i {i_img} -s {i_seg} -p sct_register_multimodal -d {os.path.join(o_folder, f'PAM50_t2_reg{run_tag}.nii.gz')} -qc {self.qc_dir} -qc-subject sub-{ID} -v 0"
            os.system(cmd_qc)

        if verbose:
            qc_indiv_path = os.path.join(self.qc_dir, f"sub-{ID}", ses_name, "func", task_name, "sct_register_multimodal", "sct_register_multimodal")
            tag = img_type + "2PAM50"
            self._plot_qc(ID=ID, ses_name=ses_name, task_name=task_name, tag=tag, qc_indiv_path=qc_indiv_path, fig_size=(10,25),alpha=0.3)
            print(" ")

        return (o_folder, o_warp_img,o_warpinv_img)

    def apply_warp(self,i_img=None,ID=None,o_folder=None,dest_img=None,warping_field=None,ses_name='',task_name='',tag='_w',threshold=None,mean=False,method='spline',redo=False,verbose=True,n_jobs=1):
        """
        Apply warping field(s) to spinal cord input image(s) using sct_apply_transfo.

        Reference:
        ----------
        - https://spinalcordtoolbox.com/user_section/command-line.html#sct-apply-transfo

        Attributes:
        -----------
        i_img : str or list
            Input 3D image(s) to transform.
        ID : str or list
            Participant ID(s).
        o_folder : str or list
            Output folder(s). If None, uses folder of warping field.
        dest_img : str or list
            Destination image(s) for the warp. If None, defaults to PAM50 template.
        warping_field : str or list
            Warping field(s) or affine matrix(ces) to apply.
        task_name : str
            Task name (include the 'task-' prefix if BIDS format).
        run_name : str
            Run name (include the 'run-' prefix BIDS format).
        tag : str
            Tag for output filename.
        redo : bool
            If True, rerun transformation even if output exists.
        verbose : bool
            Print messages about progress.
        n_jobs : int
            Number of parallel jobs.

        Outputs:
        --------
        o_imgs : list
            Filenames of warped images.
        """
        # --- Input checks --------------------------------------------------------------------------------
        if i_img is None or warping_field is None or ID is None:
            raise Warning("Please provide i_img, warping_field, and participant ID(s)")

        # Ensure lists
        i_imgs=[i_img] if isinstance(i_img,str) else i_img
        warping_fields=[warping_field] if isinstance(warping_field,str) else warping_field
        IDs=[ID] if isinstance(ID,str) else ID

        # --- Define destination images ---------------------------------------------------------------------
        if dest_img==None:
            dest_img=[]
            for ID_nb in enumerate(i_imgs):
                dest_img.append(os.apth.join(self.code_dir, "template", self.config["PAM50_t2"]))

        else:
            dest_img=[dest_img] if isinstance(dest_img,str) else dest_img

        # --- Define output folders -------------------------------------------------------------------------
        if o_folder==None:
            o_folders=[]
            for i in range(len(warping_fields)):
                o_folders.append(os.path.dirname(warping_fields[i]))
        elif isinstance(o_folder,str):
            o_folders=[o_folder]

        else:
             o_folders=o_folder

        # --- Define output filenames -----------------------------------------------------------------------
        o_imgs=[]
        for ID_nb, filename in enumerate(i_imgs):
            o_imgs.append(os.path.join(o_folders[ID_nb], os.path.basename(i_imgs[ID_nb]).split('.')[0] + tag + ".nii.gz"))

        # --- Apply transformation --------------------------------------------------------------------------
        if not all(os.path.exists(f) for f in o_imgs) or redo:
            print(" ")
            print(">>>>> Apply transformation is running with " + str(n_jobs)+ " parallel jobs on " +str(len(self.participant_IDs)) + " participant(s)")

            Parallel(n_jobs=n_jobs)(delayed(self._run_apply_warp)(i_img=i_imgs[ID_nb],
                                                                        dest_img=dest_img[ID_nb],
                                                                        warp_file=warping_fields[ID_nb],
                                                                        o_folder=o_folders[ID_nb],
                                                                        ID=IDs[ID_nb],
                                                                        tag=tag,
                                                                        threshold=threshold,
                                                                        mean=mean,
                                                                        method=method)
                                        for ID_nb in range(len(warping_fields)))



        else:

            print("Tranformation was already applied put redo=True to redo that step")

        return o_imgs


    def _run_apply_warp(self,i_img,dest_img,warp_file,o_folder,ID,tag,threshold,mean,method):

        o_img = os.path.join(o_folder, os.path.basename(i_img).split('.')[0] + tag + ".nii.gz")

        string = f"sct_apply_transfo -i {i_img} -d {dest_img} -w {warp_file} -x {method} -o {o_img}"
        os.system(string)

        if threshold:
            #Transform the output image in a binary image
            string2 = f"fslmaths {o_img} -thr {threshold} -bin {o_img}"
            os.system(string2)

        if mean==True:
            o_mean_img = os.path.join(o_folder, os.path.basename(i_img).split('.')[0] + tag + "_mean.nii.gz")
            string = f"fslmaths {o_img} -Tmean {o_mean_img}"
            os.system(string)

        print("New warped image was generated for " + ID)

        return o_img

    def _plot_qc(self, ID, ses_name, task_name, tag, qc_indiv_path, fig_size=(5,5),alpha=0.8):
        qc_indiv_dir=utils.get_latest_dir(base_dir=qc_indiv_path)
        img_bck = os.path.join(qc_indiv_dir + "background_img.png")
        img_cntr = os.path.join(qc_indiv_dir, "overlay_img.png")

        # plot the image ctrl as an overlay on the image bck
        img_bck_data = mpimg.imread(img_bck)
        img_cntr_data = mpimg.imread(img_cntr)

        plt.figure(figsize=fig_size)
        plt.imshow(img_bck_data)
        plt.imshow(img_cntr_data, alpha=alpha)  # alpha controls transparency
        plt.title(f'sub-{ID} {ses_name} {task_name} {tag} overlay QC')
        plt.axis('off')
        plt.show()


def copy_segmentation_from_ref_tag(ID, tag, ref_tag, manual_dir, preprocessing_dir):

    fname_dest = os.path.join(preprocessing_dir.format(ID), "func", tag, f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz")

    # We need to copy either the manual segmentation file if it exists for the motor task, or the
    # automatic segmentation file if it doesn't
    fname_ref_manual_seg_list = glob.glob(os.path.join(manual_dir, f"sub-{ID}", "func", f"sub-{ID}_{ref_tag}_*bold_moco_mean_seg.nii.gz"))
    fname_ref_auto_seg_list = glob.glob(os.path.join(preprocessing_dir.format(ID), "func", ref_tag, "sct_deepseg",
                                                     f"sub-{ID}_{ref_tag}_*bold_moco_mean_seg.nii.gz"))
    if len(fname_ref_manual_seg_list) > 0:
        fname_from = sorted(fname_ref_manual_seg_list)[0]  # Take run-01 (sorted list)
        print(f'=== Copying manual segmentation file from {ref_tag} to {tag} for {ID} ===', flush=True)
    elif len(fname_ref_auto_seg_list) > 0:
        fname_from = sorted(fname_ref_auto_seg_list)[0]  # Take run-01 (sorted list)
        print(f'=== Copying automatic segmentation file from {ref_tag} to {tag} for {ID} ===', flush=True)
    else:
        raise RuntimeError(
            f'No segmentation file found for {ref_tag} in either manual or automatic folders for {ID}. Cannot copy segmentation file to {tag}.')

    shutil.copy(fname_from, fname_dest)


def copy_csf_segmentation_from_ref_tag(ID, tag, ref_tag, manual_dir, preprocessing_dir):

    fname_dest = os.path.join(preprocessing_dir.format(ID), "func", tag, f"sub-{ID}_{tag}_bold_moco_mean_CSF_seg.nii.gz")

    # We need to copy either the manual segmentation file if it exists for the motor task, or the
    # automatic segmentation file if it doesn't
    fname_ref_manual_seg_list = glob.glob(os.path.join(manual_dir, f"sub-{ID}", "func", f"sub-{ID}_{ref_tag}_*bold_moco_mean_CSF_seg.nii.gz"))
    fname_ref_auto_seg_list = glob.glob(os.path.join(preprocessing_dir.format(ID), "func", ref_tag, "sct_propseg",
                                                     f"sub-{ID}_{ref_tag}_*bold_moco_mean_CSF_seg.nii.gz"))
    if len(fname_ref_manual_seg_list) > 0:
        fname_from = sorted(fname_ref_manual_seg_list)[0]  # Take run-01 (sorted list)
        print(f'=== Copying manual CSF segmentation file from {ref_tag} to {tag} for {ID} ===', flush=True)
    elif len(fname_ref_auto_seg_list) > 0:
        fname_from = sorted(fname_ref_auto_seg_list)[0]  # Take run-01 (sorted list)
        print(f'=== Copying automatic CSF segmentation file from {ref_tag} to {tag} for {ID} ===', flush=True)
    else:
        raise RuntimeError(
            f'No CSF segmentation file found for {ref_tag} in either manual or automatic folders for {ID}. Cannot copy segmentation file to {tag}.')

    shutil.copy(fname_from, fname_dest)


def copy_warping_fields_from_ref_tag(ID, tag, ref_tag, preprocessing_dir):

    fname_ref_warp_from_func_list = glob.glob(
        os.path.join(preprocessing_dir.format(ID), "func", ref_tag, "sct_register_multimodal",
                     f"sub-{ID}_{ref_tag}_*from-func_to_PAM50_mode-image_xfm.nii.gz"))
    fname_ref_warp_from_pam50_list = glob.glob(
        os.path.join(preprocessing_dir.format(ID), "func", ref_tag, "sct_register_multimodal",
                     f"sub-{ID}_{ref_tag}_*from-PAM50_to_func_mode-image_xfm.nii.gz"))
    if len(fname_ref_warp_from_func_list) != 1 or len(fname_ref_warp_from_pam50_list) != 1:
        raise RuntimeError(f'More than 1 warping fields found for {ref_tag} {ID}. Cannot copy warping fields to {tag}.')

    fname_ref_warp_from_func = sorted(fname_ref_warp_from_func_list)[0]
    fname_ref_warp_from_pam50 = sorted(fname_ref_warp_from_pam50_list)[0]
    fname_ref_warp_from_func_dest = os.path.join(preprocessing_dir.format(ID), "func", tag,
                                                 f"sub-{ID}_{tag}_from-func_to_PAM50_mode-image_xfm.nii.gz")
    fname_ref_warp_from_pam50_dest = os.path.join(preprocessing_dir.format(ID), "func", tag,
                                                  f"sub-{ID}_{tag}_from-PAM50_to_func_mode-image_xfm.nii.gz")

    print(f'=== Copying warping fields from {ref_tag} to {tag} for {ID} ===', flush=True)

    shutil.copy(fname_ref_warp_from_func, fname_ref_warp_from_func_dest)
    shutil.copy(fname_ref_warp_from_pam50, fname_ref_warp_from_pam50_dest)
