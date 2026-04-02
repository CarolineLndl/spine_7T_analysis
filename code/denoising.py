# Main imports ------------------------------------------------------------
import os, json, glob, shutil
from scipy.io import loadmat
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import utils as utils
import warnings

# Plotting
import matplotlib.gridspec as GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

#Stats
import scipy.stats as stats
from nipype.algorithms.confounds import compute_noise_components

#Nilearn
from nilearn import image

class Denoising:
    """
    The Denoising class is used to set up preprocessing analyses.

    Attributes
    ----------
    config : dict
        Configuration dictionary containing paths, participants, sessions, and structures.
    verbose : bool
        Whether to print progress messages.
    """

    def __init__(self,config, IDs=None,verbose=True):

        # Class attributes -------------------------------------------------------------------------------------
        self.config = config # load config info
        self.participant_IDs= IDs # list of the participants to analyze
        self.verbose = verbose
        self.structures = config.get("structures", [""])  # default to empty string if not specified
        self.raw_dir=os.path.join(self.config["raw_dir"]) # directory of the raw data
        self.derivatives_dir=os.path.join(self.config["raw_dir"], self.config["derivatives_dir"]) # directory of the derivatives data
        self.preproc_dir=os.path.join(self.config["raw_dir"], self.config["preprocess_dir"]["main_dir"]) # directory of the preprocess data
        self.manual_dir=os.path.join(self.config["raw_dir"], self.config["manual_dir"]) # directory of the manual corrections
        self.qc_dir=os.path.join(self.config["raw_dir"], self.config["preprocess_dir"]["QC_dir"]) # directory of the QC output
        self.denoising_dir = os.path.join(self.config["raw_dir"], config["denoising"]["dir"])

       # Create directories -------------------------------------------------------------------------------------
        main_denoising_dir = Path(self.denoising_dir.split("sub")[0])
        main_denoising_dir.mkdir(parents=True, exist_ok=True)

        # Create participant directories (if not already existed)
        for ID in self.participant_IDs:
            ID_denoising_dir=os.path.expandvars(self.denoising_dir.format(ID)) # directory of the preprocess data
            os.makedirs(ID_denoising_dir,exist_ok=True)

            # create 1 folder per session if there are multiple sessions (exemple multiple days of acquisition)
            for ses_name in self.config["design_exp"]['ses_names']:
                ses_dir = ses_name if int(self.config["design_exp"]["ses_nb"])>0 else ""
                if ses_dir != "":
                    os.makedirs(os.path.join(ID_denoising_dir, ses_dir),exist_ok=True)

                # spinal cord or brain subfolders will be created if two structures are specified in the config file
                if len(self.config["structures"])>1:
                    for structure in self.config["structures"]:
                        os.mkdir(os.path.join(ID_denoising_dir, ses_dir, structure))
                    ID_dir = os.path.join(ID_denoising_dir, ses_dir, structure)
                else:
                    ID_dir = os.path.join(ID_denoising_dir, ses_dir)

            print("New folders in denoising dir have been created") if verbose==True else None

            # Create a folder for each runs in func folder
            if "design_exp" in self.config.keys():
                for ses_name in self.config["design_exp"]['ses_names']:
                    ses_dir = ses_name if int(self.config["design_exp"]["ses_nb"])>1 else ""
                    # if "acq_names" exist in the config file
                    if "acq_names" in self.config["design_exp"].keys():
                        for task_name in self.config["design_exp"]['task_names']:
                            for acq_name in self.config["design_exp"]['acq_names']:
                                tag = "task-" + task_name + "_acq-" + acq_name
                                os.makedirs(os.path.join(ID_dir, tag), exist_ok=True)

                                # Create denoising sub-directories paths
                                for sub_dir in ["denoised_dir", "norm_dir", "smoothed_dir"]:
                                    if config["denoising"][sub_dir] != "":
                                        new_path = os.path.join(ID_dir, tag, config["denoising"][sub_dir])
                                        os.makedirs(new_path, exist_ok=True)
                                        if sub_dir == "denoised_dir":
                                            os.makedirs(os.path.join(new_path, "confounds"), exist_ok=True)

                    else:
                        for task_name in self.config["design_exp"]['task_names']:
                            task_dir=task_name if int(self.config["design_exp"]["task_nb"])>1 else ""
                            os.makedirs(os.path.join(ID_dir, task_dir), exist_ok=True)
                            # Create denoising sub-directories paths
                            for sub_dir in ["denoised_dir", "norm_dir", "smooth_dir"]:
                                if config["denoising"][sub_dir] != "":
                                    os.makedirs(os.path.join(ID_dir, task_dir, config["denoising"][sub_dir]), exist_ok=True)
                                    if sub_dir == "denoised_dir":
                                        os.makedirs(os.path.join(new_path, "confounds"), exist_ok=True)

    def moco_params(self,ID=None, slice_wise=True,input_file=None, func_file=None, structure="",task_name='',run_name='', output_file=None,redo=False,verbose=True):
        """
            Create slicewise moco parameters

            Attributes
            ----------
            ID: str
                participant ID (required , default=None)
            slice_wise: bool
                Whether the moco parameters are going to be extracted slice_wise (default) of volume_wise
                True the moco parameters are extracted within each slice of the image (default)
                False the moco parameters are extracted within the entire volume
            input_file: list
                list of moco param file (you can provide 2 for the same participant)
            func_file: filename
                The brain functional image will be useful to extract the number of slices, it is not necessary for spinal cord as the information is already included in moco_param image.
            structure : str
                could be 'brain' or 'spinalcord' name of the structure to work on (requiered, default='')
            task_name : str
                task name (optional , default="")
            run_name : str
                run name (optional , default="")
            output_file : list
                list of output files (one for each participants)
            redo : bool
                whether to redo the calculation if the output file already exists (optional , default=False)
            verbose : bool
                whether to print progress messages (optional , default=True)

            outputs
            ----------
        """

        if ID==None:
            raise(Exception('ID should be provided ex: ID="A001"'))
        physio_dir = os.path.join(self.denoising_dir.format(ID), task_name, structure, self.config["denoising"]["denoised_dir"], "confounds")  # output directory

        structure_tag = "" if structure == "" else "_" + structure
        task_tag = "" if task_name == "" else "_" + task_name
        run_tag = "" if run_name == "" else "_" + run_name

        # Select the input file  (text file if brain or nifti file if spinal cord)
        if input_file is None:
            if structure == "brain":
                input_file = glob.glob(os.path.join(self.preproc_dir.format(ID), self.config["moco_files"]["dir"].format(ID,run_tag,structure_tag), self.config["moco_files"]["moco_param"][structure]))[0]
            else:
                input_file = sorted(glob.glob(os.path.join(self.preproc_dir.format(ID), self.config["preprocess_dir"]["func_moco"].format(task_name), structure_tag, self.config["preprocess_f"]["moco_params"].format(task_tag,run_tag))))
        if verbose:
            print("----------------------------------------------------------")
            print("Moco parameters estimation: " + ID + " : " + task_name + " " + run_name)
            print("----------------------------------------------------------")

        if slice_wise:
            if structure == "brain":
                if func_file is None:
                    func_file=glob.glob(os.path.join(self.preproc_dir.format(ID), self.config["moco_files"]["dir"].format(ID,run_tag,structure_tag), self.config["moco_files"]["moco_mean_f"]))[0]

                func_img = nib.load(func_file)  # load the func image
                moco_brain = pd.read_csv(input_file, delim_whitespace=True, header=None)  # load motion parameter file
                for slice_nb in range(0,func_img.header.get_data_shape()[2]):
                    slice_str = "00" + str(slice_nb + 1) if (slice_nb+1)<10 else "0" + str(slice_nb+1)
                    output_moco_file = os.path.join(physio_dir, f"sub-{ID}_6_moco{structure_tag}{task_tag}{run_tag}_slice{slice_str}.txt")
                    if not os.path.exists(output_moco_file):
                        np.savetxt(output_moco_file, moco_brain)

                if os.path.exists(output_moco_file) and not redo and verbose:
                    print("Brain moco params were already extracted please, put redo=True to recalculate it")

                # create a dataframe with volume value for each slice as we do not have the slice wise motion corrected parameters
            if structure=="spinalcord" or structure=="":
                
                #checkek wether there is params_x in the input_file list
                X_file=[f for f in input_file if 'params_x' in f]
                Y_file=[f for f in input_file if 'params_y' in f]
                X_img=nib.load(X_file[0]) # load the X moco parameter image
                Y_img=nib.load(Y_file[0]) # load the X moco parameter image

                #extract the mocovalue for each slice
                for slice_nb in range(0,X_img.header.get_data_shape()[2]):
                    slice_str="00" + str(slice_nb + 1) if (slice_nb+1)<10 else "0" + str(slice_nb+1)
                    output_moco_file = os.path.join(physio_dir, f"sub-{ID}_2_moco{structure_tag}{task_tag}{run_tag}_slice{slice_str}.txt")
                    if not os.path.exists(output_moco_file) or redo :
                        moco_value=[]

                        for img in [X_img,Y_img]:
                            img_slice=img.slicer[:,:,slice_nb:slice_nb+1,:] # cropped func slices
                            imgseries=img_slice.get_fdata(dtype=np.float32)
                            imgseries_reshape=imgseries.reshape(img.shape[3], 1)
                            moco_value.append(imgseries_reshape)
                        moco_value=np.hstack(moco_value)

                        np.savetxt(output_moco_file, moco_value)

                if os.path.exists(output_moco_file) and not redo and verbose:
                    print("Spinal cord moco params were already extracted please, put redo=True to recalculate it")

        else:
            # moco param are going to by copy

            input_file=glob.glob(os.path.join(self.preproc_dir.format(ID), self.config["moco_files"]["dir"].format(ID,structure), self.config["moco_files"]["moco_param"][structure]))[0]
            output_moco_file = os.path.join(
                physio_dir,
                f"sub-{ID}_{'6' if structure == 'brain' else '2'}_moco_{structure}.txt")

            if not os.path.exists(output_moco_file):
                delimiter=" " if structure == 'brain' else ","
                moco_brain=pd.read_csv(input_file, delimiter=delimiter, header=None) # load motion parameter file
                np.savetxt(output_moco_file, moco_brain)


    def outliers(self,ID=None, i_img=None, structure='',mask_file=None,ses_name='',task_name='', run_name='', output_file=None,redo=False, verbose=True):
        """
            Outliers calculation with fsl
            https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLMotionOutliers

            Attributes
            ----------
            ID: str
                participant ID (required , default=None)
            i_img : str
                filename of the 4D func file, (required , default=None, moco file will be targeted)
            structure : str
                could be 'brain' or 'spinalcord' name of the structure to work on (requiered, default='')
            mask_file : str
                filename of a binary mask file (required , default=None, seg file will be targeted)
            ses_name : str
                session name (optional , default='')
            task_name : str
                task name (optional , default='')
            run_name : str
                run name (optional , default='')
            output_file : str
                output files  (optional , default=None)
            redo : bool
                whether to redo the calculation if the output file already exists (optional , default=False)

            Outputs:
            --------
        """

        # --- Input validation -------------------------------------------------------------
        if ID is None:
            raise(Exception('ID shoul be provided ex: ID="A001"'))
        if structure==None:
            raise(Exception('structure should be provided ex: ID="spinalcord"'))

        # --- Define directories -----------------------------------------------------------
        task_tag="" if task_name=="" else "_" + task_name
        run_tag="" if run_name=="" else "_" + run_name

        if i_img is None:
            i_img = glob.glob(os.path.join(self.preproc_dir.format(ID), self.config["preprocess_dir"]["func_moco"].format(task_name), self.config["preprocess_f"]["func_moco"].format(ID,task_tag,run_tag)))[0]

        if mask_file is None:
            mask_file = os.path.join(self.preproc_dir.format(ID), 'func', task_name, f"sub-{ID}_{task_name}_bold_moco_mean_seg.nii.gz")

            if not os.path.exists(mask_file):
                raise RuntimeError('No mask file found for this participant, task and run, please provide a mask file or check the preprocessing outputs')

        # --- Define output file --------------------------------------------------------
        if output_file==None:
            output_file = os.path.join(self.denoising_dir.format(ID), task_name, self.config["denoising"]["denoised_dir"].format(ID), structure, "confounds", f"sub-{ID}{task_tag}{run_tag}_outliers")
        
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        # --- Run outliers calculation --------------------------------------------------------
        cmd_fsl = f"fsl_motion_outliers -i {i_img} -o {output_file}.txt —m {mask_file} --nomoco --dvars -p {output_file}.png"

        if not os.path.exists(output_file + ".txt") or redo:
            if verbose:
                print("----------------------------------------------------------")
                print("Compute outliers estimation: " + ID + " : " + task_name + " " + run_name)
                print("----------------------------------------------------------")

            os.system(cmd_fsl)

            # fsl do not provide outputs if there are no outliers so we need to create a file with only 0 values
            if not os.path.exists(output_file + ".txt"):
                func_img=nib.load(i_img)
                vol_number=func_img.header.get_data_shape()[3]
                array = np.zeros((vol_number, 1))
                np.savetxt(output_file + ".txt", array, fmt='%d', delimiter='   ')

        return output_file + ".txt"

    def find_physio_file(self,input_dir=None, ID=None, ses_name='', task_name='', run_name='',copy=True,output_dir=None,redo=False, verbose=True):
        """
            Find physio file in the BIDS structure, could be .tsv, .log or .mat format
            Attributes
            ----------
            input_dir : str
                input directory where to find the physio file (default=None, the function will look into the raw BIDS folder)
            ID: str
                participant ID (required , default=None)
            ses_name : str
                session name (optional , default='')
            task_name : str
                task name (optional , default='')
            run_name : str
                run name (optional , default='')
            copy : bool
                whether to copy the physio file in the denoising folder (default=True)
            output_dir : str
                output directory where to copy the physio file (required if copy=True, default=None)
            redo : bool
                whether to redo the copy if the output file already exists (optional , default=False)
            verbose : bool
                whether to print progress messages (optional , default=True)

            Outputs:
            --------
        """
        # --- Input validation -------------------------------------------------------------
        if ID is None:
            raise(Exception('ID should be provided ex: ID="A001"'))
        if input_dir is None:
            input_dir = os.path.join(self.raw_dir, f"sub-{ID}", "func")

        if verbose:
            print("Looking for physio file in : " + input_dir)

        # --- Define variables -----------------------------------------------------------
        task_tag="" if task_name=="" else "_" + task_name
        run_tag="" if run_name=="" else "_" + run_name
        outputs=[];outputs_resp=[];outputs_puls=[];outputs_trig=[];outputs_tics=[]
        physio_file_tsv = glob.glob(os.path.join(input_dir, f"*{task_tag}*{run_tag}*.tsv*"))
        physio_file_log = glob.glob(os.path.join(input_dir, f"*{task_tag}*{run_tag}*.log"))
        physio_file_mat = glob.glob(os.path.join(input_dir, "**", f"*{task_tag}*{task_tag}RS.mat"))

        # --- Find and copy physio files --------------------------------------------------------
        # Case 1: only one .tsv file exists
        if len(physio_file_tsv) ==1:
            if copy: # copy the file in an other folder if copy==True
                if output_dir is None:
                    raise(Exception('output_dir should be list of directories'))
                if not os.path.exists(os.apth.join(output_dir, os.path.basename(physio_file_tsv[0]))) or redo:
                    output = shutil.copyfile(physio_file_tsv[0], os.path.join(output_dir, os.path.basename(physio_file_tsv[0]))) # copy the file in an other folder
                    print('Physio file has been copy here : ' + output)
                else:
                    output = os.path.join(output_dir, os.path.basename(physio_file_tsv[0]))

                if output.split('.')[-1] == "gz": # unzip the file if it was in .gz format
                    output = utils.unzip_file(i_file=output, ext='.tsv', zip_file=False, redo=redo)

            else:
                output=physio_file_tsv
                if output.split('.')[-1] == "gz": # unzip the file if it was in .gz format
                    output=utils.unzip_file(i_file=output, ext='.tsv', zip_file=False, redo=redo)
                outputs.append(output)

        # Case 2: two .tsv files exists (respiratory and cardiac)
        elif len(physio_file_tsv) >1:
            if copy==True: # copy the file in an other folder if copy==True
                if output_dir is None:
                    raise(Exception('output_dir should be list of directories'))
                print(os.path.join(input_dir, f"*{run_tag}*respiratory*.tsv*"))
                input_resp = glob.glob(os.path.join(input_dir, f"*{run_tag}*respiratory*.tsv*"))[0]
                input_puls = glob.glob(os.path.join(input_dir, f"*{run_tag}*cardiac*.tsv*"))[0]
                input_resp_json = glob.glob(os.path.join(input_dir, f"*{run_tag}*respiratory*.json"))[0]
                input_puls_json = glob.glob(os.path.join(input_dir, f"*{run_tag}*cardiac*.json"))[0]

                if not os.path.exists(os.path.join(output_dir, os.path.basename(input_resp))) or redo:
                    output_resp = shutil.copyfile(input_resp, os.path.join(output_dir, os.path.basename(input_resp)))
                    shutil.copyfile(input_resp_json, os.path.join(output_dir, os.path.basename(input_resp_json)))
                    output_puls = shutil.copyfile(input_puls, os.path.join(output_dir, os.path.basename(input_puls)))
                    shutil.copyfile(input_puls_json, os.path.join(output_dir, os.path.basename(input_puls_json)))
                    print('Physio file has been copy here : ' + output_resp)

                else:
                    output_resp = os.path.join(output_dir, os.path.basename(input_resp))
                    output_puls = os.path.join(output_dir, os.path.basename(input_puls))

                if output_resp.split('.')[-1] == "gz": # unzip the file if it was in .gz format
                    output_resp = utils.unzip_file(i_file=output_resp,ext='.tsv',zip_file=False,redo=redo)
                    output_puls = utils.unzip_file(i_file=output_puls,ext='.tsv',zip_file=False,redo=redo)

        # Case 3: multiple .log files exists
        elif len(physio_file_log) >0:
            # multiple log files exists (e.g *_RESP.log; *_PULS.log; *_Trigger.log)
            input_resp = glob.glob(os.path.join(input_dir, f"*{run_tag}*RESP.log"))[0]
            input_puls = glob.glob(os.path.join(input_dir, f"*{run_tag}*PULS.log"))[0]
            input_tics = glob.glob(os.path.join(input_dir, f"*{run_tag}*AcquisitionInfo.log"))[0]

            if copy: # copy the file in another folder if copy==True
                if output_dir is None:
                        raise(Exception('output_dir should be list of directories'))

                if not os.path.exists(os.path.join(output_dir, os.path.basename(input_resp))) or redo:
                    output_resp = shutil.copyfile(input_resp, os.path.join(output_dir, os.path.basename(input_resp)))  # copy the file in an other folder
                    output_puls = shutil.copyfile(input_puls, os.path.join(output_dir, os.path.basename(input_puls)))
                    # output_trig = shutil.copyfile(input_trig, os.path.join(output_dir, os.path.basename(input_trig)))
                    output_tics = shutil.copyfile(input_tics, os.path.join(output_dir, os.path.basename(input_tics)))
                else:
                    # copy the file in another folder
                    output_resp = os.path.join(output_dir, os.path.basename(input_resp))
                    output_puls = os.path.join(output_dir, os.path.basename(input_puls))
                    # output_trig = os.path.join(output_dir, os.path.basename(input_trig))
                    output_tics = os.path.join(output_dir, os.path.basename(input_tics))

            else:
                output_resp = input_resp;output_puls = input_puls;output_tics = input_tics
                #output_trig=input_trig;
            output_resp.append(output_resp); output_puls.append(output_puls); output_tics.append(output_tics) #outputs_trig.append(output_trig) ;

        # Case 4: .mat file exists
        elif len(physio_file_mat) >0:
            if copy:# copy the file in an other folder if copy==True
                if output_dir is None:
                    raise(Exception('output_dir should be list of directories'))
                if not os.path.exists(physio_file_mat[0]) or redo:
                    output = shutil.copyfile(physio_file_mat[0], os.path.join(output_dir, os.path.basename(physio_file_mat[0]))) # copy the file in an other folder
                    print('Physio file has been copy here : ' + output)
                else:
                    output = os.path.join(output_dir, os.path.basename(physio_file_mat[0]))

        #  Case 5: no physio file found
        else:
            raise(Exception('Physio files format should be in .log or .tsv or .tsv.gz or .mat'))

        return output if len(physio_file_tsv) == 1 or len(physio_file_mat) > 0 else (output_resp, output_puls) if len(physio_file_tsv) == 2 else (output_resp, output_puls, output_tics) if len(physio_file_log) > 0 else None

    def plot_physio(self,ID=None,TR=None,frq=None,denoising_mat=None,task_name="",run_name="",output_dir=None,redo=False,verbose=False):
        """
            Plot physiological recordings

            Attributes
            ----------
            ID: str
                participant ID (required , default=None)
            TR: float
                repetition time of the fMRI acquisition (if None, TR will be extracted from the config file)
            frq: float
                frequency of the physio recording (in Hz), if None, frq will be extracted from the config file
            denoising_mat : list
                list of 4D input files (one for each participants)
            task_name : str
                task name (optional , default="")
            run_name : str
                run name (optional , default="")
            output_dir : list
                list of output files (one for each participants)
            redo : bool
                whether to redo the copy if the output file already exists (optional , default=False)
            verbose : bool
                whether to print progress messages and plots (optional , default=False)

            outputs
           ----------
                output_file : str
                    filename of the physio plot
        """
        task_tag = "" if task_name=="" else "_" + task_name
        run_tag = "" if run_name=="" else "_" + run_name

        if ID is None:
            raise(Exception('ID should be provided ex: ID="A001"'))

        if TR is None:
            TR = self.config["acq_params"]["TR"]
        if frq is None:
            frq = self.config["acq_params"]["physio_frq"]

        if denoising_mat is None:
            raise Exception('denoising_mat should be provided ex: denoising_mat="path/to/denoising/mat/file.mat"')

        if output_dir is None:
            output_dir = os.path.join(self.base_dir, self.config["denoising"]["denoised_dir"].format(ID,task_name), 'physio_plots')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"sub-{ID}{task_tag}{run_tag}_physio.png")

        # 1. Load .mat file______________________________________________________
        mat_file={}

        load_mat= loadmat(denoising_mat)
        mat_file_load=load_mat['physio']['ons_secs'][0][0][0][0]
        mat_file={'t':mat_file_load[0],'t_start':mat_file_load[1],
                            'c':mat_file_load[2],'r':mat_file_load[3], # raw recordings across time c:cardiac and r: respi
                            'c_scaling':mat_file_load[4], 'r_scaling':mat_file_load[5], # 3 data / secondes
                            'c_is_reliable':mat_file_load[6],'r_is_reliable':mat_file_load[7],
                            'c_pulse':mat_file_load[8], # peaks / secondes
                            'fr':mat_file_load[9],
                            'c_sample_phase':mat_file_load[10],
                            'hr':mat_file_load[12], # heart rate / secondes
                            'rvt':mat_file_load[13]} # respiratory volume per time (per secondes)

        # 2. Convert .mat in .txt files_______________________________________
        np.savetxt(os.path.join(output_dir, f"sub-{ID}{task_tag}{run_tag}_raw_resp.txt"), mat_file['r'])
        np.savetxt(os.path.join(output_dir, f"sub-{ID}{task_tag}{run_tag}_raw_card.txt"), mat_file['c'])
        np.savetxt(os.path.join(output_dir, f"sub-{ID}{task_tag}{run_tag}_hr.txt"), mat_file['hr'])
        np.savetxt(os.path.join(output_dir, f"sub-{ID}{task_tag}{run_tag}_rvt.txt"), mat_file['rvt'])

        # 3 Plots physio and save figures_________________________________________

        fig=plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
        gs=GridSpec.GridSpec(2,2,figure=fig) # 2 rows, 3 columns
        ax1=fig.add_subplot(gs[0,0]) # First row, first column
        ax2=fig.add_subplot(gs[0,1]) # First row, second column
        ax3=fig.add_subplot(gs[1,:]) # Second row, span all columns
        fig.tight_layout()


        fig.subplots_adjust(hspace = .5, wspace=0.1)
        ax3.stem((mat_file['c_pulse'][0:]),np.ones(len(mat_file['c_pulse'][0:])),'lightpink',markerfmt=' ', basefmt=' ' )

        ax3.plot((mat_file['t'][1:]),mat_file['c'][1:],color='crimson')
        ax1.plot(np.arange(start=0, stop=len(mat_file['c_sample_phase'])),(mat_file['c_sample_phase']+70) ,color='crimson')

        ax1.plot(mat_file['hr'],color='black')
        ax2.plot(np.arange(start=0, stop=len(mat_file['t']))/frq/TR,mat_file['r'])
        ax2.plot(mat_file['rvt'],color='black')
        ax1.set_title('sub-' + ID ,fontsize=20)

        ax3.set_ylabel("Card / Peak Cardio ",fontsize=12)
        ax1.set_ylabel("resample card / Hear rate ",fontsize=12)
        ax1.set_xlabel("Volumes",fontsize=12)
        ax2.set_ylabel("Respiration and  RVT",fontsize=12) # rvt= respiratory volume per time
        ax2.set_xlabel("Volumes",fontsize=12)
        ax3.set_xlabel("Time (sec)",fontsize=12)

        if verbose==True:
            plt.show()
        if not os.path.exists(output_file) or redo:
            plt.savefig(output_file, dpi=300, bbox_inches = 'tight')
            plt.close()

            if verbose==True:
                print("physio plot were saved here : " + output_file)

        return output_file

    def confounds_ts(self,ID=None,func_file=None,slice_wise=True,mask_seg_file=None,mask_csf_file=None,compcor=False, DCT=False, output_file=None, task_name="",run_name="", structure="", n_compcor=5, n_DCT=3, redo=False, verbose=True):
        """
            Compute slice-wise or volume-wise physiological confound time series (aCompCor and/or DCT) for fMRI denoising.

            This function generates nuisance regressors from functional data using:
            - aCompCor: PCA components extracted from a CSF mask
            - DCT: high-pass cosine basis functions

            Functional images and masks can be provided directly, or automatically
            retrieved from the preprocessing directory based on the participant ID
            and the targeted structure (“brain” or “spinalcord”).

            Reference:
                https://nipype.readthedocs.io/en/1.1.0/interfaces/generated/nipype.algorithms.confounds.html


            Parameters
            ----------
            ID : str
                Participant ID (required).
            func_file : str
                Functional 4D image. If None, the file is automatically retrieved
                from the preprocessing directory.
            slice_wise : bool, optional
                If True (default), extract confounds independently for each slice.
                If False, extract confounds across the whole volume.
            mask_seg_file : str, optional
                Binary segmentation mask (GM or WM+GM) used for DCT computation.
                If None, it is automatically retrieved.
            mask_csf_file : str, optional
                Binary CSF mask used for aCompCor computation.
                If None, it is automatically retrieved.
            compcor : bool, optional
                If True, compute aCompCor components (default=False).
            DCT : bool, optional
                If True, compute DCT high-pass components (default=False).
            output_file : str or list, optional
                Optional list of output paths (not generally required).
            task_name : str, optional
                Name of the task (affects output filename).
            run_name : str, optional
                Run label (affects output filename).
            structure : str, optional
                Target structure: "brain" or "spinalcord". Required for locating the correct masks.
            n_compcor : int, optional
                Number of aCompCor components to extract (default=5).
            n_DCT : int, optional
                Number of DCT components to extract (default=3).
            redo : bool, optional
                If True, recompute confounds even if output files already exist.
            verbose : bool, optional
                If True, print progress information.

            Returns
            -------
            output_compcor_file : str
                Path to the output aCompCor components file.
            output_DCT_file : str
                Path to the output DCT components file.

            Notes
            -----
            - When `slice_wise=True`, one file per slice is generated (slice001, slice002, ...).
            - Missing masks in extremal slices produce NaN (DCT) or zero-filled (aCompCor) outputs.
            - The function automatically ensures consistent slice counts across input images.
        """

        # --- Input validation -------------------------------------------------------------
        physio_dir = os.path.join(self.denoising_dir.format(ID), task_name, self.config["denoising"]["denoised_dir"].format(ID), structure, 'confounds')  # output directory

        if ID is None:
            raise Exception('ID shoul be provided ex: ID="A001"')
        if func_file is None:
            raise Exception('func_file should be provided ex: func_file="path/to/func/file.nii.gz"')
        if mask_seg_file is None and DCT:
             raise Exception('mask_seg_file should be provided ex: mask_seg_file="path/to/seg/file.nii.gz"')
        if mask_csf_file is None and compcor:
             raise Exception('mask_csf_file should be provided ex: mask_csf_file="path/to/csf/file.nii.gz"')

        # ---  Load files -----------------------------------------------------------
        print("comcord")
        print(func_file)
        print(mask_seg_file)
        func_img = nib.load(func_file) # load the functional image
        mask_seg_img = nib.load(mask_seg_file) # load the seg mask image
        mask_csf_img = nib.load(mask_csf_file) # load the csf mask image
        TR = func_img.header.get_zooms()[3]

        # Determine number of valid slices
        n_slices = min(func_img.header.get_data_shape()[2],
               mask_seg_img.header.get_data_shape()[2],
               mask_csf_img.header.get_data_shape()[2])

        # --- Define variables -----------------------------------------------------------

        # Define output filenames
        structure_tag = "" if structure =="" else "_" + structure
        task_tag = "" if task_name=="" else "_" + task_name
        run_tag = "" if run_name=="" else "_" + run_name

        # Define output filenames
        if slice_wise:
            output_DCT_file = os.path.join(physio_dir, f"sub-{ID}_{n_DCT})_DCT{structure_tag}{task_tag}{run_tag}_slice001.txt")
            output_compcor_file = os.path.join(physio_dir, f"sub-{ID}_{n_compcor}_acompcor{structure_tag}{task_tag}{run_tag}_slice001.txt")
        else:
            output_DCT_file = os.path.join(physio_dir, f"sub-{ID}_{n_DCT})_DCT{structure_tag}{task_tag}{run_tag}.txt")
            output_compcor_file = os.path.join(physio_dir, f"sub-{ID}_{n_compcor}_acompcor{structure_tag}{task_tag}{run_tag}.txt")

        # --- Run confound extraction --------------------------------------------------------
        if DCT and (not os.path.exists(output_DCT_file) or redo):
            if verbose:
                print("----------------------------------------------------------")
                print("Compute DCT: " + ID + " : " + task_name + " " + run_name)
                print("----------------------------------------------------------")

        if compcor and (not os.path.exists(output_compcor_file) or redo):
            if verbose:
                print("----------------------------------------------------------")
                print("Compute Compcor: " + ID + " : " + task_name + " " + run_name)
                print("----------------------------------------------------------")

        if redo or (
            (DCT and not os.path.exists(output_DCT_file)) or
            (compcor and not os.path.exists(output_compcor_file))
            ):

            if slice_wise:
                # extract the metric slice by slice
                for slice_nb in range(0,n_slices):
                    func_slice=func_img.slicer[:,:,slice_nb:slice_nb+1,:] # cropped func slices
                    mask_seg_slice=mask_seg_img.slicer[:,:,slice_nb:slice_nb+1] # cropped mask slices
                    mask_csf_slice=mask_csf_img.slicer[:,:,slice_nb:slice_nb+1] # cropped mask slices
                    slice_str="00" + str(slice_nb + 1) if (slice_nb+1)<10 else "0" + str(slice_nb+1)

                    # Run DCT
                    if DCT:
                        output_DCT_file = os.path.join(physio_dir, f"sub-{ID}_{n_DCT})_DCT{structure_tag}{task_tag}{run_tag}_slice{slice_str}.txt")

                        if not os.path.exists(output_DCT_file) or redo:
                            DCT_comp=compute_noise_components(imgseries=func_slice.get_fdata(dtype=np.float32),
                                                            mask_images=[mask_seg_slice],
                                                            filter_type='cosine', # 'cosine': Discrete cosine (DCT) basis
                                                            period_cut=128,  # 'period_cut': minimum period (in sec) for DCT high-pass filter
                                                            repetition_time=TR,
                                                            components_criterion=n_DCT)

                            if DCT_comp[0].size==0:
                                DCT_comp_final=np.full((func_img.header.get_data_shape()[3],n_DCT), np.nan)

                            else:
                                DCT_comp_final=DCT_comp[1]

                            np.savetxt(output_DCT_file, DCT_comp_final)

                    # Run compcor
                    if compcor:
                        output_compcor_file = os.path.join(physio_dir, f"sub-{ID}_{n_compcor}_acompcor{structure_tag}{task_tag}{run_tag}_slice{slice_str}.txt")
                        if not os.path.exists(output_compcor_file) or redo:
                            compcor_comp=compute_noise_components(imgseries=func_slice.get_fdata(dtype=np.float32),
                                                            mask_images=[mask_csf_slice], filter_type='polynomial', degree=2,
                                                            repetition_time=TR,
                                                            components_criterion=n_compcor)

                            # create a matri with 0 value when there is no mask, it can happen of extrem slices
                            # A zero predictor contains no variability or information, so it cannot influence the signal during regression.
                            if compcor_comp[0].size==0:
                                compcor_comp_final=np.full((func_img.header.get_data_shape()[3],n_compcor), 0)

                            else:
                                compcor_comp_final=compcor_comp[0]
                            np.savetxt(output_compcor_file, compcor_comp_final)

            else:
                #extract the metric across the volume
                #Run DCT
                if DCT:
                    if not os.path.exists(output_DCT_file) or redo:
                        DCT_comp=compute_noise_components(imgseries=func_img.get_fdata(dtype=np.float32),
                                                            mask_images=[mask_seg_img],
                                                            filter_type='cosine',# 'cosine': Discrete cosine (DCT) basis
                                                            period_cut=128, # 'period_cut': minimum period (in sec) for DCT high-pass filter
                                                            repetition_time=TR,
                                                            components_criterion=n_DCT)
                        np.savetxt(output_DCT_file,DCT_comp[1])

                # Run compcor
                if compcor:
                    if not os.path.exists(output_compcor_file) or redo:
                        compcor_comp=compute_noise_components(imgseries=func_img.get_fdata(dtype=np.float32),
                                                            mask_images=[mask_csf_img],
                                                            filter_type='polynomial', #'polynomial' Legendre polynomial basis
                                                            degree=2,
                                                            repetition_time=TR,
                                                            components_criterion=n_compcor)

                        np.savetxt(output_compcor_file, compcor_comp[0])


        return output_compcor_file, output_DCT_file

    def combine_confounds(self,ID=None,confounds_infos=None,func_file=None,structure="",retroicor_confounds=False,compcor_confounds=False,moco_confounds=False,outliers_confounds=False,DCT_confounds=False,slice_wise=True,task_name="",run_name="",redo=False, verbose=True):
        """
            Combine confounds into a single file.

            Attributes
            ----------
            ID : str (required)
                Participant ID (e.g., "A001").

            confounds_infos : dict (required)
                Dictionary specifying how many components to extract from
                each confound source, e.g.:
                    {'Outliers':1, 'Motion':6, 'Retroicor':18, 'CompCor':12, 'DCT':3}
                Remove keys entirely if they are not used (do NOT set 0).

            func_file : str (optional)
                4D preprocessed functional file. If not provided, it is automatically retrieved from preprocessing folders.

            structure : str
                Either "brain" or "spinalcord". Default: "".

            retroicor_confounds, compcor_confounds, moco_confounds,
            outliers_confounds, DCT_confounds : str or False
                Paths to optional confound files.

            slice_wise : bool
                If True, one combined confound file is created per slice.

            task_name, run_name : str
                Tags appended to outputs.

            redo : bool
                If True, recompute even if output already exists.

            verbose : bool
                If True, print progress message/plotting.


            Returns
            -------
            output_file : str
                Combined confounds file.

            output_zfile : str
                Z-scored version of the combined confounds file.

            Notes
            -----
            - Automatically handles missing confound files (fills NaNs).
            - Handles slice-wise or volume-wise processing.
            - Motion files may need harmonization (FSL vs SCT delimiters).
        """
        # --- Input validation -------------------------------------------------------------
        if ID is None:
            raise Exception('ID shoul be provided ex: ID="A001"')

        if confounds_infos is None:
            raise Exception("Provide confound info: ex: {'Outliers':0,'Motion':6,'Retroicor':18,'CompCor':12,'DCT':3}")

        # --- Prepare tags and directories ---------------------------------------------------
        structure_tag = "" if structure =="" else "_" + structure
        task_tag = "" if task_name=="" else "_" + task_name
        run_tag = "" if run_name=="" else "_" + run_name

        physio_dir = os.path.join(self.denoising_dir.format(ID), task_name, self.config["denoising"]["denoised_dir"].format(ID), structure, 'confounds')  # output directory

        print("combine confounds")
        print(func_file)
        
        if func_file is None:
            func_file = glob.glob(os.path.join(self.preproc_dir.format(ID), self.config["preprocess_dir"]["func_moco"].format(task_name), self.config["preprocess_f"]["func_moco"].format(ID,task_tag,run_tag)))[0]

        if outliers_confounds:
            outliers_file = glob.glob(os.path.join(physio_dir, f"*{task_tag}{run_tag}*outliers.txt"))[0]

        #    outliers_read=pd.read_csv(outliers_confounds,sep='  ',index_col=False,header=None,engine='python')
        #    outliers_id=pd.DataFrame.to_numpy(outliers_read)

        # Calculate number of slices
        func_img = nib.load(func_file) # load the func image
        n_vol = func_img.shape[3]
        slice_number=func_img.header.get_data_shape()[2] if slice_wise==True else 1
        slice_range = range(slice_number) if slice_wise else [None]

        # --- Combine confounds --------------------------------------------------------
        # Slice handling
        slice_number = func_img.shape[2] if slice_wise else 1
        slice_range = range(slice_number) if slice_wise else [None]

        # Read outliers file (only once)
        if outliers_confounds:
            outliers_read = pd.read_csv(outliers_file, sep=r"\s+", header=None)
            outliers_id = outliers_read.to_numpy()
        else:
            outliers_id = None

        if verbose:
                    print("----------------------------------------------------------")
                    print("Combine noise confounds for participant: " + ID + " : " + task_name + " " + run_name)
                    print("----------------------------------------------------------")

        # Loop over slices/volume
        for slice_nb in slice_range:
            slice_str = f"{slice_nb+1:03d}" if slice_wise else ""
            output_tag = f"_slice{slice_str}" if slice_wise else ""
            output_file = os.path.join(physio_dir, f"sub-{ID}_allconfounds{structure_tag}{task_tag}{run_tag}{output_tag}.txt")
            print(output_file)
            if os.path.exists(output_file) and not redo:
                continue

            # Storage
            Confounds = {'All': np.empty((0, n_vol))}
            if not os.path.exists(output_file) or redo:

                for confound_name, n_comp in confounds_infos.items():
                    if n_comp <= 0:
                        raise Exception(f"Confound '{confound_name}' must have n_comp > 0")

                    # OUTLIERS
                    if confound_name == "outliers":
                        if outliers_id is not None:
                            data = outliers_id
                        else:
                            data = np.full((n_vol, n_comp), np.nan)
                    else:
                        confound_path = os.path.join(physio_dir, f"*{confound_name}*{task_tag}{run_tag}*{slice_str}.txt")

                        # Other confound files
                        pattern = glob.glob(confound_path)
                        if pattern:
                            df = pd.read_csv(pattern[0], sep=r"\s+", header=None)
                            data = df.to_numpy()
                            data = data[:, :n_comp]  # truncate if extra cols
                        else:
                            # missing file → fill with NaN
                            data = np.full((n_vol, n_comp), np.nan)

                    Confounds[confound_name] = data
                    Confounds['All'] = np.concatenate((Confounds['All'], data.T))

                # Save combined confounds
                df_all = pd.DataFrame(Confounds['All'].T)
                df_all.to_csv(output_file, index=False, header=False, sep=' ')

                # Save z-scored version
                z = stats.zscore(Confounds['All'].T, nan_policy='omit')
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                df_z = pd.DataFrame(z)
                output_zfile = output_file.replace(".txt", "_z.txt")
                df_z.to_csv(output_zfile, index=False, header=False, sep=' ')
       
        return output_file

    def plot_confound_design(self,ID=None,confound_file=None,confounds_infos=None,structure="",task_name="",run_name='',redo=False, verbose=True):
        """
            Plot confound design matrix

            Attributes
            ----------
            ID: str
                participant ID (required , default=None)
            confound_file : str
                4D preprocessed functional file.
            confounds_infos : dict
                dictonary with the name an the number of each confounds: (e.g {'Outliers':0,'Motion':6,'Retroicor':18,'CompCor':5})

            structure : str
                could be 'brain' or 'spinalcord' name of the structure to work on
            task_name : str
                task name (optional , default="")
            run_name : str
                run name (optional , default="")
            redo : bool
                whether to redo the calculation if the output file already exists (optional , default=False)
            verbose : bool
                whether to print progress messages and plots (optional , default=True)
        """
        if ID==None:
            raise(Exception('ID should be provided ex: ID="A001"'))

        if structure==None:
            raise(Exception('Structure should be provided ex: structure="spinalcord"'))

        structure_tag = "" if structure =="" else  " " + structure
        task_tag = "" if task_name=="" else " " + task_name
        run_tag = "" if run_name=="" else " " + run_name

        Confounds=pd.read_csv(confound_file,delimiter=' ',index_col=False,header=None)
        total_confounds=0

        for confound_name in confounds_infos:
            total_confounds = total_confounds + confounds_infos[confound_name]

        for confound_name in confounds_infos:
            if confound_name == "outliers":
                confounds_infos["outliers"] = Confounds.shape[1]+1-total_confounds
            elif confound_name =="outliers_brsc":
                confounds_infos["outliers_brsc"] = Confounds.shape[1]+1-total_confounds

        labels=['']
        for confound_name in confounds_infos:
            labels=np.concatenate((labels,np.repeat(confound_name,confounds_infos[confound_name])))


        fig, ax = plt.subplots(figsize=(10, 8))
        ax=sns.heatmap(Confounds[:],vmin=-1, vmax=1,xticklabels=labels[1:])  # change subject name to check another subject
        ax.set_title('Confound Matrix' +structure_tag +task_tag +run_tag +' participant: '  + ID,fontsize = 15)
        ax.set_ylabel('Volumes',fontsize = 12)
        ax.set_xlabel('Confounds',fontsize = 12)

        if verbose:
            if not os.path.exists(confound_file.split('.')[0]+'.png') or redo:
                plt.savefig(confound_file.split('.')[0]+'.png')
            plt.show()

        else:
            if not os.path.exists(confound_file.split('.')[0]+'.png') or redo:
                plt.savefig(confound_file.split('.')[0]+'.png')
            plt.close(fig)

    def clean_images(self, ID=None, slice_wise=True, func_file=None, structure="", output_file=None, confounds_file=None, mask_file=None, task_name='', run_name='', standardize="zscore", detrend=False, high_pass=0.01, low_pass=0.17, tag_name="", n_jobs=1, redo=False, verbose=True):
        """
            Denoise fMRI data using specified confounds.

            Attributes
            ----------
            ID: str
                participant ID (required , default=None)
            slice_wise : bool
                whether to denoise slice by slice (optional , default=True)
            func_file : str
                4D preprocessed functional file. If None, it is automatically retrieved from preprocessing folders.
            structure : str
                could be 'brain' or 'spinalcord' name of the structure to work on
        """
        ########### Check initiation:
        if ID is None:
            raise(Exception('ID should be provided ex: ID="A001"'))

        if structure is None or confounds_file is None :
            raise Exception("'structure', 'confounds_files' and 'confound_infos' are required ")

        ###########  Load the func file and mask to extract the number of slices and the TR:

        if func_file is None:
            func_file=glob.glob(os.path.join(self.preproc_dir.format(ID), self.config["moco_files"]["dir"].format(ID,run_name,structure), self.config["moco_files"]["moco_mean_f"]))[0]
        func_img = nib.load(func_file) # load the func image
        slice_number = func_img.header.get_data_shape()[2] if slice_wise else 1 # extract the number of slices
        TR = func_img.header.get_zooms()[3] # extract TR value

        ########### Run the loop for each slice:
        physio_dir = os.path.join(self.denoising_dir.format(ID), task_name, self.config["denoising"]["denoised_dir"].format(ID), structure)  # output directory
        output_main_file = os.path.join(physio_dir, os.path.basename(func_file.split('.')[0] + "_"+tag_name+ '.nii.gz'))

        if not os.path.exists(output_main_file) or redo:
            if verbose:
                print("----------------------------------------------------------")
                print("Denoising fMRI data for participant: " + ID + " : " + task_name + " " + run_name)
                print("----------------------------------------------------------")

            os.makedirs(os.path.join(physio_dir, structure, f"tmp{run_name}"), exist_ok=True) # create tmp folder to save each slice denoised image

            slice_number = func_img.header.get_data_shape()[2] if slice_wise else 1
            slice_range = range(slice_number) if slice_wise else [None]

            for slice_nb in range(0,slice_number):
                slice_str = f"{slice_nb + 1:03d}" if slice_wise else ""
                confounds_f_slice = confounds_file.split("_slice")[0] +"_slice" + str(slice_str) + ".txt" if slice_wise else confounds_file
                output_tag = "_slice"+str(slice_str) if slice_wise else ""
                output_file = os.path.join(physio_dir, structure, f"tmp{run_name}", os.path.basename(func_file.split('.')[0] + "_"+tag_name+ output_tag + '.nii.gz'))

                if not os.path.exists(output_file) or redo:
                    mask_img=nib.load(mask_file) # load the mask image

                    if slice_wise:
                        func_slice = func_img.slicer[:,:,slice_nb:slice_nb+1,:] # cropped func slices
                        mask_slice = mask_img.slicer[:,:,slice_nb:slice_nb+1] # cropped mask slices
                    else:
                        func_slice = func_img
                        mask_slice = mask_img

                    # extract the mask value to check if there are not empty if so do not denoised this slice
                    data = mask_slice.get_fdata()

                    if np.mean(data) != 0:
                        Clean_image=image.clean_img(func_slice,
                                                confounds=confounds_f_slice,
                                                mask_img=mask_slice,
                                                detrend=detrend,
                                                standardize=standardize,
                                                low_pass=low_pass,
                                                high_pass=high_pass,
                                                t_r=TR)

                        Clean_image.to_filename(output_file)  #save image
                    else:
                        func_slice.to_filename(output_file)

            if slice_wise:
                # merge each slices in a single img

                if not os.path.exists(output_main_file) or redo:
                    nifti_files = glob.glob(os.path.join(physio_dir, structure, f"tmp{run_name}", "*.nii.gz"))
                    nifti_files.sort()  # Alphabetical sort

                    fsl_command = "fslmerge -z " + output_main_file + " " + " ".join(nifti_files)
                    os.system(fsl_command)  # run fsl command

                    shutil.rmtree(os.path.join(physio_dir, structure, f"tmp{run_name}"))  # remove the tmp folder

        output_meanfinal_file = output_main_file.split(".")[0] + "_mean.nii.gz"

        if not os.path.exists(output_meanfinal_file):
            fsl_command = "fslmaths " + output_main_file + " -Tmean " + output_meanfinal_file
            os.system(fsl_command)# run fsl command

        return output_main_file

    def standardize(self,input_files,output_files,json_files=None,mask_files=None,redo=False):
        #demean and standardized the signal by the std
        if not os.path.exists(input_files[0]) or redo:
            for file_nb in range(0,len(input_files)):
                timeseries=nib.load(input_files[file_nb]).get_fdata() # extract Time series dats
                signals= timeseries.reshape(-1, timeseries.shape[-1]).T # reshape timeseries (nb_volumes, nb_voxels)

                if signals.shape[0] == 1:
                    warnings.warn('Standardization of 3D signal has been requested but '
                              'would lead to zero values. Skipping.')
                else:
                    signals= timeseries.reshape(-1, timeseries.shape[-1]).T # reshape timeseries (nb_volumes, nb_voxels)
                    mean =  signals.mean(axis=0)
                    std = signals.std(axis=0)
                    std[std < np.finfo(np.float64).eps] = 1.  # avoid numerical problems
                    signals=signals-mean # demean
                    signals /= std

                # save into filename
                std_image=image.new_img_like(input_files[file_nb], signals.T.reshape(timeseries.shape),copy_header=True)
                std_image.to_filename(output_files[file_nb]) #save image

                if mask_files:
                    string='fslmaths ' + output_files[file_nb] + ' -mas ' + mask_files[file_nb] + ' ' + output_files[file_nb]
                    os.system(string)

                if json_files is not None:
                    infos={"standardize":True,
                          "mask":mask_files[file_nb]}
                    with open(json_files[file_nb], 'w') as f:
                        json.dump(infos, f) # save info