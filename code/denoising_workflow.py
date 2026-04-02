#!/usr/bin/env python
# coding: utf-8

# # Spinal cord fMRI denoising 

# @ author of the script: Caroline Landelle, caroline.landelle@mcgill.ca // landelle.caroline@gmail.com
#
# Description: This workflow provides code for BOLD signal fMRI denoising, template registration and smoothing
# For each individual, we accounted for the physiological and other noise sources by modeling nuisance noises present in CSF and by collecting physiological data using the Tapas PhysiO toolbox (Kasper et al., 2017). First, we used the RETROspective Image CORrection (RETROICOR) procedure (Glover et al., 2000) [..] Second, [...] we used the CompCor (Behzadi et al., 2007) approach [...]. Finally, we applied a bandpass filter 0.01-0.17 Hz to emphasize low-frequency signals of interest.
#
# I. Extract slice wise motion parameters
# II. Compute outliers calculation
# IV. Compcor calculation
# V. Signal cleaning
#
#------------------------------------------------------------------
#------ Initialization
#------------------------------------------------------------------
# Main imports ------------------------------------------------------------
import json,sys, os, glob, re, argparse
import pandas as pd
from nilearn import image

# Get the environment variable PATH_CODE
path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(path_code, "config", "config_spine_7t_fmri.json")) as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
    config = json.load(config_file) # load config file should be open first and the path inside modified

parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs='+', default=[""])
parser.add_argument("--tasks", nargs='+', default=[""])
parser.add_argument("--verbose", default="False")
parser.add_argument("--manual_centerline", default="False")
parser.add_argument("--auto_vert_labels", default="True")
parser.add_argument("--redo", default="True")
parser.add_argument("--path-data", required=True)
args = parser.parse_args()

IDs = args.ids
tasks = args.tasks
verbose = args.verbose.lower() == "true"
manual_centerline = args.manual_centerline.lower() == "true"
auto_vert_labels = args.auto_vert_labels.lower() == "true"
redo = args.redo.lower() == "true"
path_data = os.path.abspath(args.path_data)

config["raw_dir"]=path_data
config["code_dir"]=path_code

participants_tsv = pd.read_csv(os.path.join(path_code, "config", "participants.tsv"), sep='\t',dtype={'participant_id': str})

new_IDs=[]
if IDs == [""]:
    for ID in participants_tsv["participant_id"]:
        new_IDs.append(ID)

    IDs = new_IDs

if tasks != [""]:
    config["design_exp"]["task_names"] = tasks

#Import scripts
sys.path.append(os.path.join(path_code, "code"))  # Change this line according to your directory
from denoising import Denoising
from preprocess import Preprocess_Sc, Preprocess_main

denoising=Denoising(config,IDs=IDs)
preprocess_Sc=Preprocess_Sc(config, IDs=IDs)
preprocess_main=Preprocess_main(config, IDs=IDs)

# initialize directories
preprocessing_dir = os.path.join(config["raw_dir"], config["preprocess_dir"]["main_dir"])
derivatives_dir = os.path.join(config["raw_dir"], config["derivatives_dir"])
manual_dir = os.path.join(config["raw_dir"], config["manual_dir"])
denoising_dir = os.path.join(config["raw_dir"], config["denoising"]["dir"])

#------------------------------------------------------------------
#------ Denoising
#------------------------------------------------------------------
print("")
print("=== Denoising script Start ===", flush=True)
print("Participant(s) included : ", IDs, flush=True)
print("===================================", flush=True)
print("")

for ID_nb,ID in enumerate(IDs):
    print("", flush=True)
    print(f'=== Denoising start for :  {ID} ===', flush=True)

    for task_name in config["design_exp"]["task_names"]:
        for acq_name in config["design_exp"]["acq_names"]:
            tag = "task-" + task_name + "_acq-" + acq_name
            raw_func = glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz'))

            for func_file in raw_func:
                # Check run number if multiple run exists
                match = re.search(r"_?(run-\d+)", func_file)
                if match:
                    run_name=match.group(1)
                    print(run_name)
                else:
                    run_name=""

                moco_file=glob.glob(os.path.join(preprocessing_dir.format(ID), config["preprocess_dir"]["func_moco"].format(tag), config["preprocess_f"]["func_moco"].format(ID,tag,run_name)))[0]
                cord_seg_file = glob.glob(os.path.join(preprocessing_dir.format(ID), 'func',tag, config["preprocess_f"]["func_seg"].format(ID,tag,"")))[0]
                csf_seg_file = glob.glob(os.path.join(preprocessing_dir.format(ID), 'func',tag, config["preprocess_f"]["func_csf"].format(ID,tag,"")))[0]
                
                if cord_seg_file is None:
                    raise RuntimeError(f"No mask file found for subject {ID}, task {tag}, run {run_name}. Please check the preprocessing outputs and manual corrections.")

                if csf_seg_file is None:
                    raise RuntimeError(f"No mask file found for subject {ID}, task {tag}, run {run_name}. Please check the preprocessing outputs and manual corrections.")
          
                #------------------------------------------------------------------
                #------ moco parameters
                #------------------------------------------------------------------
                moco_param_f=glob.glob(os.path.join(preprocessing_dir.format(ID), config["preprocess_dir"]["func_moco"].format(tag), config["preprocess_f"]["moco_params"].format(tag,run_name)))
                denoising.moco_params(ID=ID,input_file=moco_param_f, task_name=tag,run_name=run_name,redo=redo)

                #------------------------------------------------------------------
                #------ outliers parameters
                #------------------------------------------------------------------
                denoising.outliers(ID=ID,task_name=tag, mask_file= cord_seg_file,run_name=run_name,redo=redo)


                #------------------------------------------------------------------
                #------ Compute compcor
                #------------------------------------------------------------------
                compcor_out, DCT_out = denoising.confounds_ts(
                    ID=ID,
                    task_name=tag,
                    run_name=run_name,
                    func_file=moco_file,
                    mask_seg_file=cord_seg_file,
                    mask_csf_file=csf_seg_file,
                    n_compcor=15,
                    compcor=True,
                    DCT=False,
                    redo=redo
                )

                #------------------------------------------------------------------
                #------ Combine all confounds together
                #------------------------------------------------------------------
                confound_infos={'outliers':1,'moco':2,'compcor':15} #'outliers':1 #'moco':2,
                confounds=denoising.combine_confounds(
                    ID=ID,
                    task_name=tag,
                    run_name=run_name,
                    confounds_infos=confound_infos,
                    outliers_confounds=True,
                    retroicor_confounds=False,
                    compcor_confounds=True,
                    moco_confounds=True,
                    DCT_confounds=False,
                    slice_wise=True,
                    redo=redo
                )

                #save the plots
                denoising.plot_confound_design(
                    ID=ID,
                    confound_file=confounds.split("_slice")[0] + "_slice010_z.txt",
                    structure="",
                    task_name=tag,
                    run_name=run_name,
                    confounds_infos=confound_infos,
                    redo=redo,
                    verbose=verbose)

                #------------------------------------------------------------------
                #------ Apply denoising, HP filtering
                #------------------------------------------------------------------
                Clean_image_file=denoising.clean_images(
                    ID=ID,
                    func_file=moco_file,  # Find functional file,
                    task_name=tag,
                    run_name=run_name,
                    confounds_file=confounds,
                    mask_file=cord_seg_file,
                    high_pass=0.01,
                    low_pass= None,
                    tag_name= "HP_nostd", #std means the data were z-scored
                    standardize=False,
                    n_jobs=4,
                    redo=redo)
                
                #------------------------------------------------------------------
                #------ Apply smoothing
                #------------------------------------------------------------------
                smooth_imag_file=Clean_image_file.split(".")[0] + "_s.nii.gz"

                if not os.path.exists(smooth_imag_file):
                    smoothed_image=image.smooth_img(Clean_image_file, [1.5,1.5,6])
                    smoothed_image.to_filename(smooth_imag_file)
                
                

    print(f'=== Denoising done for : {ID} ===', flush=True)
    print("=========================================", flush=True)