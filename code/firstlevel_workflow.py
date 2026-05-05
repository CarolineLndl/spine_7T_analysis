    #!/usr/bin/env python
# coding: utf-8

# # Spinal cord fMRI denoising 

# @ author of the script: Caroline Landelle, caroline.landelle@mcgill.ca // landelle.caroline@gmail.com
#
# Description: This workflow provides code for first level analyses 
# I. Run first level analysis for each subject and task
# II. Normalize the resulting stat maps to PAM50 template space
#
#------------------------------------------------------------------
#------ Initialization
#------------------------------------------------------------------
# Main imports ------------------------------------------------------------
import re, json, sys, os, glob, argparse
import pandas as pd
from nilearn.glm import threshold_stats_img
import nibabel as nib
import numpy as np

# Get the environment variable PATH_CODE
path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(path_code, 'config', 'config_spine_7t_fmri.json')) as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
    config = json.load(config_file) # load config file should be open first and the path inside modified

parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs='+', default=[""])
parser.add_argument("--tasks", nargs='+', default=[""])
parser.add_argument("--verbose", default="False")
parser.add_argument("--redo", default="true")
parser.add_argument("--path-data", required=True)
args = parser.parse_args()

IDs = args.ids
tasks = args.tasks
verbose = args.verbose.lower() == "true"
redo = args.redo.lower() == "true"
path_data = os.path.abspath(args.path_data)

config["raw_dir"]=path_data
config["code_dir"]=path_code

participants_tsv = pd.read_csv(os.path.join(path_code, 'config', 'participants.tsv'), sep='\t',dtype={'participant_id': str})

new_IDs=[]
if IDs == [""]:
    for ID in participants_tsv["participant_id"]:
        new_IDs.append(ID)

    IDs = new_IDs

#if tasks != [""]:
#    config["design_exp"]["task_names"] = tasks

#Import scripts
sys.path.append(os.path.join(path_code, "code")) # Change this line according to your directory
import postprocess, preprocess
import figures

glm_ana=postprocess.GLM_main(config,IDs=IDs)
preprocess_Sc=preprocess.Preprocess_Sc(config,IDs=IDs)
figures=figures.Figures_main(config, IDs=IDs)

# initialize directories
preprocessing_dir = os.path.join(config["raw_dir"], config["preprocess_dir"]["main_dir"])
denoising_dir= os.path.join(config["raw_dir"], config["denoising"]["dir"])
manual_dir = os.path.join(config["raw_dir"], config["manual_dir"])
first_level_dir = os.path.join(config["raw_dir"], config["first_level"]["dir"])
main_fig_dir = os.path.join(config["raw_dir"], "derivatives", "processing", "figures")
fig_dir = os.path.join(main_fig_dir, "first_level")
os.makedirs(main_fig_dir, exist_ok=True)

#------------------------------------------------------------------
#------ I. Compute tSNR
#------------------------------------------------------------------
print("=== tSNR script Start ===", flush=True)
print("Participant(s) included : ", IDs, flush=True)
print("===================================", flush=True)
print("")

# Compute individual level
tsnr_ana=postprocess.TSNR_main(config, IDs,redo=redo)
tsnr_ana.generate_tsnr_maps_and_csv()

print("=== tSNR script Done ===", flush=True)
print("===================================", flush=True)
print("")

#------------------------------------------------------------------
#------ II. Plot EPI comparison
#------------------------------------------------------------------
print("=== Epi comparison script Start ===", flush=True)
print("===================================", flush=True)
print("")

# --- Group figure (IDs_EPIcomp only)
# EPI comparison figure
fig_epi_comparison = postprocess.EpiComparison(config, IDs, redo)
fig_epi_comparison.create_figure(show_avg=False)
    

print("=== EPI comparison script Done ===", flush=True)
print("===================================", flush=True)
print("")
#------------------------------------------------------------------
#------ III. Run First level
#------------------------------------------------------------------
config["design_exp"]["task_names"] = ["motor"]
print("")
print("=== First level analysis script Start ===", flush=True)
print("Participant(s) included : ", IDs, flush=True)
print("===================================", flush=True)
print("")

#------ I.1 Select files 
norm_mask=[]
for ID_nb, ID in enumerate(IDs):
    if ID=="090":
        continue 
    print("", flush=True)
    print(f'=== First level start for :  {ID} ===', flush=True)

    for task_name in config["design_exp"]["task_names"]:
        for acq_name in config["design_exp"]["acq_names"]:
            tag="task-" + task_name + "_acq-" + acq_name
            raw_func=glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz'))
            for func_file in raw_func:
                # Check run number if multiple run exists
                match = re.search(r"_?(run-\d+)", func_file)
                if match:
                    run_name=match.group(1)
                    print(run_name)
                else:
                    run_name=""

                denoised_fmri=glob.glob(os.path.join(denoising_dir.format(ID ), tag, config["denoising"]["denoised_dir"],"*"+run_name+"*_nostd_s.nii.gz"))[0]
                cord_seg_file = glob.glob(os.path.join(preprocessing_dir.format(ID), 'func',tag, config["preprocess_f"]["func_seg"].format(ID,tag,"")))[0]
                warp_file = os.path.join(preprocessing_dir.format(ID), 'func', tag, f"sub-{ID}_{tag}_from-func_to_PAM50_mode-image_xfm.nii.gz")

                if not os.path.exists(cord_seg_file):
                    raise RuntimeError(f"No mask file found for subject {ID}, task {tag}. Please check the preprocessing outputs and manual corrections.")

                # Select warp file
                if not os.path.exists(warp_file):
                    raise RuntimeError(f"No warp file found for subject {ID}, task {tag}. Please check the preprocessing outputs and manual corrections.")

                events_file=glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*{run_name}*events.tsv'))[0]

                #------ I.2 Run first level GLM
                stat_maps=glm_ana.run_first_level_glm(ID=ID,
                                                          i_fname=denoised_fmri,
                                                          events_file=events_file,
                                                          mask_file=cord_seg_file,
                                                          task_name=tag,
                                                          run_name=run_name,
                                                          redo=redo,
                                                          verbose=verbose)

                #------ I.2 Apply correction and extract metrics
                for i, contrast_fname in enumerate(stat_maps):
                    # Apply correction
                    corr_type="fpr";alpha=0.01;cluster=0
                    
                    fname_thr_img=stat_maps[i].split(".")[0] +f"_{corr_type}_{str(alpha)[2:]}_{str(cluster)}cluster.nii.gz"
                    
                    if not os.path.exists(fname_thr_img) or redo:
                        thresholded_map, threshold = threshold_stats_img(stat_maps[i],
                                                                        alpha=alpha,
                                                                        height_control=corr_type,
                                                                        cluster_threshold=cluster,
                                                                            two_sided=False)
                        thresholded_map.to_filename(fname_thr_img)
   
                #------ I.3 Normalization 
                # Normlaize the resulting stat maps to PAM50 template space
                for i, contrast_fname in enumerate(stat_maps):
                    norm_stat_maps=preprocess_Sc.apply_warp(
                            i_img=[stat_maps[i]], # input clean image
                            ID=[ID],
                            o_folder=[os.path.dirname(stat_maps[i])], # output folder
                            dest_img=os.path.join(path_code, "template", config["PAM50_t2"]), # PAM50 template
                            warping_field=warp_file,
                            tag="_inTemplate",
                            mean=False,
                            n_jobs=1,
                            verbose=False,
                            redo=redo)
                
                # Normalize the individual masks to template space
                norm_mask.append(preprocess_Sc.apply_warp(
                            i_img=[cord_seg_file], # input clean image
                            ID=[ID],
                            o_folder=[os.path.dirname(stat_maps[i])], # output folder
                            dest_img=os.path.join(path_code, "template", config["PAM50_t2"]), # PAM50 template
                            warping_field=warp_file,
                            tag="_inTemplate",
                            mean=False,
                            n_jobs=1,
                            threshold=0.1,
                            verbose=False,
                            redo=redo)[0])
    
    print(f'=== First level done for : {ID} ===', flush=True)
    print("=========================================", flush=True)
    print("")

#------------------------------------------------------------------
#------ II. Extract the common mask for all participants and tasks
#------------------------------------------------------------------
glm_dir = os.path.join(config["raw_dir"], config["first_level"]["dir"].format("glm",""))
common_mask_fname = os.path.join(glm_dir.split("sub")[0], "common_mask_PAM50.nii.gz")

if not os.path.exists(common_mask_fname) or redo:
    norm_mask_data = [nib.as_closest_canonical(nib.load(f)).get_fdata() for f in norm_mask]
    n_files = len(norm_mask_data)

    # Compute common mask (n-1)---
    sum_mask = np.sum(norm_mask_data, axis=0)
    common_mask_data = (sum_mask >= n_files-3).astype(np.uint8)
    common_mask_fname = os.path.join(glm_dir.split("sub")[0], "common_mask_PAM50.nii.gz")
    common_mask_img = nib.Nifti1Image(common_mask_data, affine=nib.load(norm_mask[0]).affine)
    common_mask_img.to_filename(common_mask_fname)
    common_mask_data = common_mask_img.get_fdata()

    # ---  Extract the z-slices that contain the common mask ---
    z_indices = np.where(np.any(common_mask_data > 0, axis=(0,1)))[0]
    z_min, z_max = z_indices[[0, -1]]
    z_size = z_max - z_min + 1


#------------------------------------------------------------------
#------ III.  Plot first level results: shimBase vs. shimSlice
#------------------------------------------------------------------
i_fnames_by_runs = []
for ID in IDs:
    # Check if there are multiple runs for this participant
    i_fnames_runs = []
    for task_name in config["design_exp"]["task_names"]:
        for acq_name in config["design_exp"]["acq_names"]:
            tag="task-" + task_name + "_acq-" + acq_name
            raw_func=sorted(glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz')))
            if len(raw_func)==2 and tag=="task-motor_acq-shimSlice+3mm":
                for fname in raw_func:
                    match = re.search(r"_?(run-\d+)", fname)
                    run_name = match.group(1)
                    i_fnames_runs.append(glob.glob(os.path.join(first_level_dir.format("glm",ID), f"{tag}", f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"))[0])
            else:
                match = re.search(r"_?(run-\d+)", raw_func[0])
                if match:
                    run_name=match.group(1)
                else:
                    run_name=""
                i_fnames_runs.append(glob.glob(os.path.join(first_level_dir.format("glm",ID), f"{tag}", f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"))[0])

    i_fnames_by_runs.append(i_fnames_runs)

figures.plot_first_level_maps(i_fnames=i_fnames_by_runs,
                                         output_fname=os.path.join(fig_dir, f"first_level_task_by_runs_n{len(i_fnames_by_runs)}.png"),
                                          background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
                                          mask_fname=common_mask_fname,
                                          titles=["shimBase","shimSlice","shimSlice"],
                                         #underlay_fname=os.path.join(path_code, "template", config["PAM50_cord"]),
                                          task_name=tag,
                                          participant_ids=IDs,
                                          verbose=True,
                                          redo=redo)

