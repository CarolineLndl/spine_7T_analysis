    #!/usr/bin/env python
# coding: utf-8

# # Spinal cord fMRI second level
#
# Description: This workflow provides code for second level analyses 
# I. Run second level glm analysis for each subject and task
# II. Run ICC analyses
#------------------------------------------------------------------
#------ Initialization
#------------------------------------------------------------------
# Main imports ------------------------------------------------------------
import re, json, sys, os, glob, argparse
import pandas as pd
from nilearn.glm import threshold_stats_img
import nibabel as nib
import numpy as np
from collections import defaultdict

# Get the environment variable PATH_CODE
path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(path_code + '/config/config_spine_7t_fmri.json') as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
    config = json.load(config_file) # load config file should be open first and the path inside modified

parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs='+', default=[""])
parser.add_argument("--tasks", nargs='+', default=[""])
parser.add_argument("--verbose", default="False")
parser.add_argument("--redo", default="True")
parser.add_argument("--path-data", required=True)
args = parser.parse_args()

IDs = args.ids
tasks = args.tasks
verbose = args.verbose.lower() == "true"
redo = args.redo.lower() == "true"
path_data = os.path.abspath(args.path_data)

config["raw_dir"]=path_data
config["code_dir"]=path_code

participants_tsv = pd.read_csv(path_code + '/config/participants.tsv', sep='\t',dtype={'participant_id': str})

new_IDs=[]
if IDs == [""]:
    for ID in participants_tsv["participant_id"]:
        new_IDs.append(ID)

    IDs = new_IDs

if tasks != [""]:
    config["design_exp"]["task_names"] = tasks

#Import scripts
sys.path.append(path_code + "/code/") # Change this line according to your directory
import postprocess, preprocess, figures

glm_ana=postprocess.GLM_main(config,IDs=IDs)
preprocess_Sc=preprocess.Preprocess_Sc(config,IDs=IDs)
tsnr_ana=postprocess.TSNR_main(config, IDs,redo)
figures=figures.Figures_main(config, IDs=IDs)

# initialize directories
preprocessing_dir = os.path.join(config["raw_dir"], config["preprocess_dir"]["main_dir"])
denoising_dir= os.path.join(config["raw_dir"], config["denoising"]["dir"])
manual_dir = os.path.join(config["raw_dir"], config["manual_dir"])
main_fig_dir = os.path.join(config["raw_dir"], "derivatives/processing/figures/")
fig_task_dir = os.path.join(main_fig_dir, "task")
first_level_dir = os.path.join(config["raw_dir"], config["first_level"]["dir"])
second_level_dir = os.path.join(config["raw_dir"], config["second_level"]["dir"])

mask = os.path.join(first_level_dir.format('glm',"").split("sub")[0], "common_mask_PAM50.nii.gz")

#------------------------------------------------------------------
#------ Compute average tSNR
#------------------------------------------------------------------
for acq_name in config["design_exp"]["acq_names"]:
    tsnr_id_fname=[]
    cord_seg_file=[]
    warp_file=[]
    for ID in IDs:
        i_fnames_runs=[]
        tsnr_path=first_level_dir.format("tsnr",ID)
        dirs = [d for d in os.listdir(tsnr_path) if os.path.isdir(os.path.join(tsnr_path, d))]

        #select rest folder if exists otherwise take motor folder
        rest_dirs = [d for d in dirs if "rest" in d and acq_name in d]
        if len(rest_dirs) > 0:
            selected_dirs = rest_dirs
        else:
            selected_dirs = [d for d in dirs if acq_name in d]
        

        tsnr_id_fname.append(glob.glob(tsnr_path +"/"+ selected_dirs[0] + "/*_moco_tSNR.nii.gz")[0])
        cord_seg_file.append(glob.glob(os.path.join(preprocessing_dir.format(ID), 'func',selected_dirs[0], config["preprocess_f"]["func_seg"].format(ID,selected_dirs[0],"")))[0])
        warp_file.append(glob.glob(os.path.join(preprocessing_dir.format(ID), 'func', selected_dirs[0], f"sub-{ID}_{selected_dirs[0]}_from-func_to_PAM50_mode-image_xfm.nii.gz"))[0])

    fname_avg_tsnr=tsnr_ana.generate_average_tsnr_in_pam50(
        IDs=IDs,
        acq_name=acq_name,
        tsnr_fnames=tsnr_id_fname,
        seg_fnames=cord_seg_file,
        warp_fnames=warp_file,
        fname_mask=mask)

#------------------------------------------------------------------
#------ Run second level analysis
#------------------------------------------------------------------

print("")
print("=== Second level analysis script Start ===", flush=True)
print("Number of Participant included : ", len(IDs), flush=True)
print("===================================", flush=True)
print("")

common_mask_fname = os.path.join(first_level_dir.split("sub")[0], "common_mask_PAM50.nii.gz")

metrics_csv_pair=[]
for task_name in ["motor"]:
    for acq_name in config["design_exp"]["acq_names"]:
        i_fnames=[]
        tag="task-" + task_name + "_acq-" + acq_name
        os.makedirs(second_level_dir.format(tag), exist_ok=True)
        for ID in IDs:
            if ID=="090":
                continue
            # define the run name if multiple runs exist
            raw_func=sorted(glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz')))

            # take only the first run
            func_file = raw_func[0]

            # extract run number if exists
            match = re.search(r"_?(run-\d+)", func_file)
            run_name = match.group(1) if match else ""
            
            # find the corresponding first-level file
            i_fnames.append(glob.glob(os.path.join(first_level_dir.format('glm',ID), f"{tag}", f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"))[0])

        z_map_file=glm_ana.run_second_level_glm(i_fnames=i_fnames,
                                                            mask_fname=common_mask_fname,
                                                            task_name=tag,
                                                            run_name="",
                                                            parametric=False,
                                                            n_perm=10000,
                                                            vox_thr=0.01,
                                                            redo=redo,
                                                            verbose=verbose)

        metrics_csv,values_csv=glm_ana.extract_metrics(i_fname=z_map_file,threshold=0)
        metrics_csv_pair.append(metrics_csv)
                                                
        print("")
        print(f'=== Second level done for : {tag} ===', flush=True)
        print("=========================================", flush=True)

#------------------------------------------------------------------
#------ Plot group level tSNR and GLM
#------------------------------------------------------------------
# select the second level files
i_fnames_glm_pair=[];i_fnames_tSNR_pair=[]
for task_name in config["design_exp"]["task_names"]:
    for acq_name in config["design_exp"]["acq_names"]:
        tag="task-" + task_name + "_acq-" + acq_name
        i_fnames_glm_pair.append(os.path.join(second_level_dir.format(tag),f"n{len(IDs)-1}_{tag}_t_clustercorrected.nii.gz"))
        i_fnames_tSNR_pair.append(os.path.join(second_level_dir.format("tsnr"),f"tsnr_n{len(IDs)}_{acq_name}_avg_in_PAM50.nii.gz"))

output_fig=os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"], "second_level")

figures.bar_plot(csv_pair=metrics_csv_pair,output_fname=f"{output_fig}/n{len(IDs)}_glm_nb_vox.png")

figures.plot_two_maps(i_fnames_pair=i_fnames_glm_pair, 
                                   output_fname=f"{output_fig}/n{len(IDs)}_glm_avg_map.png",
                                   stat_min=2.3, 
                                   stat_max=6,
                                   cbar_label='t-value',
                                   background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
                                   underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]))

figures.plot_two_maps(i_fnames_pair=i_fnames_tSNR_pair, 
                                   output_fname=f"{output_fig}/n{len(IDs)}_tsnr_avg_map.png",
                                   stat_min=5, 
                                   stat_max=18,
                                   cmap='turbo',
                                   cbar_label='tSNR',
                                   background_fname=os.path.join(path_code, "template", config["PAM50_t2"]))

# --- Combine side by side ---
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
fig, axes = plt.subplots(1, 2, figsize=(4, 3.5))
for ax, fname, title in zip(axes,
                             [f"{output_fig}/n{len(IDs)}_tsnr_avg_map.png", f"{output_fig}/n{len(IDs)}_glm_avg_map.png"],
                             ["tSNR", "GLM"]):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=7, fontweight='bold', fontname="Arial")

plt.tight_layout()
plt.savefig(f"{output_fig}/n{len(IDs)}_combined_map.png", dpi=300)
plt.close()

## Next steps:
# extract and save the number of voxels within each group maps
# extract and save the t-value distribution
# generate bar plot
# generate distribtuion plot
# generate tsnr violin plot
# plot the three at the right side of the previous figure

#------------------------------------------------------------------
#------ compute test-retest reproductibility using ICC 
#------------------------------------------------------------------

# ----------  betwen shimSlice run01 vs run02 ---
print("", flush=True)
print(f'=== ICC between sliceShim run-01 and run-02  start', flush=True)
print("=========================================", flush=True)
output_dir=second_level_dir.format("icc_shimSlice_run01_run02")
i_fnames_by_runs = []
tag="task-motor_acq-shimSlice+3mm"
IDs_2runs=[]
for ID in IDs:
    if ID=="090":
        continue
    raw_func = sorted(glob.glob(os.path.join( config["raw_dir"], f"sub-{ID}", "func", f"sub-{ID}_{tag}_*bold.nii.gz")))
    
    # Only keep participants with 2 runs
    if len(raw_func) != 2:
        continue
    IDs_2runs.append(ID)
    i_fnames_runs = []
    for fname in raw_func:
        run_name = re.search(r"_?(run-\d+)", fname).group(1)
        stat_map = glob.glob(os.path.join(
            first_level_dir.format("glm",ID), tag, f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"
        ))[0]
        i_fnames_runs.append(stat_map)
    
    i_fnames_by_runs.append(i_fnames_runs)

icc_maps,icc_maps_s=glm_ana.run_icc(IDs=IDs_2runs,i_fnames=i_fnames_by_runs,o_dir=output_dir, mask_file=mask, threshold=0)
#postprocess.plot_ICC_maps(i_fname=icc_maps,
 #                         output_fname=output_fig + "/icc_run-01_run-02.png",
  #                        cmap="turbo",
   #                       stat_min=0.1,
    #                      stat_max=0.9,
     #                     background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
      #                    underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]))

print("", flush=True)
print(f'=== ICC between sliceShim run-01 and run-02  done', flush=True)
print("=========================================", flush=True)

# ----------  betwen shimBase and shimSlice ---
print("", flush=True)
print(f'=== ICC between sliceShim aand sliceBase  start', flush=True)
print("=========================================", flush=True)
output_dir=second_level_dir.format("icc_shimBase_shimSlice")
os.makedirs(output_dir, exist_ok=True)
i_fnames_by_runs = []

for ID in IDs_2runs:
    i_fnames_runs = []
    for acq_name in config["design_exp"]["acq_names"]:
        tag="task-motor" + "_acq-" + acq_name
        raw_func = sorted(glob.glob(os.path.join(config["raw_dir"], f"sub-{ID}", "func", f"sub-{ID}_{tag}_*bold.nii.gz")))
        match = re.search(r"_?(run-\d+)", raw_func[0])
        run_name = match.group(1) if match else ""

        stat_map = glob.glob(os.path.join(first_level_dir.format("glm",ID), tag, f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"))[0]
        i_fnames_runs.append(stat_map)
    
    i_fnames_by_runs.append(i_fnames_runs)

icc_maps,icc_maps_s=glm_ana.run_icc(IDs=IDs_2runs,i_fnames=i_fnames_by_runs,o_dir=output_dir,  mask_file=mask, threshold=0)
#postprocess.plot_ICC_maps(i_fname=icc_maps,
 #                         output_fname=output_fig + "/icc_shimBase_ShimSlice.png",
  #                        cmap="turbo",
   #                       stat_min=0.1,
    #                      stat_max=0.9,
     #                     background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
      #                    underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]))

print("", flush=True)
print(f'=== ICC between sliceShim aand sliceBase  done', flush=True)
print("=========================================", flush=True)
