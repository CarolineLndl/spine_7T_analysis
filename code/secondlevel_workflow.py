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

# Get the environment variable PATH_CODE
path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(path_code, "config", "config_spine_7t_fmri.json")) as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
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

participants_tsv = pd.read_csv(os.path.join(path_code, "config", "participants.tsv"), sep='\t', dtype={'participant_id': str})

new_IDs=[]
if IDs == [""]:
    for ID in participants_tsv["participant_id"]:
        new_IDs.append(ID)

    IDs = new_IDs

if tasks != [""]:
    config["design_exp"]["task_names"] = tasks

#Import scripts
sys.path.append(os.path.join(path_code, "code")) # Change this line according to your directory
import postprocess, preprocess, figures

glm_ana = postprocess.GLM_main(config,IDs=IDs)
preprocess_Sc = preprocess.Preprocess_Sc(config,IDs=IDs)
tsnr_ana = postprocess.TSNR_main(config, IDs,redo)
figures = figures.Figures_main(config, IDs=IDs)

# initialize directories
preprocessing_dir = os.path.join(config["raw_dir"], config["preprocess_dir"]["main_dir"])
denoising_dir= os.path.join(config["raw_dir"], config["denoising"]["dir"])
manual_dir = os.path.join(config["raw_dir"], config["manual_dir"])
main_fig_dir = os.path.join(config["raw_dir"], "derivatives", "processing", "figures")
fig_task_dir = os.path.join(main_fig_dir, "task")
first_level_dir = os.path.join(config["raw_dir"], config["first_level"]["dir"])
second_level_dir = os.path.join(config["raw_dir"], config["second_level"]["dir"])

mask = os.path.join(first_level_dir.format('glm',"").split("sub")[0], "common_mask_PAM50.nii.gz")

#------------------------------------------------------------------
#------ Compute average tSNR
#------------------------------------------------------------------
for acq_name in config["design_exp"]["acq_names"]:
    tsnr_id_fname = []
    cord_seg_file = []
    warp_file = []
    for ID in IDs:
        i_fnames_runs = []
        snr_path = first_level_dir.format("snr",ID)
        dirs = [d for d in os.listdir(snr_path) if os.path.isdir(os.path.join(snr_path, d))]

        # Select rest folder if it exists otherwise take motor folder
        rest_dirs = [d for d in dirs if "rest" in d and acq_name in d]
        if len(rest_dirs) > 0:
            selected_dirs = rest_dirs
        else:
            selected_dirs = [d for d in dirs if acq_name in d]

        task_name = selected_dirs[0].split("_")[0].split("-")[1]
        
        tsnr_id_fname.append(glob.glob(os.path.join(snr_path, selected_dirs[0], "*_moco_tsnr.nii.gz"))[0])
        cord_seg_file.append(glob.glob(os.path.join(preprocessing_dir.format(ID), 'func',selected_dirs[0], config["preprocess_f"]["func_seg"].format(ID,selected_dirs[0],"")))[0])
        warp_file.append(glob.glob(os.path.join(preprocessing_dir.format(ID), 'func', selected_dirs[0], f"sub-{ID}_{selected_dirs[0]}_from-func_to_PAM50_mode-image_xfm.nii.gz"))[0])

    fname_avg_tsnr = tsnr_ana.generate_average_tsnr_in_pam50(
        IDs=IDs,
        task_name=task_name,
        acq_name=acq_name,
        tsnr_fnames=tsnr_id_fname,
        seg_fnames=cord_seg_file,
        warp_fnames=warp_file,
        fname_mask=mask)
    
    output_fig = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"], "second_level")

    box_plot={}
    for metric in ["tsnr", "ssnr"]:
        fname_csv = glob.glob(os.path.join(snr_path.split("sub")[0], f"{metric}*_metrics_reduced.csv"))[0]
        stat_file = glob.glob(os.path.join(snr_path.split("sub")[0], f"{metric}*_metrics_reduced_stats.csv"))[0]
        (ymin, ymax) = (6, 17) if metric == "tsnr" else (1.5, 4.5)
        y_label="temporal SNR" if metric == "tsnr" else "spatial SNR"
        
        box_plot[metric] = figures.boxplots(csv_file=fname_csv, output_fname=os.path.join(output_fig, f"{len(IDs)}_{metric}_boxplot.png"),
                                  ymin=ymin, ymax=ymax, stats_file=stat_file,
                                  specify_y_label=y_label,
                                  x_data="acq", x_order=["shimBase","shimSlice"],
                                  indiv_values=True,
                                  y_data=metric, redo=redo)

#------------------------------------------------------------------
#------ Compute average FD
#------------------------------------------------------------------
os.makedirs(second_level_dir.format("FD"), exist_ok=True)
fd_files=[]
for acq_name in config["design_exp"]["acq_names"]:
    i_fnames=[];run_names=[]
    tag="task-motor" + "_acq-" + acq_name

    for ID in IDs:
        # define the run name if multiple runs exist
        raw_func=sorted(glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz')))
        func_file = raw_func[0] # take only the first run
        match = re.search(r"_?(run-\d+)", func_file)
        run_name = match.group(1) if match else ""
        run_names.append(run_name)

    output_file=second_level_dir.format("FD") + f"/n{len(IDs)}_{tag}_FD.csv"
    fd_files.append(glm_ana.extract_FD(IDs=IDs,task_name=tag,run_name=run_names,output_file=output_file,redo=False))

postprocess.pair_ttest(
    csv_files=fd_files,
    value_col='mean_FD',
    acq_col='acq',
    task_filter=None, task_col=None,
    output_fname=output_file.split('.csv')[0]+"_stats.csv", redo=True)

#------------------------------------------------------------------
#------ Run second level analysis
#------------------------------------------------------------------

print("")
print("=== Second level analysis script Start ===", flush=True)
print("Number of Participant included : ", len(IDs), flush=True)
print("===================================", flush=True)
print("")

common_mask_fname = os.path.join(first_level_dir.split("sub")[0], "common_mask_PAM50.nii.gz").format("glm")

values_csv_pair={};metrics_csv_pair={}
for cluster_corr in [0.01,0.001]:
    values_csv_pair[cluster_corr]={};metrics_csv_pair[cluster_corr]={}
    for vox_thr in [0.005]: # for now we only use 0.005 as it is the one used in the first level, but we can add more if needed
        values_csv_pair[cluster_corr][vox_thr]=[];metrics_csv_pair[cluster_corr][vox_thr]=[]
        for task_name in ["motor"]:
            for acq_name in config["design_exp"]["acq_names"]:
                i_fnames=[]
                tag = "task-" + task_name + "_acq-" + acq_name
                for ID in IDs:
                    raw_func = sorted(glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz')))
                    func_file = raw_func[0]  # take only the first run

                    # extract run number if exists
                    match = re.search(r"_?(run-\d+)", func_file)
                    run_name = match.group(1) if match else ""

                    i_fnames.append(glob.glob(os.path.join(first_level_dir.format('glm',ID), f"{tag}", f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"))[0])  # find the corresponding first-level file
                
                z_map_file = glm_ana.run_second_level_glm(i_fnames=i_fnames,
                                                                mask_fname=common_mask_fname,
                                                                task_name=tag,
                                                                run_name="",
                                                                parametric=False,
                                                                n_perm=10000,
                                                                vox_thr=vox_thr,
                                                                cluster_corr=cluster_corr,
                                                                redo=redo,
                                                                verbose=verbose)

                metrics_csv,values_csv = glm_ana.extract_metrics(i_fname=z_map_file,threshold=0)
                metrics_csv_pair[cluster_corr][vox_thr].append(metrics_csv)
                values_csv_pair[cluster_corr][vox_thr].append(values_csv)

                print(metrics_csv_pair)
                                                    
                print("")
                print(f'=== Second level done for : {tag}, cluster: {cluster_corr} vox: {vox_thr} ===', flush=True)
                print("=========================================", flush=True)

#------------------------------------------------------------------
#------ Plot group level tSNR and GLM
#------------------------------------------------------------------
# --- Plot tSNR --- 
i_fnames_tSNR_pair=[] 
for acq_name in config["design_exp"]["acq_names"]:
    i_fnames_tSNR_pair.append(os.path.join(second_level_dir.format("snr"), f"tsnr_n{len(IDs)}_{acq_name}_avg_in_PAM50.nii.gz"))

tsnr_plot = figures.plot_fmri_maps(i_fnames=i_fnames_tSNR_pair,
                                   output_fname=os.path.join(output_fig, f"n{len(IDs)}_tsnr_avg_map.png"),
                                   stat_min=5, 
                                   stat_max=16,
                                   cmap='turbo',
                                   cbar_label='tSNR',
                                   background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),redo=redo)

figures.combine_plots(output_fname=os.path.join(output_fig, f"n{len(IDs)}_combined_SNR_plots.png"),
                      map_files=[tsnr_plot],
                      graph_files=[box_plot["tsnr"],box_plot["ssnr"]],
                      figsize=(3.5, 3.5), redo=redo)

# select the second level files
# --- Plot GLM ---
output_fig = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"], "second_level")
i_fnames_glm_pair = {}
glm_plot={};glm_axial_plot={};dist_plot={};bar_plot={}
task_name = "motor"
for cluster_corr in [0.01,0.001]:
    i_fnames_glm_pair[cluster_corr] = {}
    glm_plot[cluster_corr] = {}; glm_axial_plot[cluster_corr] = {}; dist_plot[cluster_corr] = {}; bar_plot[cluster_corr]={}
    z_slices=[280,266,256,243,225] # slices to display in axial view, 
    for vox_thr in [0.005]:
        i_fnames_glm_pair[cluster_corr][vox_thr] = []
        for acq_name in config["design_exp"]["acq_names"]:
            tag = "task-" + task_name + "_acq-" + acq_name
            i_fnames_glm_pair[cluster_corr][vox_thr].append(os.path.join(second_level_dir.format("glm"),f"cluster_p{cluster_corr}_vox{vox_thr}_perm10000",tag, f"n{len(IDs)}_{tag}_t_clustercorrected.nii.gz"))

        glm_plot[cluster_corr][vox_thr] = figures.plot_fmri_maps(i_fnames=i_fnames_glm_pair[cluster_corr][vox_thr],
                                   output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_{cluster_corr}_vox{vox_thr}_avg_map.png"),
                                   stat_min=3, 
                                   stat_max=6,
                                   cbar_label='t-value',
                                   z_slices=z_slices,
                                   background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
                                   underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]),redo=redo)

        glm_axial_plot[cluster_corr][vox_thr] =figures.plot_fmri_maps_axial(i_fnames=i_fnames_glm_pair[cluster_corr][vox_thr],
                             output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_axial_{cluster_corr}_vox{vox_thr}_avg_map.png"),
                             stat_min=3,
                             stat_max=6,
                             cbar_label='t-value',
                             background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
                             underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]),
                             z_slices=z_slices,
                             n_slices=len(z_slices),  
                             redo=redo)


        bar_plot[cluster_corr][vox_thr] = figures.bar_plot(
        csv_pair=metrics_csv_pair[cluster_corr][vox_thr],figsize=(2, 2.7),
        output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_cluster{cluster_corr}_vox{vox_thr}_nb_vox.png"),redo=redo)

        dist_plot[cluster_corr][vox_thr] = figures.plot_dist(
            csv_pair=[values_csv_pair[cluster_corr][vox_thr][1],values_csv_pair[cluster_corr][vox_thr][0]],
            maps_name=["shimSlice","shimBase"],
            colors=["#ED263F","#ADA8A8"],figsize=(2, 2.7),
            output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_cluster{cluster_corr}_vox{vox_thr}_distr.png"),redo=redo)
        
        figures.combine_plots(output_fname=os.path.join(output_fig, f"n{len(IDs)}_combined_task_cluster_{cluster_corr}_vox_{vox_thr}_plots.png"),
                      map_files=[glm_plot[cluster_corr][vox_thr]],
                      axial_files=[glm_axial_plot[cluster_corr][vox_thr]],
                      graph_files=[bar_plot[cluster_corr][vox_thr],dist_plot[cluster_corr][vox_thr]],
                      label_idx=True,
                      figsize=(5, 3.8), redo=True)

#------------------------------------------------------------------
#------ compute test-retest reproductibility using ICC 
#------------------------------------------------------------------

# ----------  between shimSlice run01 vs run02 ---
print("", flush=True)
print(f'=== ICC between sliceShim run-01 and run-02  start', flush=True)
print("=========================================", flush=True)
output_dir = os.path.join(second_level_dir.format("icc"), "shimSlice_run01_vs_run02")
os.makedirs(output_dir, exist_ok=True)
i_fnames_by_runs = []
tag = "task-motor_acq-shimSlice+3mm"
IDs_2runs = []
for ID in IDs:
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

icc_maps,icc_maps_s=glm_ana.run_icc(IDs=IDs_2runs,i_fnames=i_fnames_by_runs,o_dir=output_dir, mask_file=mask, threshold=0, redo=redo)
z_slices=[324,306,268,238,222,204] # slices to display in axial view,
icc_plot=figures.plot_fmri_maps(i_fnames=[icc_maps],
                          output_fname=os.path.join(f"{output_fig}", f"n{len(IDs_2runs)}_icc_run-01_run-02.png"),
                          titles=[""],
                          cmap="rainbow",
                          cbar_label='ICC',
                          z_slices=z_slices,
                          stat_min=0.3,
                          stat_max=0.9,
                          background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
                          redo=redo)

icc_axial_plot=figures.plot_fmri_maps_axial(i_fnames=[icc_maps],
                             output_fname=os.path.join(output_fig, f"n{len(IDs_2runs)}_icc__run-01_run-02_axial.png"),
                             cmap="rainbow",
                             stat_min=0.3,
                             stat_max=0.9,
                             titles=[""],
                             background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
                             underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]),
                             z_slices=z_slices,
                             n_slices=len(z_slices),  
                             redo=redo)

figures.combine_plots(output_fname=os.path.join(output_fig, f"n{len(IDs_2runs)}_combined_icc_plots.png"),
                      map_files=[icc_plot],label_idx=False,
                      axial_files=[icc_axial_plot],
                      figsize=(2, 4), redo=redo)


print("", flush=True)
print(f'=== ICC between sliceShim run-01 and run-02  done', flush=True)
print("=========================================", flush=True)

# ----------  betwen shimBase and shimSlice ---
print("", flush=True)
print(f'=== ICC between sliceShim and sliceBase  start', flush=True)
print("=========================================", flush=True)
output_dir = os.path.join(second_level_dir.format("icc"), "shimBase_vs_shimSlice")
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

icc_maps,icc_maps_s = glm_ana.run_icc(IDs=IDs_2runs, i_fnames=i_fnames_by_runs, o_dir=output_dir, mask_file=mask, threshold=0)

print("", flush=True)
print(f'=== ICC between sliceShim and sliceBase  done', flush=True)
print("=========================================", flush=True)
