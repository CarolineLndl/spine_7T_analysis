# Main imports ------------------------------------------------------------
import os, gzip, copy
import numpy as np
import nibabel as nib
import json, glob
from pathlib import Path
from datetime import datetime
import warnings

# Nilearn imports ----------------------------------------------------------
from nilearn import image
from nilearn.image import smooth_img
from nilearn.maskers import NiftiMasker


def tmean_img(ID=None, i_img=None, o_img=None, redo=False, verbose=False):
        """
        This function will help to calculate mean images across volumes (tmean).
        use fslmaths
        
        Attributes:
        ----------
        ID: name of the participant
        i_img: input filename of functional images (str, default:None, an error will be raise), 4D image
        o_img: output folder name filename (str, default:None, the input filename will be used as a base)
        
        Outputs: 
        ----------
        Mean image inputfile_tmean.nii.gz
        """
        if ID is None:
            raise Warning("Please provide the ID of the participant, ex: _.stc(ID='A001')")
        
        if i_img is None:
            raise Warning("Please provide filename of the input file")
      
        # Select the default output directory (input directory) 
        if o_img is None:
            o_img = i_img.split(".")[0] + "_tmean.nii.gz"

        # calculate the tmean:
        if not os.path.exists(o_img) or redo:
            string = f"fslmaths {i_img} -Tmean {o_img}"
            os.system(string)  # run the string as a command line
            
        if verbose:
            print("Done : check the outputs files in fsleyes by copy and past:")
            print("fsleyes " + o_img)
            
        return o_img


def group_mean_img(IDs=None, i_dir=None, o_dir=None, prefix_tag='', suffix_tag="", tag='', remove_4d=True, redo=False, verbose=False):
        """
        This function will help to calculate mean images across volumes (tmean).
        use fslmaths
        
        Attributes:
        ----------
        IDs: name of the participants
        i_img: input filename tag of functional images (str, default:None, an error will be raise), 4D image
        o_dir: output folder name filename (str, default:None, the input filename will be used as a base)
        
        Outputs: 
        ----------
        Mean image inputfile_tmean.nii.gz
        """

        if IDs is None:
            raise Warning("Please provide the IDs of the participants, ex: _.stc(ID=['A001','A002'])")
        
        if i_dir is None:
            raise Warning("Please provide directory of the input file")
      
        # Select the default output directory (input directory) 
        if o_dir is None:
            o_dir=os.path.dirname(i_dir)
        
        if not os.path.exists(o_dir):
            os.makedirs(o_dir)

        o_img = o_dir + "n_" + str(len(IDs))+"_"+tag
        o_img_mean = o_img + "_mean.nii.gz"
        o_img_std = o_img + "_std.nii.gz"
        o_img_z = o_img + "_z.nii.gz"

        ######## Merge indive files:
        file_4d=o_img_mean.split("mean")[0] +"4d.nii.gz"

        # Loop through each participant ID and construct the file path
        input_files = []
        for ID_nb, ID in enumerate(IDs):
            indiv_dir = i_dir[ID_nb] if isinstance(i_dir, list) else i_dir
            
            file_name = f'{prefix_tag}{ID}{suffix_tag}.nii.gz'  # Replace with your actual file naming convention
            print(os.path.join(indiv_dir, file_name))
            file_path = glob.glob(os.path.join(indiv_dir, file_name))[0]
            
            if os.path.isfile(file_path):
                input_files.append(file_path)
            else:
                print(f"File not found: {file_path}")
            
        ###### calculate the tmean:
        if not os.path.exists(o_img_mean) or redo==True:
            os.system(f"fslmerge -t {file_4d} " + ' '.join(input_files)) # Concatenate the input files using fslmerge

            print("Creating the mean image")
            os.system(f"fslmaths {file_4d} -Tmean {o_img_mean}")
            os.system(f"fslmaths {file_4d} -Tstd {o_img_std}")
            os.system(f"fslmaths {o_img_mean} -div {o_img_std} {o_img_z}")


            if remove_4d==True:
                os.remove(file_4d)
            
        if verbose ==True:
            print("Done : check the outputs files in fsleyes by copy and past:")
            print("fsleyes " + o_img_mean)
        
            
        return o_img_mean


def unzip_file(i_file, o_folder=None, ext=".nii", zip_file=False, redo=False, verbose=False):
        """
        unzip the file to match with SPM
        Attributes
        ----------
        i_file <filename>: input file
        o_img: output folder name filename (str, default:None, the input filename will be used as a base)
        ext <str>: extension after unzip default: ".nii", put ".nii.gz" to zip a file
        zip_file <Bolean>: zip the file instead of unzip a file (default: False)
        redo <Bolean>: to rerun the analysis put True (default: False)
        
        return
        ----------
        o_file: <filename>: file name of unziped or zipped files 
        """
        if o_folder is not None:
            output_file = os.path.join(o_folder, os.path.basename(i_file).split('.')[0] + ext)
            
        else:
            output_file = i_file.split('.')[0] + ext
            
        # Zip file
        if zip_file:
            if not os.path.exists(i_file.split('.')[0] + ext) or redo:
                if verbose:
                    print("Unzip is running")
                string= 'gzip ' + i_file
                os.system(string)
                if o_folder:
                    os.rename(i_file.split('.')[0] + ext, output_file)
            else:
                if verbose:
                    print("Zip was already done please put redo=True to redo that step")
                else:
                    pass
                 
        else:
            if not os.path.exists(i_file.split('.')[0] + ext) or redo:
            
                input = gzip.GzipFile(i_file, 'rb') # load the  .nii.gz
                s = input.read(); input.close()
                unzip = open(i_file.split('.')[0] + ext, 'wb') # save the .nii
                unzip.write(s); unzip.close()
                os.rename(i_file.split('.')[0] + ext, output_file)
                
                if verbose:
                    print('Unzip done for: ' + os.path.basename(i_file))
                else:
                    pass
                
            else :
                if verbose:
                    print("Unzip was already done please put redo=True to redo that step")
                else:
                    pass
        
            
        return output_file


def standardize(i_img=None,o_folder=None,json_files=None,mask_img=None,tag="",redo=False,verbose=False):
        """
        unzip the file to match with SPM
        Attributes
        ----------
        i_img <filename>, mendatory, default: None: input filename
        o_folder <dirname> optional, default None : output directory (e.g: output_file='/mydir/')
        json_file <str>:
        mask_img <filename> optional, default None, If provided, signal is only standardized from voxels inside the mask.
        redo <Bolean>: to rerun the analysis put True (default: False)
        """
        
        if (i_img is None):
            raise ValueError("Please provide the input filename, ex: _.cleam_images(i_img='/mydir/sub-1_filename.nii.gz')")
     
        timeseries=nib.load(i_img).get_fdata() # extract Time series dats
        signals= timeseries.reshape(-1, timeseries.shape[-1]).T # reshape timeseries (nb_volumes, nb_voxels)

        if signals.shape[0] == 1:
            warnings.warn('Standardization of 3D signal has been requested but '
                              'would lead to zero values. Skipping.')
        else:
            signals= timeseries.reshape(-1, timeseries.shape[-1]).T # reshape timeseries (nb_volumes, nb_voxels)
            std = signals.std(axis=0)
            std[std < np.finfo(np.float64).eps] = 1.  # avoid numerical problems
            signals /= std

        # save into filename
        o_filename = i_img.split('.')[0] + tag + ".nii.gz"
        json_file = o_filename.split('.')[0] + ".json"
        if not os.path.exists(o_filename) or redo==True:
            o_img = image.new_img_like(i_img, signals.T.reshape(timeseries.shape),copy_header=True)
            o_img.to_filename(o_filename) #save image

            if mask_img:
                string = f"fslmaths {o_filename} -mas {mask_img} o_filename"
                os.system(string)

            infos={"standardize":True,"mask":mask_img}
            with open(json_file, 'w') as f:
                json.dump(infos, f) # save info
                    

def compute_tsnr_map(fname_file, ofolder, redo, first_n_vols=None, smooth=False):
    """
    Attributes:
    ----------
    fname_file: Filename of the input 4D NIfTI file to compute tSNR from
    ofolder: Output folder name
    redo: Overwrite existing tSNR file if True
    first_n_vols: If specified, only use the first n volumes to compute tSNR

    Returns:
        str: Filename of the tSNR NIfTI file
    """
    if not os.path.exists(fname_file):
        raise FileNotFoundError(f"Input file not found: {fname_file}")

    fname_tsnr = os.path.join(ofolder, os.path.basename(fname_file).split(".")[0] + "_tsnr.nii.gz")
    # compute tSNR *******************************************************************************
    if not os.path.exists(fname_tsnr) or redo:
        if not os.path.exists(os.path.dirname(fname_tsnr)):
            os.makedirs(os.path.dirname(fname_tsnr))
        nii = nib.load(fname_file)
        if first_n_vols is None:
            first_n_vols = nii.shape[3]
        if first_n_vols > nii.shape[3]:
            raise ValueError(f"first_n_vols ({first_n_vols}) is greater than the number of volumes in the file ({nii.shape[3]})")
        data = nii.get_fdata()[:, :, :, :first_n_vols]
        tsnr = np.mean(data, axis=3) / np.std(data, axis=3)
        nii_tsnr = nib.Nifti1Image(tsnr, affine=nii.affine, header=nii.header)

        if smooth:
            nii_tsnr = smooth_img(nii_tsnr, fwhm=[3, 3, 6])

        nii_tsnr.to_filename(fname_tsnr)

    return fname_tsnr


def extract_mean_within_mask(fname_file, fname_mask):
    """
    Attributes:
    ----------
    fname_file: Filename of the input 4D NIfTI file to compute tSNR from
    fname_mask: Filename of the input 3D NIfTI file mask to compute tSNR metric within

    Returns:
        float: Mean value within the mask
    """
    # select the mask
    masker_stc = NiftiMasker(mask_img=fname_mask, smoothing_fwhm=None, standardize=False, detrend=False)
    # mask the image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        file_masked = masker_stc.fit_transform(fname_file)
    # calculate the mean value
    mean_value = np.mean(file_masked)

    return mean_value


def tSNR(ID=None,i_img=None,o_dir=None,mask=None,warp_img=None,structure='spinalcord',redo=False):
    """
        This function calculate the tSNR within the brain or spinal cord
        
        Attributes:
        ----------
        config: load config file
        ID: participant ID
        i_img: 4d func image
        inTemplate: put True to coregister the tSNR map into template space
        redo: put True to re-run the analysis on existing file (default=False)
    
    """

    img_tSNR = compute_tsnr_map(i_img, o_dir, redo)

    # extract value inside the mask
    o_txt = os.path.join(o_dir, f"sub-{ID}", os.path.basename(i_img).split(".")[0] + "mean.txt")
    if os.path.exists(o_txt):
        if redo:
            os.remove(o_txt) 
    if not os.path.exists(o_txt):
        mean_tSNR_masked = extract_mean_within_mask(img_tSNR, mask)
            
        with open(o_txt, 'a') as f:  # 'a' mode for appending to the file
            f.write(f"{mean_tSNR_masked}\n")  # Write the

    return o_txt, img_tSNR


def compute_SNR(i_file, mask_file):
    """
        This function calculate the SNR within the brain or spinal cord
        
        Attributes:
        ----------
        config: load config file
        ID: participant ID
        i_file: 3d mean func image filename
        mask_file: 3d mask image filename
    
    """
    if i_file is None:
        raise ValueError("Please provide the input filename, (i_file='/mydir/sub-1_mean.nii.gz')")
    
    if mask_file is None:
        raise ValueError("Please provide the mask filename, (mask_file='/mydir/sub-1_mask.nii.gz')")

    # Load mean motion corrected image
    i_img = nib.load(i_file)
    i_data = i_img.get_fdata()

    # Load spinal cord mask
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata().astype(bool)

    # Apply mask
    signal_in_mask = i_data[mask_data]

    # Compute sSNR
    mean_signal = np.mean(signal_in_mask)
    std_signal = np.std(signal_in_mask)

    sSNR = mean_signal / std_signal

    return sSNR


def get_latest_dir(base_dir):
    """
    Returns the path to the latest 'date' folder inside:
    qc_dir/sub-ID/func/ses_name/task_name/run_name/sct_get_centerline/
    """
    
    # Gather candidate folders
    base_dir = Path(base_dir)
    date_folders = [d for d in base_dir.iterdir() if d.is_dir() and "_" in d.name]

    # Parse folder names as datetimes
    def parse_date(name):
        try:
            return datetime.strptime(name, "%Y_%m_%d_%H%M%S.%f")
        except ValueError:
            return None

    dated = [(d, parse_date(d.name)) for d in date_folders]
    dated = [x for x in dated if x[1] is not None]

    if not dated:
        raise ValueError(f"No valid date folders found in {base_dir}")

    # Pick the latest one
    latest_folder = max(dated, key=lambda x: x[1])[0]
    return str(latest_folder)


def print_participant_metrics(participants_tsv, IDs):
    df_filtered = copy.deepcopy(participants_tsv)
    for ID in participants_tsv['participant_id']:
        if ID not in IDs:
            df_filtered = df_filtered[df_filtered['participant_id'] != ID]
    print("=== Participant metrics ===", flush=True)
    print(f"Age: {df_filtered['age'].mean()} ± {df_filtered['age'].std()}, [{df_filtered['age'].min()}, {df_filtered['age'].max()}]", flush=True)
    print(f"Females: {df_filtered['sex'].value_counts().get('F', 0)}, Males: {df_filtered['sex'].value_counts().get('M', 0)}", flush=True)


def extract_params(fname_nii):
    fname_json = fname_nii.replace(".nii.gz", ".json")
    if not os.path.exists(fname_nii):
        raise FileNotFoundError(f"NIfTI file not found for {fname_nii}")
    if not os.path.exists(fname_json):
        raise FileNotFoundError(f"JSON file not found for {fname_json}")

    params = {}

    with open(fname_json, 'r') as json_file:
        json_data = json.load(json_file)

    params['RepetitionTime'] = json_data.get('RepetitionTime', None)
    params['EchoTime'] = json_data.get('EchoTime', None)
    params['FlipAngle'] = json_data.get('FlipAngle', None)

    nii = nib.load(fname_nii)
    if nii.ndim >= 4:
        params['NumberOfVolumes'] = nii.shape[3]

    return params
