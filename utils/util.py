import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, f1_score, average_precision_score, matthews_corrcoef
from tqdm import tqdm
import pandas as pd
import re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def set_seed(seed):
    
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  

    
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


def flatten_group(group, env_columns):
    
    flat_data = {}
    for col in env_columns:
        for i, val in enumerate(group[col], 1):
            flat_data[f"{col}_{i}"] = val
    return pd.Series(flat_data)

def get_data(component, return_data = "noe", is_filt=False, score_str=None):


    
    if component == "QI":
        if is_filt:
            if score_str != None:
                components_path = f'/data/home/wxl22/changzu_ans/dataset/filt_comp/all_results_with_QI_filt_{score_str}.xlsx'
            else:
                components_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/all_results_with_QI_filt.xlsx'
        else:
            components_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/all_results_with_QI.xlsx'
        environment_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/97\u5730\u533a26\u73af\u5883\u5b63\u5ea6\u5747\u503c.xlsx'
    elif component == "NC":
        if is_filt:
            if score_str != None:
                components_path = f'/data/home/wxl22/changzu_ans/dataset/filt_comp/all_results_with_NC_filt_{score_str}.xlsx'
            else:
                components_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/all_results_with_NC_filt.xlsx'
        else:
            components_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/all_results_with_NC.xlsx'
        environment_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/97\u5730\u533a26\u73af\u5883\u5b63\u5ea6\u5747\u503c.xlsx'
    elif component == "BC":
        if is_filt:
            if score_str != None:
                components_path = f'/data/home/wxl22/changzu_ans/dataset/filt_comp/all_results_with_BC_filt_{score_str}.xlsx'
            else:
                components_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/all_results_with_BC_filt.xlsx'
        else:
            components_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/all_results_with_BC.xlsx'
        environment_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/97\u5730\u533a26\u73af\u5883\u5b63\u5ea6\u5747\u503c.xlsx'
    elif component == "pareto_value":
        if is_filt:
            components_path = '/data/home/wxl22/changzu_ans/dataset/filt_comp/all_results_with_pareto_value_filt.xlsx'
        else:
            components_path = '/data/home/wxl22/changzu_ans/dataset/processed/pareto_result.xlsx'
        environment_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/97\u5730\u533a26\u73af\u5883\u5b63\u5ea6\u5747\u503c.xlsx'
    df_components = pd.read_excel(components_path)
    df_environment = pd.read_excel(environment_path)


    
    def format_sample_id(sample):
        match = re.match(r'^(.*?)(?:-\d)?$', sample)
        if match:
            base_id = match.group(1)
            match_num = re.match(r'^([A-Za-z]+)(\d+)$', base_id)
            if match_num:
                prefix, number = match_num.groups()
                if len(number) == 1:
                    return f"{prefix.upper()}0{number}"
                return f"{prefix.upper()}{number}"
        return sample

    df_environment['Sample'] = df_environment['Sample'].apply(format_sample_id)
    df_components['Sample'] = df_components['Sample'].apply(format_sample_id)
    if is_filt:
        
        exclude_ids = {"MC01", "MC05", "MC22", "NC2406"}
        df_environment = df_environment[~df_environment['Sample'].isin(exclude_ids)].reset_index(drop=True)
        df_components  = df_components[~df_components['Sample'].isin(exclude_ids)].reset_index(drop=True)

    
    env_group_counts = df_environment['Sample'].value_counts()
    if not all(env_group_counts == 16):
        print("Warning: the following sample IDs do not have 16 rows:")
        print(env_group_counts[env_group_counts != 16])
    else:
        print("All sample IDs contain 16 rows.")

    
    comp_duplicates = df_components['Sample'].duplicated().sum()
    if comp_duplicates > 0:
        print(f"Warning: components.xlsx contains {comp_duplicates} duplicate sample IDs:")
        print(df_components[df_components['Sample'].duplicated(keep=False)])
        df_components = df_components.drop_duplicates(subset=['Sample'], keep='first')
        print("Duplicates removed; kept the first occurrence.")

    
    env_columns = [col for col in df_environment.columns if col not in ['Sample', 'LON', 'LAT']]
    environmentFactors = []
    sampleid_order_env = []

    for sampleid, group in df_environment.groupby('Sample'):
        
        lon = group['LON'].iloc[0]
        lat = group['LAT'].iloc[0]
        alt = group['ALT'].iloc[0]
        
        
        flat_data = [lon, lat, alt]  
        for col in env_columns:
            flat_data.extend(group[col].values)  
        
        environmentFactors.append(flat_data)
        sampleid_order_env.append(sampleid)
    print(len(environmentFactors))
    

    if component not in df_components.columns:
        raise ValueError(f"Column '{component}' does not exist in components.xlsx. Please verify the column name.")

    labels = []
    sampleid_order_comp = []  

    for sampleid in sampleid_order_env:  
        if sampleid in df_components['Sample'].values:
            label = df_components[df_components['Sample'] == sampleid][component].iloc[0]
            labels.append(label)
            sampleid_order_comp.append(sampleid)
        else:
            print(f"Warning: sample ID {sampleid} not found in components.xlsx; skipping.")
            environmentFactors.pop(sampleid_order_env.index(sampleid))  
            sampleid_order_env.remove(sampleid)

    
    print(f"environmentFactors length: {len(environmentFactors)}")
    print(f"labels length: {len(labels)}")
    if len(environmentFactors) != len(labels):
        raise ValueError("environmentFactors and labels have different lengths!")

    
    
    
    
    

    scaler = StandardScaler()
    environment_scaled = scaler.fit_transform(environmentFactors)
    if component in ["pareto_value"]:
        labels = np.array(labels)
    else:
        labels = np.log1p(np.array(labels))
    label_scaler = StandardScaler()
    y_scaled = label_scaler.fit_transform(labels.reshape(-1, 1)).squeeze()
    print("\nFirst 5 labels:")
    print(y_scaled[:5])
    if return_data == "all":
        return environment_scaled, y_scaled, label_scaler, sampleid_order_comp,scaler
    else:
        return environment_scaled, y_scaled, label_scaler, sampleid_order_comp

def get_china_data(data_scaler):
    from pathlib import Path
    
    environment_local_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/97\u5730\u533a26\u73af\u5883\u5b63\u5ea6\u5747\u503c.xlsx'
    environment_china_path = '/data/home/wxl22/changzu_ans/dataset/row/all_data.xlsx'

    df_local_environment = pd.read_excel(environment_local_path)
    df_china_environment = pd.read_excel(environment_china_path)
    print(df_china_environment.isna().any().any())
    df_china_environment.fillna(0, inplace=True)
    print(df_china_environment.isna().any().any())

    
    cols_to_drop = ["YEAR", "MM", "DD", "DOY", "YYYMMDD", "YYYYMMDD"]  

    
    df_china_environment = df_china_environment.drop(columns=cols_to_drop, errors="ignore")


    print(df_china_environment.head())
    
    Path("output").mkdir(exist_ok=True)
    txt = df_china_environment.head().to_string(index=False)
    with open("/data/home/wxl22/changzu_ans/dataset/row/df_china_environment_head.txt", "w", encoding="utf-8") as f:
        f.write(txt)

    
    local_columns_ordered = [col for col in df_local_environment.columns if col != 'Sample']
    
    df_china_environment = df_china_environment[local_columns_ordered]

    
    env_china_columns = [col for col in df_china_environment.columns if col not in ['LON', 'LAT', 'ALT']]
    env_china_columns = df_china_environment[env_china_columns]  

    n_group = 16  
    local_ll = []
    environmentFactors_china = []
    num_samples = len(env_china_columns) // n_group
    for i in range(num_samples):
        group = df_china_environment.iloc[i * n_group: (i + 1) * n_group]
        
        lon = group['LON'].iloc[0]
        lat = group['LAT'].iloc[0]
        alt = group['ALT'].iloc[0]
        
        flat_data = [lon, lat, alt]  
        local_ll.append([lon, lat])
        for col in env_china_columns:
            flat_data.extend(group[col].values)  
        environmentFactors_china.append(flat_data)



    print(environmentFactors_china[102])
    arr = np.array(environmentFactors_china, dtype=np.float64)
    
    print("NaN count in raw data:", np.isnan(arr).sum())
    print("Inf count in raw data:", np.isinf(arr).sum())
    nan_rows = np.any(np.isnan(arr), axis=1)
    inf_rows = np.any(np.isinf(arr), axis=1)
    print("Sample indices containing NaN:", np.where(nan_rows)[0])
    print("Sample indices containing Inf:", np.where(inf_rows)[0])


    environment_scaled = data_scaler.fit_transform(environmentFactors_china)
    print("Columns with zero standard deviation:", np.where(data_scaler.scale_ == 0)[0])
    print("Contains NaN after normalization:", np.isnan(environment_scaled).any())

    return environment_scaled, local_ll

def get_china_data_orin(data_scaler):
    from pathlib import Path
    
    environment_local_path = '/data/home/wxl22/changzu_ans/dataset/processed/new/97\u5730\u533a26\u73af\u5883\u5b63\u5ea6\u5747\u503c.xlsx'
    environment_china_path = '/data/home/wxl22/changzu_ans/dataset/row/mixed26.csv'

    df_local_environment = pd.read_excel(environment_local_path)
    df_china_environment = pd.read_csv(environment_china_path)
    print(df_china_environment.isna().any().any())
    df_china_environment.fillna(0, inplace=True)
    print(df_china_environment.isna().any().any())

    cols_to_drop = ["YEAR", "MM", "DD", "DOY"]  

    
    df_china_environment = df_china_environment.drop(columns=cols_to_drop, errors="ignore")

    df_china_environment = quarterly_mean_by_sample(df_china_environment)
    print(df_china_environment.shape)

    cols_to_drop = ["YEAR", "QUARTER"]  

    
    df_china_environment = df_china_environment.drop(columns=cols_to_drop, errors="ignore")

    


    print(df_china_environment.head())
    
    Path("output").mkdir(exist_ok=True)
    txt = df_china_environment.head().to_string(index=False)
    with open("/data/home/wxl22/changzu_ans/dataset/row/df_china_environment_head.txt", "w", encoding="utf-8") as f:
        f.write(txt)

    
    local_columns_ordered = [col for col in df_local_environment.columns if col != 'Sample']
    
    df_china_environment = df_china_environment[local_columns_ordered]

    
    env_china_columns = [col for col in df_china_environment.columns if col not in ['LON', 'LAT', 'ALT']]
    env_china_columns = df_china_environment[env_china_columns]  

    n_group = 16  
    local_ll = []
    alt_list = []
    environmentFactors_china = []
    num_samples = len(env_china_columns) // n_group
    for i in range(num_samples):
        group = df_china_environment.iloc[i * n_group: (i + 1) * n_group]
        
        lon = group['LON'].iloc[0]
        lat = group['LAT'].iloc[0]
        alt = group['ALT'].iloc[0]
        
        flat_data = [lon, lat, alt]  
        alt_list.append(alt)
        local_ll.append([lon, lat])
        for col in env_china_columns:
            flat_data.extend(group[col].values)  
        environmentFactors_china.append(flat_data)



    
    arr = np.array(environmentFactors_china, dtype=np.float64)
    
    print("NaN count in raw data:", np.isnan(arr).sum())
    print("Inf count in raw data:", np.isinf(arr).sum())
    nan_rows = np.any(np.isnan(arr), axis=1)
    inf_rows = np.any(np.isinf(arr), axis=1)
    print("Sample indices containing NaN:", np.where(nan_rows)[0])
    print("Sample indices containing Inf:", np.where(inf_rows)[0])


    environment_scaled = data_scaler.fit_transform(environmentFactors_china)
    print("Columns with zero standard deviation:", np.where(data_scaler.scale_ == 0)[0])
    print("Contains NaN after normalization:", np.isnan(environment_scaled).any())

    return environment_scaled, local_ll, alt_list


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(trueYAll, predYAll, metric=['r2', 'rmse', 'rpd', 'rer', 'mse', 'mae']):
    
    
    
    

    result = {}

    if 'r2' in metric:
        result['r2'] = r2_score(trueYAll, predYAll)

    if 'rmse' in metric:
        rmse = np.sqrt(mean_squared_error(trueYAll, predYAll))
        result['rmse'] = rmse

    if 'rpd' in metric:
        std_true = np.std(trueYAll)
        result['rpd'] = std_true / result.get('rmse', np.sqrt(mean_squared_error(trueYAll, predYAll)))

    if 'rer' in metric:
        mean_true = np.mean(trueYAll)
        result['rer'] = mean_true / result.get('rmse', np.sqrt(mean_squared_error(trueYAll, predYAll)))

    if 'mse' in metric:
        result['mse'] = mean_squared_error(trueYAll, predYAll)

    if 'mae' in metric:
        result['mae'] = mean_absolute_error(trueYAll, predYAll)

    return result

def validate_model(model, val_loader, criterion, device, args, label_scaler):
    model.eval()
    val_loss = 0
    trueYAll = []
    predYAll = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.float().to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            
            y_pred_np = outputs.detach().cpu().numpy().reshape(-1, 1)
            y_true_np = batch_y.detach().cpu().numpy().reshape(-1, 1)

            
            y_pred_orig = label_scaler.inverse_transform(y_pred_np).squeeze()
            y_true_orig = label_scaler.inverse_transform(y_true_np).squeeze()

            trueYAll.extend(y_true_orig.tolist())
            predYAll.extend(y_pred_orig.tolist())
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    eval_result = evaluate(trueYAll, predYAll, metric=['r2', 'rmse', 'rpd', 'rer', 'mae'])

    return avg_val_loss, eval_result























            









import logging
import os
import sys
from texttable import Texttable

class Logger:
    logger = None

    @staticmethod
    def get_logger(filename: str = None):
        if not Logger.logger:
            Logger.init_logger(filename=filename)
        return Logger.logger

    @staticmethod
    def init_logger(
            level=logging.INFO,
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: \n %(message)s',
            filename: str = None):
        logger = logging.getLogger(filename)
        logger.setLevel(level)
        fmt = logging.Formatter(fmt)
        
        if os.path.exists(filename):
            os.remove(filename)

        
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.setLevel(level)
        Logger.logger = logger
        return logger


def args_print(args, logger):
    print('\n')
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())


def deep_compute_gradcam(model, batch_data,input_length=256):
    
    model.zero_grad()
    output = model(batch_data, return_data="pred")
    

    if output.ndim == 0:
        output.backward()
    else:
        output.sum().backward()

    
    gradients = model.cnn_grads.detach()           
    activations = model.cnn_activations.detach()   

    weights = gradients.mean(dim=2)                
    cam = torch.sum(weights.unsqueeze(-1) * activations, dim=1)  

    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)

    
    if cam.shape[1] < input_length:
        cam = F.interpolate(cam.unsqueeze(1), size=input_length, mode='linear', align_corners=False).squeeze(1)

    return cam.cpu().numpy()  






















def plot_gradcam_on_spectrum(env, cam, name, num_id=0, where="train", wavelengths=None):
    import matplotlib.pyplot as plt
    import numpy as np

    env_id = [
        "ALLSKY_SFC_SW_DWN",
        "WS2M",
        "T2M_RANGE",
        "T2M_MAX",
        "T2M_MIN",
        "T2M",
        "RH2M",
        "PS",
        "PRECTOTCORR",
        "TOA_SW_DWN",
        "WS2M_MAX",
        "WS2M_MIN",
        "WS2M_RANGE",
        "WD2M",
        "CLRSKY_SFC_SW_DWN",
        "ALLSKY_SFC_LW_DWN",
        "T2MDEW",
        "T2MWET",
        "TS",
        "GWETTOP",
        "GWETROOT",
        "GWETPROF",
        "DL",
        "ALT"
    ]

    

    env = np.squeeze(env)  
    cam = np.squeeze(cam)
    L = len(env)
    x = wavelengths if wavelengths is not None else np.arange(L)
    
    env = env.cpu().numpy() if isinstance(env, torch.Tensor) else env
    cam = cam.cpu().numpy() if isinstance(cam, torch.Tensor) else cam
    jet = plt.cm.get_cmap('jet', 256)
    new_colors = jet(np.linspace(0, 1, 256))
    new_colors[0] = np.array([1, 1, 1, 1])  
    white_jet = mcolors.ListedColormap(new_colors)
    
    

    
    
    
    
    plt.figure(figsize=(14,4))
    plt.plot(x, env, color='black')
    plt.imshow(cam[np.newaxis, :], extent=[x[0], x[-1], env.min(), env.max()],
            cmap=white_jet, alpha=0.5, aspect='auto', vmax=1, vmin=0)
    plt.colorbar(label='Grad-cam Intensity', pad=0.01)
    
    for i in range(2, L, 16):
        
        plt.axvline(x=x[i], color='blue', linestyle='--', linewidth=0.5)
        plt.text(x[i]+10, plt.ylim()[0], env_id[int(i/16)], rotation=90, fontsize=8, ha='right', va='top')

    
    
    
    plt.xlim(0, L)
    plt.xticks([])  
    plt.ylim(env.min(), env.max())
    
    plt.grid(False)
    plt.tight_layout()

    
    save_path = f"/data/home/wxl22/changzu_ans/images/{name}_{where}_gradcam_sample{num_id}.png"
    plt.savefig(save_path, dpi=600)
    plt.close()



def plot_gradcam_all(env, cam, name, czID, num_id=0, where="train", wavelengths=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec

    env_id = [
        "ALLSKY_SFC_SW_DWN",
        "WS2M",
        "T2M_RANGE",
        "T2M_MAX",
        "T2M_MIN",
        "T2M",
        "RH2M",
        "PS",
        "PRECTOTCORR",
        "TOA_SW_DWN",
        "WS2M_MAX",
        "WS2M_MIN",
        "WS2M_RANGE",
        "WD2M",
        "CLRSKY_SFC_SW_DWN",
        "ALLSKY_SFC_LW_DWN",
        "T2MDEW",
        "T2MWET",
        "TS",
        "GWETTOP",
        "GWETROOT",
        "GWETPROF",
        "DL",
        "ALT"
    ]

    

    env = np.squeeze(env)  
    column_sums = np.sum(cam, axis=0)
    column_sums = column_sums / len(cam)
    
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)
    cam = np.squeeze(cam)
    L = 387
    x = wavelengths if wavelengths is not None else np.arange(L)


    ax1 = plt.subplot(gs[0])
    ymin, ymax = ax1.get_ylim()
    text_y = 0.5 * ymax + 0.01 * (ymax - ymin)  
    ax1.vlines(x, ymin=0, ymax=column_sums, color='#274F7C', linewidth=(x[-1] - x[0])/ len(column_sums))
    for i in range(2, L, 16):
        ax1.axvline(x=x[i]+0.5, color='red', linestyle='--', linewidth=0.5)
        ax1.text(x[i]+8, text_y, env_id[int(i/16)],
                rotation=90, fontsize=8, ha='center', va='bottom', color='blue')


    
    ax1.set_ylim([0, 0.4])
    ax1.set_xlim([x[0], x[-1]])  
    ax1.set_xticks([])  
    ax1.set_ylabel("Sum")

    
    ax2 = plt.subplot(gs[1])
    
    
    cam = cam.cpu().numpy() if isinstance(cam, torch.Tensor) else cam
    jet = plt.cm.get_cmap('jet', 256)
    new_colors = jet(np.linspace(0, 1, 256))
    new_colors[0] = np.array([1, 1, 1, 1])  
    white_jet = mcolors.ListedColormap(new_colors)

    im = ax2.imshow(cam, cmap='jet', aspect='auto', extent=[x[0], x[-1], 0, cam.shape[0]], vmin=0, vmax=0.7, origin='lower')


    
    group_prefixes = [cz[:2] for cz in czID]

    
    group_prefixes = ['NC' if p == 'MC' else p for p in group_prefixes]

    
    group_change_rows = [0]  
    prev_prefix = group_prefixes[0]

    for i in range(1, len(group_prefixes)):
        if group_prefixes[i] != prev_prefix:
            group_change_rows.append(i)
            prev_prefix = group_prefixes[i]

    
    for row_idx in group_change_rows:
        ax2.text(x[0] - 25, row_idx, group_prefixes[row_idx],
                va='center', ha='right', fontsize=8, color='black')




    cbar_ax = fig.add_axes([0.92, 0.115, 0.02, 0.595])  
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Grad-CAM Intensity")

    
    

    
    
    
    



    ax2.set_xticks([])

    save_path = f"/data/home/wxl22/changzu_ans/images/{name}_{where}_gradcam_sample{num_id}.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.4)
    plt.close()



def plot_gradcam_Line(env, cam, name, czID, num_id=0, where="train", wavelengths=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    print(czID)

    
    env_id = [
        "ALLSKY_SFC_SW_DWN",
        "WS2M",
        "T2M_RANGE",
        "T2M_MAX",
        "T2M_MIN",
        "T2M",
        "RH2M",
        "PS",
        "PRECTOTCORR",
        "TOA_SW_DWN",
        "WS2M_MAX",
        "WS2M_MIN",
        "WS2M_RANGE",
        "WD2M",
        "CLRSKY_SFC_SW_DWN",
        "ALLSKY_SFC_LW_DWN",
        "T2MDEW",
        "T2MWET",
        "TS",
        "GWETTOP",
        "GWETROOT",
        "GWETPROF",
        "DL",
        "ALT"
    ]

    cam = np.squeeze(cam)  
    L = cam.shape[1]
    group_size = 16
    num_groups = (L - 2) // group_size + 2  
    segment_means = []
    segment_stds = []
    segment_labels = []

    for idx in range(num_groups):
        start_idx = 2 + (idx - 2) * group_size
        end_idx = min(2 + (idx - 1) * group_size, L)
        segment = cam[:, start_idx:end_idx]  
        if idx == 0:
            segment_means.append(cam[:, 0:1].sum(axis=1).mean())
            segment_stds.append(cam[:, 0:1].sum(axis=1).std())
        elif idx == 1:
            segment_means.append(cam[:, 1:2].sum(axis=1).mean())
            segment_stds.append(cam[:, 1:2].sum(axis=1).std())
        else:
            segment_means.append(segment.sum(axis=1).mean() / (end_idx - start_idx))
            segment_stds.append((segment.sum(axis=1) / (end_idx - start_idx)).std())

        
        if idx == 0:
            segment_labels.append("LON")
        elif idx == 1:
            segment_labels.append("LAT")
        else:
            segment_labels.append(env_id[idx - 2])
    print(segment_labels)
    print(segment_means)
    print(segment_stds)

    
    df = pd.DataFrame({
        "segment": segment_labels,
        "mean": segment_means,
        "std": segment_stds
    })

    
    csv_save_path = f"/data/home/wxl22/changzu_ans/images/bar_gradcam_{name}_{where}_sample{num_id}.csv"
    df.to_csv(csv_save_path, index=False)
    
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])

    bar_positions = np.arange(len(segment_means))
    ax.bar(bar_positions, segment_means, yerr=segment_stds, color='#274F7C', alpha=0.8)

    
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(segment_labels, rotation=90, fontsize=8)
    ax.set_ylim([0, 0.6])
    ax.set_ylabel("Average Grad-CAM Importance")
    ax.set_title(f"Grad-CAM Feature Importance - {where} Sample {num_id}")

    plt.tight_layout()
    save_path = f"/data/home/wxl22/changzu_ans/images/new_bar_gradcam_{name}_{where}_sample{num_id}.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.4)
    plt.close()

def plot_gradcam_grouped(env, cam, czID, name, where="train", wavelengths=None, real_split="yes"):
    import numpy as np
    print("group")

    if hasattr(env, "cpu"):
        env = env.cpu().numpy()
    if hasattr(cam, "cpu"):
        cam = cam.cpu().numpy()

    env = np.squeeze(env)
    cam = np.squeeze(cam)

    
    if hasattr(czID, "cpu"):
        czID = czID.cpu().tolist()
    elif isinstance(czID, np.ndarray):
        czID = czID.tolist()
    elif isinstance(czID, str):
        czID = [czID]

    czID_np = np.array(czID)  

    
    bc_mask = np.array([id.startswith("BC") for id in czID])
    ncmc_mask = np.array([id.startswith("NC") or id.startswith("MC") for id in czID])
    if real_split == "yes":
        if bc_mask.any():
            plot_gradcam_Line(None, cam[bc_mask], name + "_BC", czID_np[bc_mask], num_id=0, where=where, wavelengths=wavelengths)

        if ncmc_mask.any():
            plot_gradcam_Line(None, cam[ncmc_mask], name + "_NC", czID_np[ncmc_mask], num_id=0, where=where, wavelengths=wavelengths)
    else:
        plot_gradcam_Line(None, cam, name + "_ALL", czID_np, num_id=0, where=where, wavelengths=wavelengths)




import pandas as pd
from typing import Optional, Tuple, List

def quarterly_mean_by_sample(
    df: pd.DataFrame,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    date_col: str = "YYYYMMDD",
    keep_quarter_label: bool = True,
) -> pd.DataFrame:
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    work = df.copy()

    
    def _find_col(cands: List[str]) -> Optional[str]:
        for c in work.columns:
            if c.lower() in cands:
                return c
        return None

    if lat_col is None:
        lat_col = _find_col(["lat", "latitude", "y", "LAT"])
    if lon_col is None:
        lon_col = _find_col(["lon", "lng", "longitude", "x", "LON"])
    if lat_col is None or lon_col is None:
        raise ValueError("Latitude/longitude columns not found; specify lat_col/lon_col or ensure columns are named lat/lon/latitude/longitude.")

    
    if date_col not in work.columns:
        
        for c in ["date", "Date", "DATE", "datetime", "time", "timestamp", "YYYYMMDD"]:
            if c in work.columns:
                date_col = c
                break
    if date_col not in work.columns:
        raise ValueError(f"Date column '{date_col}' not found and no common date column detected.")

    
    s = work[date_col]
    
    dt = None
    try:
        as_str = s.astype(str).str.replace(r"\D", "", regex=True)  
        mask_8 = as_str.str.len() == 8
        dt_try = pd.to_datetime(as_str.where(mask_8, None), format="%Y%m%d", errors="coerce")
        
        if dt_try.notna().mean() >= 0.6:
            dt = dt_try
    except Exception:
        dt = None
    
    if dt is None or dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        raise ValueError(f"Failed to parse date column '{date_col}'; check the format.")

    work["_YEAR"] = dt.dt.year
    work["_QUARTER"] = ((dt.dt.month - 1) // 3 + 1).astype("Int64")  

    
    num_cols = work.select_dtypes(include="number").columns.tolist()
    
    for col in [lat_col, lon_col, date_col, "_YEAR", "_QUARTER"]:
        if col in num_cols:
            num_cols.remove(col)
    if not num_cols:
        raise ValueError("No numeric columns available for averaging.")

    
    agg = (
        work.groupby([lat_col, lon_col, "_YEAR", "_QUARTER"], dropna=True)[num_cols]
            .mean()
            .reset_index()
    )

    
    
    years_by_sample = (
        work.dropna(subset=[lat_col, lon_col, "_YEAR"])
            .groupby([lat_col, lon_col])["_YEAR"]
            .unique()
    )

    
    full_index_tuples: List[Tuple] = []
    for (lat, lon), years in years_by_sample.items():
        for y in sorted(set(int(v) for v in years if pd.notna(v))):
            for q in (1, 2, 3, 4):
                full_index_tuples.append((lat, lon, y, q))
    full_index = pd.MultiIndex.from_tuples(full_index_tuples, names=[lat_col, lon_col, "_YEAR", "_QUARTER"])

    agg = (
        agg.set_index([lat_col, lon_col, "_YEAR", "_QUARTER"])
           .reindex(full_index)         
           .reset_index()
    )

    
    agg = agg.rename(columns={"_YEAR": "YEAR", "_QUARTER": "QUARTER"})
    if keep_quarter_label:
        agg["QUARTER"] = "Q" + agg["QUARTER"].astype("Int64").astype(str)

    
    agg = agg.sort_values([lat_col, lon_col, "YEAR", "QUARTER"]).reset_index(drop=True)
    return agg


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

def visualize_distribution(predictions, bins=50, show_kde=True, save_path=None):
    
    arr = np.asarray(predictions).ravel()
    arr = arr[~np.isnan(arr)]
    
    plt.figure(figsize=(8, 5), dpi=150)
    sns.histplot(arr, bins=bins, kde=show_kde, color="steelblue")
    plt.xlabel("Prediction value")
    plt.ylabel("Count")
    plt.title("Distribution of prediction values")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
