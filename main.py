from utils.para_config import init_args
from utils.util import set_seed, get_data, get_china_data, get_china_data_orin
import os
import pandas as pd
import json
import torch
import copy
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.model_selection import train_test_split
from models.base_models import OneDCNN, LSTM, GRU
from torch.utils.data import DataLoader
from utils.CZDataset import CZDataset
import torch.optim as optim
import torch.nn as nn
from utils.util import validate_model, Logger, args_print, plot_gradcam_on_spectrum, deep_compute_gradcam, plot_gradcam_all, plot_gradcam_Line, plot_gradcam_grouped
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
from torch_geometric.loader import DataLoader as pygDataLoader
from torch_geometric.data import Batch
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.drower import draw_suitability_choropleth_from_points,draw_suitability_map_china,get_equal_bins,get_quantile_bins,get_natural_breaks_bins,compute_zones
import pandas as pd
import xarray as xr
import time
from utils.drawer_china import draw_suitability_interpolation


def irm_penalty(per_env_losses, device):
    
    scale = torch.tensor(1.0, device=device, requires_grad=True)
    penalty = 0.0
    for l in per_env_losses:
        
        g = torch.autograd.grad(l * scale, [scale], create_graph=True)[0]
        penalty = penalty + g.pow(2)
    return penalty

def split_into_envs(batch_X, batch_y, batch_env=None):
    
    if batch_env is None:
        n = batch_X.size(0)
        idx = torch.randperm(n, device=batch_X.device)
        mid = n // 2
        e0 = idx[:mid]
        e1 = idx[mid:]
        return [(batch_X[e0], batch_y[e0]), (batch_X[e1], batch_y[e1])]
    else:
        envs = []
        for e in torch.unique(batch_env):
            m = (batch_env == e)
            if m.any():
                envs.append((batch_X[m], batch_y[m]))
        return envs


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = init_args()
    args.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f'{args.component}'
    exp_dir = os.path.join(args.log_dir, experiment_name+args.current_time)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/log.log')
    args_print(args, logger)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    all_seed_info = {
        "train": {
            "r2": [],
            "rmse": [],
            "rpd": [],
            "rer": [],
            "mse": [],
            "mae": []
        },
        "test": {
            "r2": [],
            "rmse": [],
            "rpd": [],
            "rer": [],
            "mse": [],
            "mae": []
        },
        "ood": {
            "r2": [],
            "rmse": [],
            "rpd": [],
            "rer": [],
            "mse": [],
            "mae": []
        }
    }

    set_seed(args.seed)
    environments, components, main_scaler, sampleID = get_data(args.component, is_filt=args.is_filt)
    
    X = environments.values if isinstance(environments, pd.DataFrame) else environments
    y = components.values if isinstance(components, pd.Series) or isinstance(components, pd.DataFrame) else components

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, shuffle=True)

    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train).to(device)
    y_test = torch.tensor(y_test).to(device)

    
    train_dataset = CZDataset(X_train, y_train)
    test_dataset = CZDataset(X_test, y_test)

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)



    
    mse_scores = []
    r2_scores = []
    mae_scores = []
    if args.goal == "train":
        
        for i in [1]:
            
            if args.model.lower() == "1dcnn":
                model = OneDCNN(input_dim=environments.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()

            best_train_metric = float('inf')  
            corresponding_test_perf = None
            best_model_state = None
            
            irm_lambda = getattr(args, "irm_lambda", 0.0)  

            for epoch in range(args.epochs):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for batch in train_loader:
                    
                    if len(batch) == 3:
                        batch_X, batch_y, batch_env = batch
                        batch_env = batch_env.to(device)  
                    else:
                        batch_X, batch_y = batch
                        batch_env = None

                    batch_X = batch_X.to(device)
                    batch_y = batch_y.float().to(device)

                    optimizer.zero_grad()

                    
                    env_slices = split_into_envs(batch_X, batch_y, batch_env)

                    per_env_losses = []
                    per_env_sizes = []

                    
                    for X_e, y_e in env_slices:
                        outputs_e = model(X_e)
                        loss_e = criterion(outputs_e, y_e)  
                        per_env_losses.append(loss_e)
                        per_env_sizes.append(X_e.size(0))

                    
                    emp_risk = torch.stack(per_env_losses).mean()

                    
                    penalty = irm_penalty(per_env_losses, device=batch_X.device)

                    
                    loss = emp_risk + irm_lambda * penalty

                    loss.backward()
                    optimizer.step()

                    
                    total_loss += (emp_risk.detach().item() * batch_X.size(0))
                    total_samples += batch_X.size(0)

                avg_loss = total_loss / max(total_samples, 1)
                
                

                
                train_loss, train_perf = validate_model(model, train_loader, criterion, device, args, main_scaler)
                test_loss, test_perf = validate_model(model, test_loader, criterion, device, args, main_scaler)
                print(f"  Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, test Loss: {test_loss}, train: {train_perf}, test: {test_perf}")
                
                if test_perf['rmse'] < best_train_metric:
                    best_train_metric = test_perf['rmse']
                    corresponding_test_perf = test_perf.copy()
                    best_model_state = copy.deepcopy(model.state_dict())

            
            mse_scores.append(corresponding_test_perf['rmse'])
            r2_scores.append(corresponding_test_perf['r2'])
            mae_scores.append(corresponding_test_perf['mae'])

            print(f"{corresponding_test_perf}")
            if args.save_model:
                if args.is_filt:
                    model_save_path = f"./changzu_ans/checkpoints/single_{args.component.lower()}_{args.seed}_best_model_filt.pth"
                else:
                    model_save_path = f"./changzu_ans/checkpoints/single_{args.component.lower()}_{args.seed}_best_model.pth"
                torch.save(best_model_state, model_save_path)
            
        print(f"Mean RMSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        print(f"Mean R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        print(f"Mean MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
        print(f"best R² in {np.argmax(r2_scores)+1} fold is {r2_scores[np.argmax(r2_scores)]}")
    elif args.goal == "vis":


        X_train = torch.tensor(X, dtype=torch.float32).to(device)
        train_dataset = CZDataset(X, y, sampleID)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = OneDCNN(input_dim=environments.shape[1]).to(device)
        if args.is_filt:
            model.load_state_dict(torch.load(f"./changzu_ans/checkpoints/single_{args.component.lower()}_{args.seed}_best_model_filt.pth"))
        else:
            model.load_state_dict(torch.load(f"./changzu_ans/checkpoints/single_{args.component.lower()}_{args.seed}_best_model.pth"))

        model.eval()
        czID_all = []
        cam_all = []

        if args.pareto == True:
            for num_id in range(len(train_dataset)):
                
                where = "train"
                
                env, lab, czID = train_dataset[num_id]
                env = torch.tensor(env, dtype=torch.float32).to(device)
                env = env.unsqueeze(0).to(torch.float32).to(device)

                
                cam = deep_compute_gradcam(model, env, input_length=386)
                cam_all.append(cam[0])
                czID_all.append(czID)
                
                
            if args.hotMap == 'yes':
                plot_gradcam_all(env=None, cam = cam_all, name=args.component, num_id=num_id, where=where, czID=czID_all)
                print(len(czID_all))
            else:
                
                plot_gradcam_grouped(env, cam_all, czID_all, name=args.component, where="train", real_split=args.real_split)
        else:
            num_id = 80
            where = "train"
            
            env, lab, _ = train_dataset[num_id]
            env = torch.tensor(env, dtype=torch.float32).to(device)
            env = env.unsqueeze(0).to(torch.float32).to(device)

            
            cam = deep_compute_gradcam(model, env, input_length=386)
            cam_all.append(cam[0])
            
            
            plot_gradcam_on_spectrum(env=env, cam = cam_all, name=args.component, num_id=num_id, where=where)
    elif args.goal == "pred":
        _, _, main_scaler, _, data_scaler = get_data(args.component, return_data="all", is_filt=args.is_filt)
        environments, local, alts = get_china_data_orin(data_scaler)
        

        class CZDataset(Dataset):
            def __init__(self, X, local, alts):
                self.X = X
                self.local = local
                self.alts = alts

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return torch.tensor(self.X[idx], dtype=torch.float32), self.local[idx][0], self.local[idx][1], self.alts[idx]

        X = environments.values if isinstance(environments, pd.DataFrame) else environments
        X_train = torch.tensor(X, dtype=torch.float32).to(device)
        train_dataset = CZDataset(X, local, alts)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = OneDCNN(input_dim=environments.shape[1]).to(device)
        if args.is_filt:
            model.load_state_dict(torch.load(f"./changzu_ans/checkpoints/single_{args.component.lower()}_{args.seed}_best_model_filt.pth"))
        else:
            model.load_state_dict(torch.load(f"./changzu_ans/checkpoints/single_{args.component.lower()}_{args.seed}_best_model.pth"))


        alt_punish_set = [0.5]
        lon_punish_set = [0.4]
        lat_punish_set = [0.2]
        for alt_punish in alt_punish_set:
            for lon_punish in lon_punish_set:
                for lat_punish in lat_punish_set:
                    print(f"draw graph with alt_punish {alt_punish}, lon_punish {lon_punish}, lat_punish {lat_punish}")
                    predictions = []
                    lon_all = []
                    lat_all = []
                    alt_all = []
                    with torch.no_grad():  
                        for inputs, lon, lat, alt in tqdm(train_loader, total=len(train_loader), desc="Predicting"):
                            inputs = inputs.to(device)
                            outputs = model(inputs, goal=args.goal, alt=alt)
                            y_pred_np = outputs.detach().cpu().numpy().reshape(-1, 1)
                            preds = main_scaler.inverse_transform(y_pred_np).squeeze(axis=1)
                            preds_min = preds.min()
                            preds_max = preds.max()
                            
                            
                            
                            
                            
                            
                            
                            if isinstance(alt, torch.Tensor):
                                alt_np = alt.detach().cpu().numpy().reshape(-1)
                            else:
                                alt_np = np.asarray(alt).reshape(-1)

                            
                            mask_alt = alt_np > 1500
                            
                            preds[mask_alt] = np.sign(preds[mask_alt]) * np.abs(preds[mask_alt]) * alt_punish


                            
                            if isinstance(lon, torch.Tensor):
                                lon_np = lon.detach().cpu().numpy().reshape(-1)
                            else:
                                lon_np = np.asarray(lon).reshape(-1)
                            mask_lon = (lon_np <= 102.78) | (lon_np >= 129.07)
                            
                            preds[mask_lon] = np.sign(preds[mask_lon]) * np.abs(preds[mask_lon]) * lon_punish

                            
                            if isinstance(lat, torch.Tensor):
                                lat_np = lat.detach().cpu().numpy().reshape(-1)
                            else:
                                lat_np = np.asarray(lat).reshape(-1)
                            mask_lat = (lat_np <= 23.17) | (lat_np >= 46.80)
                            
                            preds[mask_lat] = np.sign(preds[mask_lat]) * np.abs(preds[mask_lat]) * lat_punish
                            predictions.append(preds)
                            lon_all.append(lon)
                            lat_all.append(lat)
                            alt_all.append(alt)
                    predictions = np.concatenate(predictions, axis=0)
                    x = predictions
                    
                    
                    lon_all = np.concatenate(lon_all, axis=0)
                    lat_all = np.concatenate(lat_all, axis=0)
                    alt_all = np.concatenate(alt_all, axis=0)
                    df_all = pd.DataFrame({
                        "LON": lon_all,
                        "LAT": lat_all,
                        "QI_pred": predictions
                    })
                    if args.is_filt:
                        df_all.to_csv(f"./changzu_ans/china_{args.component}_pred_filt.csv", index=False)
                    else:
                        df_all.to_csv(f"./changzu_ans/china_{args.component}_pred.csv", index=False)

                    
                    now = datetime.now()
                    Time = now.strftime("%y%m%d_%H%M%S")

                    df = pd.DataFrame({
                            "lon": lon_all,
                            "lat": lat_all,
                            "pred": predictions
                        })

                    
                    da = df.pivot(index="lat", columns="lon", values="pred")
                    predictions_2d = da.values
                    lon = da.columns.values
                    lat = da.index.values

                    

                    bins,_,_ = compute_zones(predictions, method=args.bins_method, k=args.bins_n, clip_quantiles=(0.15, 0.85))

                    print("bins:",len(bins))

                    if args.draw_mode == "IDW":
                        
                        draw_suitability_interpolation(
                            lon_all, lat_all, predictions,
                            china_level1_path="./changzu_ans/utils/china/\u4e2d\u56fd\u884c\u653f\u533a\u5212/CN-sheng-A.shp",
                            method="idw",
                            grid_res=args.grid_res,
                            bins=bins,
                            title="Suitability zones (IDW)",
                            save_path=f"./changzu_ans/china_images/2_IDW_zones_points_alt_{alt_punish}_lon_{lon_punish}_lat_{lat_punish}_{Time}.png"
                        )
                    else:
                        
                        draw_suitability_interpolation(
                            lon_all, lat_all, predictions,
                            china_level1_path="./changzu_ans/utils/china/\u4e2d\u56fd\u884c\u653f\u533a\u5212/CN-sheng-A.shp",
                            method="kriging",
                            grid_res=args.grid_res,
                            bins=bins,
                            show_uncertainty=True,
                            title="Suitability zones (Kriging)",
                            save_path=f"./changzu_ans/china_images/3_Kriging_zones_points_alt_{alt_punish}_lon_{lon_punish}_lat_{lat_punish}_{Time}.png",
                            save_path_unc=f"./changzu_ans/china_images/4_Kriging_unc_zones_points_alt_{alt_punish}_lon_{lon_punish}_lat_{lat_punish}_{Time}_unc.png",
                        )

                    print("END")
