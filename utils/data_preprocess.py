import pandas as pd
import numpy as np

def data_preprocess(score, score_str):
    
    comp = ["gamma-Elemene","Elemol","Agarospirol","Hinesol","beta-Eudesmol", "Atractylon","Hedycaryol","Longipinocarvone","Atractylodin","Furanoeudesma-1,3-diene"]
    score = score
    
    
    
    
    
    
    
    
    
    
    


    
    

    
    b = pd.read_excel("/data/home/wxl22/changzu_ans/dataset/processed/new/all_results.xlsx")

    
    exclude_ids = {"MC01", "MC05", "MC22", "NC2406"}
    if "Sample" not in b.columns:
        raise KeyError("Column 'sampleID' not found; ensure the column name is sampleID.")
    b["Sample"] = b["Sample"].astype(str).str.strip()
    b = b[~b["Sample"].isin(exclude_ids)].reset_index(drop=True)

    
    available = [c for c in comp if c in b.columns]
    missing = [c for c in comp if c not in b.columns]
    if missing:
        print("[Warning] The following component columns were not found and will be skipped:", missing)

    
    if not available:
        raise ValueError("No component columns from 'comp' were found in dataframe b.")

    
    
    
    dir_map = {c: 'high' for c in available}
    
    
    

    
    shape_map = {c: 1.0 for c in available}

    
    
    
    auto_quants = b[available].quantile([0.10, 0.90])
    L_auto = auto_quants.loc[0.10].to_dict()
    U_auto = auto_quants.loc[0.90].to_dict()

    
    auto_quants_range = b[available].quantile([0.05, 0.25, 0.75, 0.95])
    L0_auto = auto_quants_range.loc[0.05].to_dict()
    TL_auto = auto_quants_range.loc[0.25].to_dict()
    TU_auto = auto_quants_range.loc[0.75].to_dict()
    U0_auto = auto_quants_range.loc[0.95].to_dict()

    
    
    
    
    
    
    manual_thresholds = {}

    
    def desirability_high(x, L, U, s=1.0):
        
        L, U = float(L), float(U)
        eps = 1e-12
        denom = max(U - L, eps)
        out = np.where(x <= L, 0.0,
            np.where(x >= U, 1.0, np.power((x - L) / denom, s)))
        return out

    def desirability_low(x, L, U, t=1.0):
        
        L, U = float(L), float(U)
        eps = 1e-12
        denom = max(U - L, eps)
        out = np.where(x <= L, 1.0,
            np.where(x >= U, 0.0, np.power((U - x) / denom, t)))
        return out

    def desirability_range(x, L0, TL, TU, U0, r=1.0, t=1.0):
        
        x = x.astype(float)
        L0, TL, TU, U0 = float(L0), float(TL), float(TU), float(U0)
        
        out = np.zeros_like(x, dtype=float)
        
        mask_left = (x > L0) & (x < TL)
        out = np.where(mask_left, np.power((x - L0) / max(TL - L0, 1e-12), r), out)
        
        mask_mid = (x >= TL) & (x <= TU)
        out = np.where(mask_mid, 1.0, out)
        
        mask_right = (x > TU) & (x < U0)
        out = np.where(mask_right, np.power((U0 - x) / max(U0 - TU, 1e-12), t), out)
        
        return out

    
    d_cols = {}
    for c in available:
        mode = manual_thresholds.get(c, {}).get("mode", dir_map.get(c, 'high'))
        
        s = manual_thresholds.get(c, {}).get("s", shape_map.get(c, 1.0))
        t = manual_thresholds.get(c, {}).get("t", shape_map.get(c, 1.0))

        x = b[c].astype(float)

        if mode == 'high':
            L = manual_thresholds.get(c, {}).get("L", L_auto[c])
            U = manual_thresholds.get(c, {}).get("U", U_auto[c])
            d_cols[f"D_{c}"] = desirability_high(x, L, U, s=s)

        elif mode == 'low':
            L = manual_thresholds.get(c, {}).get("L", L_auto[c])
            U = manual_thresholds.get(c, {}).get("U", U_auto[c])
            d_cols[f"D_{c}"] = desirability_low(x, L, U, t=t)

        elif mode == 'range':
            L0 = manual_thresholds.get(c, {}).get("L0", L0_auto[c])
            TL = manual_thresholds.get(c, {}).get("TL", TL_auto[c])
            TU = manual_thresholds.get(c, {}).get("TU", TU_auto[c])
            U0 = manual_thresholds.get(c, {}).get("U0", U0_auto[c])
            d_cols[f"D_{c}"] = desirability_range(x, L0, TL, TU, U0, r=s, t=t)

        else:
            raise ValueError(f"Unknown mode: {mode} (supported values: 'high'/'low'/'range')")

    d_df = pd.DataFrame(d_cols, index=b.index)

    
    
    d_names = list(d_df.columns)

    
    score_map = dict(zip(comp, score))
    w = np.array([score_map.get(c, 0) for c in available], dtype=float)  
    
    w_for_d = np.array([score_map.get(name.replace("D_",""), 0) for name in d_names], dtype=float)

    
    if w_for_d.sum() <= 0:
        w_for_d = np.ones_like(w_for_d) / len(w_for_d)
    else:
        w_for_d = w_for_d / w_for_d.sum()

    
    eps = 1e-12
    d_values = d_df.values  
    mask = ~np.isnan(d_values)
    
    log_d = np.log(np.clip(d_values, eps, 1.0))

    
    row_weight_sum = (mask * w_for_d).sum(axis=1)  
    
    row_weight_sum = np.where(row_weight_sum > 0, row_weight_sum, np.nan)

    
    log_qi = (log_d * w_for_d).sum(axis=1) / row_weight_sum
    QI = 100.0 * np.exp(log_qi)

    
    out = b.copy()
    

    out["QI"] = QI

    out.to_excel(f"/data/home/wxl22/changzu_ans/dataset/filt_comp/all_results_with_QI_filt_{score_str}.xlsx", index=False)
    print("Completed: wrote all_results_with_QI_filt_{score_str}.xlsx")
