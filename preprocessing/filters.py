
"""
preprocessing/filters.py

Advanced filtering:
- Stability (based on H sign)
- MOST consistency (H * RiB < 0)
- Physical range filtering
- NaN handling
- Target transformation
"""

import numpy as np


def apply_filters(X, y, df, config):

    # -------- Extract required columns --------
    rib = X['rib']
    if config.data =='irg':
        H = df['H_kin']
        L = df['LE']
        #---------Quality control on H and L----
        if config.target == 'H':
            mask_qc = df['H_QC'].between(1, 6)
        elif config.target == 'L':
            mask_qc = df['LE_QC'].between(1, 6)
        else:
            mask_qc=np.ones_like(H, dtype=bool)
    else:
        H = df['H']
        mask_qc=np.ones_like(H, dtype=bool)
        

    # -------- Stability based on H --------
   
    if config.stability == "unstable":
        mask_H = H > 0
    elif config.stability == "stable":
        mask_H = H < 0
    elif config.stability == "all":
        mask_H = np.ones_like(H, dtype=bool)
    else:
        raise ValueError(f"Invalid stability: {config.stability}")

    # -------- MOST consistency --------
    mask_rib = (H * rib < 0)

    # -------- Physical range (TARGET-DEPENDENT) --------
    if config.target == "H":
        mask_range = (H >= -100) & (H <= 600)
    elif config.target == "TAU":
        # Tau should be positive and bounded
        mask_range = (y >= 0) & (y <= 3)
    elif config.target == "L":
        mask_range = (y<500) & (y>-500)
    else:
        raise ValueError(f"Invalid target: {config.target}")



    # -------- NaN mask --------
    mask_nan = (~X.isnull().any(axis=1)) & (~y.isnull())

    # -------- Combine all masks --------
    # mask = mask_H & mask_rib & mask_range & mask_nan
    # print(sum(mask),sum(mask_rib),sum(mask_H & mask_range & mask_nan))
    mask = mask_H & mask_range & mask_nan & mask_qc & mask_rib
    

    # -------- Apply mask --------
    X_filtered = X[mask].copy()
    y_filtered = y[mask].copy()

    # -------- Reset index --------
    X_filtered = X_filtered.reset_index(drop=True)
    y_filtered = y_filtered.reset_index(drop=True)

    total = len(df)

    print("\n===== FILTER STATS =====")
    print(f"Total rows: {total}")

    def count(mask, name):
        print(f"{name:<15}: {mask.sum():>6} ")

    count(mask_H, "mask_H")
    count(mask_rib, "mask_rib")
    count(mask_range, "mask_range")
    count(mask_nan, "mask_nan")
    count(mask_qc, "mask_qc")

    print("------------------------")
    print(f"Final kept     : {mask.sum()} ")
    print(f"Final removed  : {total - mask.sum()} ")
    print("========================\n")


    # # -------- Debug prints --------
    # print(f"[INFO] Total rows: {len(df)}")
    # print(f"[INFO] After filtering: {len(X_filtered)}")
    # print(f"[INFO] Removed rows: {len(df) - len(X_filtered)}")

    return X_filtered, y_filtered

