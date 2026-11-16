"""
=========================================================================================
MODULE: Centralized Evaluation Data Extractor & Chronological Selector
        (statistics/clean_data_extractor.py)
=========================================================================================
DESCRIPTION:
    Consolidates the extraction and chronological filtering engines (`estadistica9.py` 
    and `estadistica10.py`) into a single unified module.
=========================================================================================
"""

import os
import glob
import re
import numpy as np
import pandas as pd

SEEDS_WITH_HAIR = {
    '214631', '681147', '324235', '135790', '246801', '112233',
    '123456', '654321', '987654', '443322', '998877', '666666'
}
SEEDS_WITHOUT_HAIR = {
    '831824', '866271', '53668', '303030', '404040', '101010',
    '202020', '825040'
}


def find_model_directories(base_paths_list: list) -> list:
    discovered_folders = []
    timestamp_pattern = re.compile(r'^202\d-\d{2}-\d{2}-\d{2}-\d{2}(?:-\d{2})?')

    for base_path in base_paths_list:
        if not os.path.exists(base_path):
            continue
        for root, dirs, _ in os.walk(base_path):
            matched_here = [d for d in dirs if timestamp_pattern.match(d)]
            for m in matched_here:
                discovered_folders.append(os.path.join(root, m))
                dirs.remove(m)

    return sorted(list(set(discovered_folders)))


def infer_training_condition(model_dir_path: str) -> str:
    for log_file in glob.glob(os.path.join(model_dir_path, "log_*.txt")):
        log_low = os.path.basename(log_file).lower()
        if any(k in log_low for k in ('sin_pelos', 'sinpelos', 'without_hair', 'withouthair', 'hair_removed')):
            return 'Sin Pelos'
        if any(k in log_low for k in ('con_pelos', 'conpelos', 'with_hair', 'withhair')):
            return 'Con Pelos'

    info_files = (
        glob.glob(os.path.join(model_dir_path, "info_param*.txt")) +
        glob.glob(os.path.join(model_dir_path, "param_info*.txt")) +
        glob.glob(os.path.join(model_dir_path, "initial_info*.txt")) +
        glob.glob(os.path.join(model_dir_path, "info_inicio*.txt"))
    )
    for info_file in info_files:
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                content = f.read().lower().replace(" ", "")
                if any(k in content for k in ('"afeitado":true', "'afeitado':true", '"hair_removed":true', "'hair_removed':true")):
                    return 'Sin Pelos'
                if any(k in content for k in ('"afeitado":false', "'afeitado':false", '"hair_removed":false', "'hair_removed':false")):
                    return 'Con Pelos'
        except Exception:
            pass

    start_files = (
        glob.glob(os.path.join(model_dir_path, "info_inicio*.txt")) +
        glob.glob(os.path.join(model_dir_path, "info_start*.txt")) +
        glob.glob(os.path.join(model_dir_path, "initial_info*.txt"))
    )
    for ini_file in start_files:
        try:
            with open(ini_file, 'r', encoding='utf-8') as f:
                seed_match = re.search(r'Seed:\s*(\d+)', f.read())
                if seed_match:
                    seed_val = seed_match.group(1)
                    if seed_val in SEEDS_WITH_HAIR:
                        return 'Con Pelos'
                    if seed_val in SEEDS_WITHOUT_HAIR:
                        return 'Sin Pelos'
        except Exception:
            pass

    return 'Desconocido'


def simplify_architecture(path_or_name: str) -> str:
    n = path_or_name.lower()
    if 'resnet' in n:
        return 'ResNet-50'
    if 'conv' in n:
        return 'ConvNeXt'
    if 'swin' in n:
        return 'Swin Transformer'
    if 'vgg' in n:
        return 'VGG-16'
    if 'alexnet' in n:
        return 'AlexNet'
    if 'vim' in n or 'mamba' in n or 'vision' in n:
        return 'Vision Mamba'
    return 'Otro'


def match_checkpoint_for_metric(eval_dir_path: str, target_metric: str) -> tuple:
    ckpt_folder = os.path.basename(os.path.normpath(eval_dir_path)).lower()

    is_generic_multi = (
        ckpt_folder == 'multi' or
        ckpt_folder.startswith('multi[') or
        ('multi' in ckpt_folder and not any(m in ckpt_folder for m in ('val_loss', 'f1_score', 'f2_score', 'mel_f2')))
    )

    if target_metric == 'mel_f2':
        if 'mel_f2' in ckpt_folder:
            return True, 2
        if is_generic_multi or 'multi' in ckpt_folder:
            return True, 1
        return False, 0

    if target_metric == 'f2_score':
        if 'f2_score' in ckpt_folder and 'mel_f2' not in ckpt_folder:
            return True, 3
        if 'f2_score' in ckpt_folder:
            return True, 2
        if is_generic_multi or 'multi' in ckpt_folder:
            return True, 1
        return False, 0

    return False, 0


def extract_raw_evaluation_data(base_paths: list, target_metric: str) -> pd.DataFrame:
    model_paths = find_model_directories(base_paths)
    records = []

    for model_path in model_paths:
        mod_name = os.path.basename(model_path)
        arch = simplify_architecture(model_path)
        cond_train = infer_training_condition(model_path)
        if cond_train == 'Desconocido':
            continue

        for csv_path in glob.glob(os.path.join(model_path, "**", "score_*.csv"), recursive=True):
            eval_dir = os.path.dirname(csv_path)
            is_valid_ckpt, ckpt_priority = match_checkpoint_for_metric(eval_dir, target_metric)
            if not is_valid_ckpt:
                continue

            parts = os.path.normpath(eval_dir).split(os.sep)
            phase = "TRAIN"
            phase_folder = ""
            for p in parts:
                if p.startswith(("Results_", "Resultados_", "TEST_Results", "VAL_Results", "TRAIN_Results")):
                    phase_folder = p
                    p_up = p.upper()
                    phase = "VAL" if "VAL" in p_up else ("TEST" if "TEST" in p_up else "TRAIN")
                    break

            if phase == "TRAIN":
                continue

            is_cross = any(k in eval_dir for k in ("Verificacion_Cruzada", "Cross_Domain_Evaluation"))
            cond_test = cond_train
            if is_cross:
                cond_test = "Sin Pelos" if cond_train == "Con Pelos" else "Con Pelos"
            else:
                c_low = phase_folder.lower()
                if any(k in c_low for k in ('afeitado', 'sin_pelos', 'sinpelos', 'hair_removed', 'without_hair')):
                    cond_test = 'Sin Pelos'
                elif any(k in c_low for k in ('pelos', 'original', 'with_hair', 'raw')):
                    cond_test = 'Con Pelos'

            metric_val = np.nan
            try:
                df_csv = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
                col_renames = {}
                for col in df_csv.columns:
                    cl = str(col).lower().strip()
                    if 'mel' in cl and 'f2' in cl:
                        col_renames[col] = 'mel_f2'
                    elif 'f1' in cl:
                        col_renames[col] = 'f1_score'
                    elif 'f2' in cl and 'mel' not in cl:
                        col_renames[col] = 'f2_score'
                df_csv.rename(columns=col_renames, inplace=True)

                if target_metric in df_csv.columns:
                    df_csv[target_metric] = pd.to_numeric(df_csv[target_metric], errors='coerce')
                    valid_series = df_csv[target_metric].dropna()
                    if not valid_series.empty:
                        metric_val = float(valid_series.iloc[-1])
            except Exception:
                pass

            if not pd.isna(metric_val):
                records.append({
                    'Arquitectura': arch,
                    'Modelo': mod_name,
                    'Entrenado_Con': cond_train,
                    'Testeado_En': cond_test,
                    'Fase': phase,
                    'Es_Cruzado': is_cross,
                    'Prioridad_Ckpt': ckpt_priority,
                    'Valor_Metrica': metric_val,
                    'Ruta_Directorio': eval_dir
                })

    expected_cols = [
        'Arquitectura', 'Modelo', 'Entrenado_Con', 'Testeado_En',
        'Fase', 'Es_Cruzado', 'Valor_Metrica', 'Ruta_Directorio'
    ]
    if not records:
        return pd.DataFrame(columns=expected_cols)

    df_raw = pd.DataFrame(records)
    df_raw = df_raw.sort_values(
        by=['Arquitectura', 'Modelo', 'Fase', 'Testeado_En', 'Prioridad_Ckpt', 'Es_Cruzado'],
        ascending=[True, True, True, True, False, False]
    )
    df_dedup = df_raw.drop_duplicates(
        subset=['Arquitectura', 'Modelo', 'Fase', 'Testeado_En'],
        keep='first'
    ).drop(columns=['Prioridad_Ckpt'])

    return df_dedup.reset_index(drop=True)


def get_clean_evaluation_data(base_paths: list, target_metric: str, max_n_por_condicion: int = None) -> pd.DataFrame:
    df_raw = extract_raw_evaluation_data(base_paths, target_metric)
    if df_raw.empty or max_n_por_condicion is None:
        return df_raw

    df_filtered_final = pd.DataFrame()
    architectures = df_raw['Arquitectura'].unique()

    for arch in architectures:
        if arch == 'Otro':
            continue

        df_arch = df_raw[df_raw['Arquitectura'] == arch].copy()
        df_arch = df_arch.sort_values(by='Modelo', ascending=True)

        df_test_native = df_arch[
            (df_arch['Fase'] == 'TEST') &
            (df_arch['Entrenado_Con'] == df_arch['Testeado_En'])
        ]

        allowed_cp = (
            df_test_native[df_test_native['Entrenado_Con'] == 'Con Pelos']['Modelo']
            .drop_duplicates()
            .head(max_n_por_condicion)
            .values
        )
        allowed_sp = (
            df_test_native[df_test_native['Entrenado_Con'] == 'Sin Pelos']['Modelo']
            .drop_duplicates()
            .head(max_n_por_condicion)
            .values
        )

        valid_models = np.concatenate([allowed_cp, allowed_sp])
        df_arch_chrono = df_arch[df_arch['Modelo'].isin(valid_models)]
        df_filtered_final = pd.concat([df_filtered_final, df_arch_chrono], ignore_index=True)

    return df_filtered_final


# Aliases for backwards compatibility
extraer_datos_crudos = extract_raw_evaluation_data
obtener_datos_limpios = get_clean_evaluation_data