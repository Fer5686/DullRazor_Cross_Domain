"""
=========================================================================================
MODULE: Cross-Architecture Macro Comparison & Baseline Benchmarking
        (statistics/macro_architecture_comparison.py)
=========================================================================================
DESCRIPTION:
    Evaluates the macro-level hierarchy across deep learning backbones (AlexNet, VGG-16, 
    ResNet-50, Swin Transformer, ConvNeXt, Vision Mamba) under native conditions:
    - Native Raw Performance (Trained With Hair / Tested With Hair).
    - Native Clean Performance (Trained Without Hair / Tested Without Hair via DullRazor).
    
    Statistical Hypothesis Test:
    Performs Welch's two-sample t-test (`equal_var=False`, `ddof=1`) and signed pooled 
    Cohen's d comparing each architecture's Native Raw performance against the Classical 
    Clean Baseline (ResNet-50 trained and tested on DullRazor hair-removed images).
    Exports publication-ready APA / Elsevier Excel workbooks.
=========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# 0. PATH CONFIGURATION & CENTRAL DATA ENGINE IMPORT
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "Main_code")

for p in (CURRENT_DIR, PROJECT_ROOT, MAIN_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config_paths
from clean_data_extractor import get_clean_evaluation_data as obtener_datos_limpios

MODELS_DIR = getattr(config_paths, "MODELS_DIR", getattr(config_paths, "REPO_DIR"))
DOCS_DIR = getattr(config_paths, "DOCS_DIR")


# ==============================================================================
# 1. MULTI-LANGUAGE DICTIONARY (English, Spanish, Portuguese)
# ==============================================================================
MACRO_I18N = {
    'en': {
        'sheet_name': 'Global_Comparison',
        'col_arch': 'Architecture Hierarchy',
        'col_raw': 'Native Raw Performance\n(Train Hair / Test Hair)',
        'col_clean': 'Native Clean Performance\n(Train Clean / Test Clean)',
        'col_ttest': 'Welch T-Test: Architecture Raw vs. Classical Clean Baseline',
        'baseline_label': 'Classical Clean Baseline',
        'outperforms': '(Outperforms Baseline)',
        'underperforms': '(Underperforms Baseline)'
    },
    'es': {
        'sheet_name': 'Comparativa_Global',
        'col_arch': 'Jerarquía de Arquitecturas',
        'col_raw': 'Rendimiento Nativo Crudo\n(Entr. Pelos / Test Pelos)',
        'col_clean': 'Rendimiento Nativo Limpio\n(Entr. Limpio / Test Limpio)',
        'col_ttest': 'T-Test Welch: Arquitectura Cruda vs. Baseline Clásico Limpio',
        'baseline_label': 'Baseline Clásico',
        'outperforms': '(Supera al Baseline)',
        'underperforms': '(Inferior al Baseline)'
    },
    'pt': {
        'sheet_name': 'Comparacao_Global',
        'col_arch': 'Hierarquia de Arquiteturas',
        'col_raw': 'Desempenho Nativo Bruto\n(Trein. Pelos / Teste Pelos)',
        'col_clean': 'Desempenho Nativo Limpo\n(Trein. Limpo / Teste Limpo)',
        'col_ttest': 'T-Test Welch: Arquitetura Bruta vs. Baseline Clássico Limpo',
        'baseline_label': 'Baseline Clássico',
        'outperforms': '(Supera o Baseline)',
        'underperforms': '(Inferior ao Baseline)'
    }
}


def normalize_language(lang_code: str) -> str:
    lang = str(lang_code).lower().strip()
    if lang in ['es', 'castellano', 'español', 'spanish']:
        return 'es'
    if lang in ['pt', 'portugues', 'português', 'portuguese']:
        return 'pt'
    return 'en'


# ==============================================================================
# 2. WELCH'S T-TEST & SIGNED COHEN'S D
# ==============================================================================
def calc_welch_ttest(a, b) -> tuple:
    """
    Computes Welch's two-sample t-test (unequal variances) and signed pooled Cohen's d.
    A positive Cohen's d indicates that population `a` (Raw Architecture) has a higher 
    mean than population `b` (Clean Baseline).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    
    _, p = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt(
        ((len(a) - 1) * np.std(a, ddof=1) ** 2 + (len(b) - 1) * np.std(b, ddof=1) ** 2) /
        (len(a) + len(b) - 2)
    )
    d = (np.mean(a) - np.mean(b)) / max(pooled_std, 1e-5)
    return float(p), float(d)


# ==============================================================================
# 3. ELSEVIER / APA EXCEL TABLE EXPORT
# ==============================================================================
def export_macro_elsevier_table(df_results: pd.DataFrame, metric_name: str, save_dir: str, language: str = 'en'):
    lang_code = normalize_language(language)
    t = MACRO_I18N[lang_code]
    
    os.makedirs(save_dir, exist_ok=True)
    excel_path = os.path.join(save_dir, f"Macro_Architecture_Analysis_{metric_name}_{lang_code.upper()}.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        ws = workbook.add_worksheet(t['sheet_name'])
        ws.set_column('A:A', 25)
        ws.set_column('B:C', 30)
        ws.set_column('D:D', 44)

        fmt_header_left = workbook.add_format({'bold': True, 'bottom': 1, 'top': 1, 'align': 'left', 'valign': 'vcenter'})
        fmt_header = workbook.add_format({'bold': True, 'bottom': 1, 'top': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
        
        fmt_cell_left = workbook.add_format({'align': 'left', 'valign': 'vcenter'})
        fmt_cell = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        fmt_bottom_left = workbook.add_format({'bottom': 1, 'align': 'left', 'valign': 'vcenter'})
        fmt_bottom = workbook.add_format({'bottom': 1, 'align': 'center', 'valign': 'vcenter'})

        ws.write(0, 0, t['col_arch'], fmt_header_left)
        ws.write(0, 1, t['col_raw'], fmt_header)
        ws.write(0, 2, t['col_clean'], fmt_header)
        ws.write(0, 3, t['col_ttest'], fmt_header)

        for i, row in df_results.iterrows():
            row_idx = i + 1
            is_last = (i == len(df_results) - 1)
            c_left = fmt_bottom_left if is_last else fmt_cell_left
            c_center = fmt_bottom if is_last else fmt_cell
            
            p_val = row['P-Value vs Baseline']
            d_val = row['Cohen d vs Baseline']
            
            if pd.isna(p_val):
                ttest_text = t['baseline_label']
            else:
                ttest_text = f"p = {p_val:.4f}, d = {d_val:.4f}"
                if p_val < 0.05 and d_val > 0:
                    ttest_text += f" {t['outperforms']}"
                elif p_val < 0.05 and d_val < 0:
                    ttest_text += f" {t['underperforms']}"

            ws.write(row_idx, 0, row['Architecture'], c_left)
            ws.write(row_idx, 1, f"{row['Raw Mean']:.4f} \u00B1 {row['Raw Std']:.4f}", c_center)
            ws.write(row_idx, 2, f"{row['Clean Mean']:.4f} \u00B1 {row['Clean Std']:.4f}", c_center)
            
            is_sig = (not pd.isna(p_val) and p_val < 0.05)
            if is_sig:
                c_ttest = (
                    workbook.add_format({'bottom': 1, 'align': 'center', 'valign': 'vcenter', 'bold': True})
                    if is_last else
                    workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True})
                )
            else:
                c_ttest = c_center
                
            ws.write(row_idx, 3, ttest_text, c_ttest)


# ==============================================================================
# 4. MAIN MACRO ANALYSIS GENERATOR
# ==============================================================================
def generate_macro_analysis(base_folders: list, save_dir: str, max_n: int = 3, language: str = 'en'):
    architecture_order = [
        'AlexNet', 'VGG-16', 'ResNet-50', 
        'Swin Transformer', 'Vision Mamba', 'ConvNeXt'
    ]
    metrics = ['f2_score', 'mel_f2']

    for metric in metrics:
        print(f"\n{'=' * 70}\n📊 GENERATING MACRO ARCHITECTURE MATRIX FOR: {metric.upper()}\n{'=' * 70}")
        
        df = obtener_datos_limpios(base_folders, metric)
        if df.empty:
            print(f"⚠️ Warning: No clean evaluation records found for '{metric}'.")
            continue
        
        results = []
        raw_data_by_arch = {}
        clean_data_by_arch = {}

        # 1. Extract symmetric N populations per architecture
        for arch in architecture_order:
            df_arch = df[df['Arquitectura'] == arch]
            if df_arch.empty:
                continue

            mod_cp = sorted(df_arch[df_arch['Entrenado_Con'] == 'Con Pelos']['Modelo'].unique())[:max_n]
            mod_sp = sorted(df_arch[df_arch['Entrenado_Con'] == 'Sin Pelos']['Modelo'].unique())[:max_n]
            df_arch = df_arch[df_arch['Modelo'].isin(list(mod_cp) + list(mod_sp))]

            df_test = df_arch[df_arch['Fase'] == 'TEST']
            
            raw_vals = df_test[(df_test['Entrenado_Con'] == 'Con Pelos') & (df_test['Testeado_En'] == 'Con Pelos')]['Valor_Metrica'].dropna().values
            clean_vals = df_test[(df_test['Entrenado_Con'] == 'Sin Pelos') & (df_test['Testeado_En'] == 'Sin Pelos')]['Valor_Metrica'].dropna().values

            if len(raw_vals) >= 2 and len(clean_vals) >= 2:
                raw_data_by_arch[arch] = raw_vals
                clean_data_by_arch[arch] = clean_vals

        # 2. Define Classical Clean Baseline (Prioritizes ResNet-50 Clean, fallback to AlexNet Clean)
        baseline_arch = (
            'ResNet-50' if 'ResNet-50' in clean_data_by_arch 
            else ('AlexNet' if 'AlexNet' in clean_data_by_arch else None)
        )
        if not baseline_arch:
            print("⚠️ Classical baseline architecture (ResNet-50 or AlexNet) not found in dataset.")
            continue
            
        baseline_population = clean_data_by_arch[baseline_arch]
        print(f"📌 Reference Classical Clean Baseline: {baseline_arch} (N={len(baseline_population)}, Mean={np.mean(baseline_population):.4f})")

        # 3. Compute metrics and compare against Classical Clean Baseline
        for arch in architecture_order:
            if arch not in raw_data_by_arch:
                continue
            
            raw_vals = raw_data_by_arch[arch]
            clean_vals = clean_data_by_arch[arch]
            
            p, d = np.nan, np.nan
            if arch != baseline_arch:
                # Welch's T-Test: Current Architecture Raw vs. Classical Clean Baseline
                p, d = calc_welch_ttest(raw_vals, baseline_population)

            results.append({
                'Architecture': arch,
                'Raw Mean': np.mean(raw_vals),
                'Raw Std': np.std(raw_vals, ddof=1),
                'Clean Mean': np.mean(clean_vals),
                'Clean Std': np.std(clean_vals, ddof=1),
                'P-Value vs Baseline': p,
                'Cohen d vs Baseline': d
            })
            print(f"   ➤ {arch:18s} | Raw: {np.mean(raw_vals):.4f}±{np.std(raw_vals, ddof=1):.4f} | Clean: {np.mean(clean_vals):.4f}±{np.std(clean_vals, ddof=1):.4f} | p vs Base: {p:.4f}")

        df_res = pd.DataFrame(results)
        export_macro_elsevier_table(df_res, metric, save_dir, language=language)
        print(f"✅ Macro comparison table exported successfully ({metric}).")


if __name__ == "__main__":
    # Scans the centralized trained_models directory recursively (no per-model paths needed)
    experiment_directories = [
        MODELS_DIR
    ]

    # Saves generated macro comparison Excel workbooks inside the repository's Datasets/HAM10000/PAPER_FIGURES_VERIFY folder
    figures_output_dir = os.path.join(DOCS_DIR, "HAM10000", "PAPER_FIGURES_VERIFY")
    os.makedirs(figures_output_dir, exist_ok=True)

    print(f"📂 Scanning models in: {MODELS_DIR}")
    print(f"📊 Exporting macro analysis tables to: {figures_output_dir}")

    # Supported language options: 'en' (English), 'es' (Spanish), 'pt' (Portuguese)
    generate_macro_analysis(experiment_directories, figures_output_dir, max_n=5, language='en')