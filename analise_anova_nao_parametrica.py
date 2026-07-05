# ==========================================================
# ANÁLISE KRUSKAL-WALLIS (ANOVA NÃO-PARAMÉTRICA)
# Estrutura Completa e Modular
# ==========================================================

import pandas as pd
import numpy as np
import polars as pl
from itertools import combinations
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Estatística
from scipy.stats import (
    bootstrap,
    levene,
    f_oneway,
    kruskal,
    ttest_ind,
    mannwhitneyu,
    ranksums
)
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import scikit_posthocs as sp


# ==========================================================
# SEÇÃO 1: CONFIGURAÇÃO INICIAL
# ==========================================================

class ConfigANOVA:
    """Centraliza todas as configurações da análise"""
    
    GROUP_COL = "consumers"
    VALUE_COL = "variacao_s"
    
    # Bootstrap
    N_BOOT = 1000
    CONFIDENCE = 0.95
    
    # Paralelização
    N_JOBS = -1
    BACKEND = "loky"
    
    # Significância
    ALPHA = 0.05
    
    # Métodos de ajuste para comparações múltiplas
    METHODS_PVALUES = ["fdr_bh", "simes-hochberg", "bonferroni"]


# ==========================================================
# SEÇÃO 2: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================================

class DataPreparation:
    """Leitura, validação e preparação dos dados"""
    
    @staticmethod
    def carregar_dados(df_input, group_col, value_col):
        """
        Carrega dados com Polars e converte para Pandas
        
        Parameters:
        -----------
        df_input : DataFrame (pandas ou polars)
            Dados brutos
        group_col : str
            Nome da coluna de grupos
        value_col : str
            Nome da coluna de valores
            
        Returns:
        --------
        tuple : (df_polars, df_pandas, grupos_dict)
        """
        
        # Conversão para Polars
        if isinstance(df_input, pd.DataFrame):
            df_pl = pl.from_pandas(df_input)
        else:
            df_pl = df_input
        
        # Limpeza
        df_pl = (
            df_pl
            .select([group_col, value_col])
            .drop_nulls()
        )
        
        # Conversão para Pandas (necessária para pingouin)
        df_pd = df_pl.to_pandas()
        
        # Dicionário de grupos
        grupos = {
            g: np.array(x[value_col])
            for g, x in df_pd.groupby(group_col)
        }
        
        print(f"\n✓ Dados carregados com sucesso")
        print(f"  - Grupos: {list(grupos.keys())}")
        print(f"  - Total de observações: {len(df_pd)}")
        print(f"  - Números por grupo: {[len(v) for v in grupos.values()]}")
        
        return df_pl, df_pd, grupos
    
    @staticmethod
    def validar_dados(df_pd, grupos, group_col, value_col):
        """Validação básica dos dados"""
        
        print("\n" + "=" * 80)
        print("VALIDAÇÃO DOS DADOS")
        print("=" * 80)
        
        # Valores faltantes
        n_missing = df_pd[value_col].isna().sum()
        print(f"\n✓ Valores faltantes: {n_missing}")
        
        # Variação
        print(f"\n✓ Mínimo: {df_pd[value_col].min():.4f}")
        print(f"✓ Máximo: {df_pd[value_col].max():.4f}")
        print(f"✓ Amplitude: {df_pd[value_col].max() - df_pd[value_col].min():.4f}")
        
        # Distribuição por grupo
        print(f"\n✓ Distribuição por grupo:")
        for g, dados in grupos.items():
            print(f"  {g}: n={len(dados)}")


# ==========================================================
# SEÇÃO 3: ESTATÍSTICAS DESCRITIVAS
# ==========================================================

class DescritiveStats:
    """Análise descritiva completa por grupo"""
    
    @staticmethod
    def calcular_descritivas(grupos, value_col, n_boot=1000, confidence=0.95):
        """
        Calcula estatísticas descritivas com IC95% bootstrap
        
        Returns:
        --------
        pd.DataFrame : Tabela com todas as estatísticas
        """
        
        print("\n" + "=" * 80)
        print("ESTATÍSTICAS DESCRITIVAS")
        print("=" * 80)
        
        desc_rows = []
        
        for grupo, dados in grupos.items():
            
            n = len(dados)
            media = np.mean(dados)
            mediana = np.median(dados)
            sd = np.std(dados, ddof=1)
            se = sd / np.sqrt(n)
            
            # Bootstrap para IC95%
            ci = bootstrap(
                (dados,),
                np.mean,
                n_resamples=n_boot,
                confidence_level=confidence,
                method="BCa",
                vectorized=False
            )
            
            # IQR
            q1 = np.percentile(dados, 25)
            q3 = np.percentile(dados, 75)
            iqr = q3 - q1
            
            # Assimetria e curtose
            from scipy.stats import skew, kurtosis
            assimetria = skew(dados)
            curtose_val = kurtosis(dados)
            
            desc_rows.append({
                "Grupo": grupo,
                "N": n,
                "Média": media,
                "Mediana": mediana,
                "DP": sd,
                "EP": se,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Assimetria": assimetria,
                "Curtose": curtose_val,
                "Mín": np.min(dados),
                "Máx": np.max(dados),
                "IC95_inf": ci.confidence_interval.low,
                "IC95_sup": ci.confidence_interval.high
            })
        
        desc_df = pd.DataFrame(desc_rows)
        print("\n", desc_df.to_string(index=False))
        
        return desc_df


# ==========================================================
# SEÇÃO 4: PRESSUPOSTOS
# ==========================================================

class Pressupostos:
    """Testes dos pressupostos da ANOVA"""
    
    @staticmethod
    def testar_homogeneidade_variancia(grupos):
        """
        Teste de Levene
        H0: Variâncias são iguais
        """
        
        print("\n" + "=" * 80)
        print("PRESSUPOSTOS: HOMOGENEIDADE DE VARIÂNCIA (LEVENE)")
        print("=" * 80)
        
        lev_stat, lev_p = levene(*grupos.values())
        
        resultado = "SIM ✓" if lev_p > 0.05 else "NÃO ✗"
        
        print(f"\nEstatística: {lev_stat:.4f}")
        print(f"p-valor: {lev_p:.4f}")
        print(f"Variâncias homogêneas? {resultado}")
        print(f"Conclusão: {'Pressuposto atendido' if lev_p > 0.05 else 'Pressuposto violado - usar testes não-paramétricos'}")
        
        return pd.DataFrame({
            "Teste": ["Levene"],
            "Estatística": [lev_stat],
            "p-valor": [lev_p],
            "Homogêneo": [lev_p > 0.05]
        })
    
    @staticmethod
    def testar_normalidade_por_grupo(df_pd, group_col, value_col):
        """Teste de Shapiro-Wilk por grupo"""
        
        print("\n" + "=" * 80)
        print("PRESSUPOSTOS: NORMALIDADE POR GRUPO (SHAPIRO-WILK)")
        print("=" * 80)
        
        norm_rows = []
        
        for grupo in df_pd[group_col].unique():
            dados = df_pd[df_pd[group_col] == grupo][value_col].values
            
            from scipy.stats import shapiro
            stat, p = shapiro(dados)
            
            normal = "SIM ✓" if p > 0.05 else "NÃO ✗"
            
            norm_rows.append({
                "Grupo": grupo,
                "N": len(dados),
                "W": stat,
                "p-valor": p,
                "Normal": normal
            })
        
        norm_df = pd.DataFrame(norm_rows)
        print("\n", norm_df.to_string(index=False))
        
        return norm_df


# ==========================================================
# SEÇÃO 5: TESTES OMNIBUS
# ==========================================================

class TestesOmnibus:
    """Testes gerais de igualdade entre grupos"""
    
    @staticmethod
    def teste_kruskal_wallis(grupos):
        """
        Teste de Kruskal-Wallis (não-paramétrico)
        H0: Distribuições são iguais
        """
        
        print("\n" + "=" * 80)
        print("TESTE OMNIBUS: KRUSKAL-WALLIS (NÃO-PARAMÉTRICO)")
        print("=" * 80)
        
        h_stat, p_val = kruskal(*grupos.values())
        
        resultado = "DIFERENÇAS SIGNIFICATIVAS ✓" if p_val < 0.05 else "SEM DIFERENÇAS SIGNIFICATIVAS"
        
        print(f"\nEstatística H: {h_stat:.4f}")
        print(f"p-valor: {p_val:.6f}")
        print(f"Resultado: {resultado}")
        
        return pd.DataFrame({
            "Teste": ["Kruskal-Wallis"],
            "H": [h_stat],
            "p-valor": [p_val],
            "Significativo": [p_val < 0.05]
        })
    
    @staticmethod
    def teste_anova_parametrica(grupos):
        """ANOVA clássica (para comparação)"""
        
        print("\n" + "=" * 80)
        print("TESTE OMNIBUS: ANOVA CLÁSSICA (PARAMÉTRICA - PARA REFERÊNCIA)")
        print("=" * 80)
        
        f_stat, p_val = f_oneway(*grupos.values())
        
        resultado = "DIFERENÇAS SIGNIFICATIVAS ✓" if p_val < 0.05 else "SEM DIFERENÇAS SIGNIFICATIVAS"
        
        print(f"\nEstatística F: {f_stat:.4f}")
        print(f"p-valor: {p_val:.6f}")
        print(f"Resultado: {resultado}")
        
        return pd.DataFrame({
            "Teste": ["ANOVA"],
            "F": [f_stat],
            "p-valor": [p_val],
            "Significativo": [p_val < 0.05]
        })


# ==========================================================
# SEÇÃO 6: COMPARAÇÕES MÚLTIPLAS
# ==========================================================

class ComparacoesMultiplas:
    """Comparações pairwise com ajuste de p-valor"""
    
    @staticmethod
    def mann_whitney_com_ajuste(grupos, method="fdr_bh"):
        """
        Mann-Whitney U test (não-paramétrico) com ajuste de p-valor
        """
        
        print("\n" + "=" * 80)
        print(f"COMPARAÇÕES MÚLTIPLAS: MANN-WHITNEY U COM AJUSTE {method.upper()}")
        print("=" * 80)
        
        pares = []
        pvals = []
        stats_vals = []
        
        for g1, g2 in combinations(grupos.keys(), 2):
            
            u_stat, p = mannwhitneyu(
                grupos[g1],
                grupos[g2],
                alternative='two-sided'
            )
            
            pares.append((g1, g2))
            pvals.append(p)
            stats_vals.append(u_stat)
        
        # Ajuste de p-valores
        p_adj = multipletests(pvals, method=method)[1]
        
        mw_rows = []
        
        for (g1, g2), p_orig, p_corr, u_stat in zip(pares, pvals, p_adj, stats_vals):
            
            sig = "SIM ✓" if p_corr < 0.05 else "NÃO"
            
            mw_rows.append({
                "Grupo 1": g1,
                "Grupo 2": g2,
                "U": u_stat,
                "p-original": p_orig,
                "p-ajustado": p_corr,
                "Significativo": sig
            })
        
        mw_df = pd.DataFrame(mw_rows)
        print("\n", mw_df.to_string(index=False))
        
        return mw_df
    
    @staticmethod
    def wilcoxon_rank_sum(grupos):
        """Alternativa: Wilcoxon Rank Sum Test"""
        
        print("\n" + "=" * 80)
        print("COMPARAÇÕES MÚLTIPLAS: WILCOXON RANK SUM (z-test)")
        print("=" * 80)
        
        pares = []
        pvals = []
        zstats = []
        
        for g1, g2 in combinations(grupos.keys(), 2):
            
            z_stat, p = ranksums(grupos[g1], grupos[g2])
            
            pares.append((g1, g2))
            pvals.append(p)
            zstats.append(z_stat)
        
        # Ajuste Hochberg
        p_adj = multipletests(pvals, method="simes-hochberg")[1]
        
        ws_rows = []
        
        for (g1, g2), p_orig, p_corr, z_stat in zip(pares, pvals, p_adj, zstats):
            
            sig = "SIM ✓" if p_corr < 0.05 else "NÃO"
            
            ws_rows.append({
                "Grupo 1": g1,
                "Grupo 2": g2,
                "z": z_stat,
                "p-original": p_orig,
                "p-ajustado": p_corr,
                "Significativo": sig
            })
        
        ws_df = pd.DataFrame(ws_rows)
        print("\n", ws_df.to_string(index=False))
        
        return ws_df


# ==========================================================
# SEÇÃO 7: TAMANHOS DE EFEITO
# ==========================================================

class TamanhosEfeito:
    """Cálculo de múltiplas métricas de tamanho de efeito"""
    
    @staticmethod
    def eta_squared(df_pd, group_col, value_col, grupos):
        """Eta-squared (η²) - efeito global"""
        
        N = len(df_pd)
        k = len(grupos)
        grand_mean = df_pd[value_col].mean()
        
        ss_between = sum(
            len(v) * (np.mean(v) - grand_mean) ** 2
            for v in grupos.values()
        )
        
        ss_total = np.sum((df_pd[value_col] - grand_mean) ** 2)
        
        eta2 = ss_between / ss_total
        
        return eta2
    
    @staticmethod
    def omega_squared(df_pd, value_col, grupos):
        """Omega-squared (ω²) - estimador menos viesado"""
        
        N = len(df_pd)
        k = len(grupos)
        grand_mean = df_pd[value_col].mean()
        
        ss_between = sum(
            len(v) * (np.mean(v) - grand_mean) ** 2
            for v in grupos.values()
        )
        
        ss_total = np.sum((df_pd[value_col] - grand_mean) ** 2)
        ss_within = ss_total - ss_between
        
        df_between = k - 1
        df_within = N - k
        ms_within = ss_within / df_within
        
        omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within)
        
        return omega2
    
    @staticmethod
    def epsilon_squared(df_pd, value_col, grupos):
        """Epsilon-squared (ε²)"""
        
        N = len(df_pd)
        k = len(grupos)
        grand_mean = df_pd[value_col].mean()
        
        ss_between = sum(
            len(v) * (np.mean(v) - grand_mean) ** 2
            for v in grupos.values()
        )
        
        ss_total = np.sum((df_pd[value_col] - grand_mean) ** 2)
        ss_within = ss_total - ss_between
        
        df_between = k - 1
        df_within = N - k
        ms_within = ss_within / df_within
        
        epsilon2 = (ss_between - df_between * ms_within) / ss_total
        
        return epsilon2
    
    @staticmethod
    def rank_biserial(x, y):
        """Rank-Biserial correlation (para comparações pairwise)"""
        
        n1, n2 = len(x), len(y)
        
        # Combinação e ranking
        combined = np.concatenate([x, y])
        ranks = np.argsort(np.argsort(combined)) + 1
        
        r1 = np.sum(ranks[:n1])
        
        # Mann-Whitney U
        u = r1 - n1 * (n1 + 1) / 2
        
        # Correlação rank-biserial
        r = 1 - (2 * u) / (n1 * n2)
        
        return r
    
    @staticmethod
    def calcular_todos_efeitos(df_pd, group_col, value_col, grupos):
        """Consolidação de todos os tamanhos de efeito"""
        
        print("\n" + "=" * 80)
        print("TAMANHOS DE EFEITO - GLOBAIS")
        print("=" * 80)
        
        eta2 = TamanhosEfeito.eta_squared(df_pd, group_col, value_col, grupos)
        omega2 = TamanhosEfeito.omega_squared(df_pd, value_col, grupos)
        epsilon2 = TamanhosEfeito.epsilon_squared(df_pd, value_col, grupos)
        
        efeito_df = pd.DataFrame({
            "η² (Eta-squared)": [eta2],
            "ω² (Omega-squared)": [omega2],
            "ε² (Epsilon-squared)": [epsilon2]
        })
        
        print("\n", efeito_df.to_string(index=False))
        
        # Interpretação
        print("\n" + "-" * 80)
        print("INTERPRETAÇÃO (Cohen):")
        print("-" * 80)
        
        for nome, valor in [("η²", eta2), ("ω²", omega2), ("ε²", epsilon2)]:
            if valor < 0.01:
                interp = "Negligenciável"
            elif valor < 0.06:
                interp = "Pequeno"
            elif valor < 0.14:
                interp = "Médio"
            else:
                interp = "Grande"
            
            print(f"{nome}: {valor:.4f} → {interp}")
        
        return efeito_df


# ==========================================================
# SEÇÃO 8: BOOTSTRAP AVANÇADO
# ==========================================================

class BootstrapAnalise:
    """Análises baseadas em bootstrap"""
    
    @staticmethod
    def bootstrap_por_grupo(grupos, n_boot=1000, n_jobs=-1):
        """Bootstrap dos estatísticos por grupo"""
        
        print("\n" + "=" * 80)
        print("BOOTSTRAP POR GRUPO")
        print("=" * 80)
        
        def _bootstrap_mean(data):
            resample = np.random.choice(data, size=len(data), replace=True)
            return np.mean(resample)
        
        bootstrap_rows = []
        
        for grupo, dados in grupos.items():
            
            media_obs = np.mean(dados)
            
            # Bootstrap paralelo
            boot_means = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_bootstrap_mean)(dados)
                for _ in range(n_boot)
            )
            
            boot_means = np.array(boot_means)
            
            bootstrap_rows.append({
                "Grupo": grupo,
                "Média Observada": media_obs,
                "Viés": np.mean(boot_means) - media_obs,
                "Erro Padrão Bootstrap": np.std(boot_means, ddof=1),
                "IC95_inf": np.percentile(boot_means, 2.5),
                "IC95_sup": np.percentile(boot_means, 97.5)
            })
        
        boot_df = pd.DataFrame(bootstrap_rows)
        print("\n", boot_df.to_string(index=False))
        
        return boot_df
    
    @staticmethod
    def bootstrap_comparacoes(grupos, n_boot=1000, confidence=0.95):
        """Bootstrap para diferenças entre grupos"""
        
        print("\n" + "=" * 80)
        print("BOOTSTRAP - COMPARAÇÕES MÚLTIPLAS (IC95%)")
        print("=" * 80)
        
        def diff_means(x, y):
            return np.mean(x) - np.mean(y)
        
        boot_comp_rows = []
        
        for g1, g2 in combinations(grupos.keys(), 2):
            
            x = grupos[g1]
            y = grupos[g2]
            
            diff = np.mean(x) - np.mean(y)
            
            res = bootstrap(
                (x, y),
                diff_means,
                paired=False,
                n_resamples=n_boot,
                confidence_level=confidence,
                method="BCa",
                vectorized=False
            )
            
            # Significância (IC não inclui zero)
            sig = "SIM ✓" if not (res.confidence_interval.low <= 0 <= res.confidence_interval.high) else "NÃO"
            
            boot_comp_rows.append({
                "Grupo 1": g1,
                "Grupo 2": g2,
                "Dif. Média": diff,
                "IC95_inf": res.confidence_interval.low,
                "IC95_sup": res.confidence_interval.high,
                "Significativo": sig
            })
        
        boot_comp_df = pd.DataFrame(boot_comp_rows)
        print("\n", boot_comp_df.to_string(index=False))
        
        return boot_comp_df


# ==========================================================
# SEÇÃO 9: RELATÓRIO CONSOLIDADO
# ==========================================================

class RelatorioFinal:
    """Geração de relatório consolidado"""
    
    @staticmethod
    def gerar_relatorio(
        desc_df, 
        levene_df, 
        norm_df,
        kw_df, 
        anova_df,
        mw_df, 
        efeito_df, 
        boot_df,
        boot_comp_df
    ):
        """Compila todos os resultados em um relatório estruturado"""
        
        print("\n\n" + "=" * 80)
        print("RELATÓRIO CONSOLIDADO - ANÁLISE ANOVA NÃO-PARAMÉTRICA")
        print("=" * 80)
        
        print("\n1. DESCRITIVAS")
        print("-" * 80)
        print(desc_df.to_string(index=False))
        
        print("\n\n2. PRESSUPOSTOS")
        print("-" * 80)
        print("\n2.1 - Homogeneidade de Variância (Levene):")
        print(levene_df.to_string(index=False))
        
        print("\n2.2 - Normalidade (Shapiro-Wilk):")
        print(norm_df.to_string(index=False))
        
        print("\n\n3. TESTES OMNIBUS")
        print("-" * 80)
        print("\n3.1 - Kruskal-Wallis (não-paramétrico):")
        print(kw_df.to_string(index=False))
        
        print("\n3.2 - ANOVA (paramétrica - referência):")
        print(anova_df.to_string(index=False))
        
        print("\n\n4. COMPARAÇÕES MÚLTIPLAS")
        print("-" * 80)
        print("Mann-Whitney U com ajuste FDR:")
        print(mw_df.to_string(index=False))
        
        print("\n\n5. TAMANHOS DE EFEITO")
        print("-" * 80)
        print(efeito_df.to_string(index=False))
        
        print("\n\n6. BOOTSTRAP")
        print("-" * 80)
        print("\n6.1 - Por Grupo:")
        print(boot_df.to_string(index=False))
        
        print("\n6.2 - Comparações Múltiplas:")
        print(boot_comp_df.to_string(index=False))
        
        print("\n" + "=" * 80)
        print("FIM DO RELATÓRIO")
        print("=" * 80 + "\n")


# ==========================================================
# SEÇÃO 10: PIPELINE PRINCIPAL
# ==========================================================

def executar_analise_completa(df_input, group_col="consumers", value_col="variacao_s"):
    """
    Executa a análise ANOVA não-paramétrica completa
    
    Parameters:
    -----------
    df_input : pd.DataFrame
        DataFrame com os dados
    group_col : str
        Nome da coluna de grupos
    value_col : str
        Nome da coluna de valores
    """
    
    print("\n" + "=" * 80)
    print("ANÁLISE KRUSKAL-WALLIS COMPLETA")
    print("=" * 80)
    
    # 1. Carregamento
    df_pl, df_pd, grupos = DataPreparation.carregar_dados(
        df_input, group_col, value_col
    )
    
    # 2. Validação
    DataPreparation.validar_dados(df_pd, grupos, group_col, value_col)
    
    # 3. Descritivas
    desc_df = DescritiveStats.calcular_descritivas(
        grupos, 
        value_col,
        n_boot=ConfigANOVA.N_BOOT,
        confidence=ConfigANOVA.CONFIDENCE
    )
    
    # 4. Pressupostos
    levene_df = Pressupostos.testar_homogeneidade_variancia(grupos)
    norm_df = Pressupostos.testar_normalidade_por_grupo(df_pd, group_col, value_col)
    
    # 5. Testes Omnibus
    kw_df = TestesOmnibus.teste_kruskal_wallis(grupos)
    anova_df = TestesOmnibus.teste_anova_parametrica(grupos)
    
    # 6. Comparações Múltiplas
    mw_df = ComparacoesMultiplas.mann_whitney_com_ajuste(grupos, method="fdr_bh")
    ws_df = ComparacoesMultiplas.wilcoxon_rank_sum(grupos)
    
    # 7. Tamanhos de Efeito
    efeito_df = TamanhosEfeito.calcular_todos_efeitos(
        df_pd, group_col, value_col, grupos
    )
    
    # 8. Bootstrap
    boot_df = BootstrapAnalise.bootstrap_por_grupo(
        grupos,
        n_boot=ConfigANOVA.N_BOOT,
        n_jobs=ConfigANOVA.N_JOBS
    )
    
    boot_comp_df = BootstrapAnalise.bootstrap_comparacoes(
        grupos,
        n_boot=ConfigANOVA.N_BOOT,
        confidence=ConfigANOVA.CONFIDENCE
    )
    
    # 9. Relatório Consolidado
    RelatorioFinal.gerar_relatorio(
        desc_df, levene_df, norm_df, kw_df, anova_df,
        mw_df, efeito_df, boot_df, boot_comp_df
    )
    
    # Retornar todos os DataFrames
    return {
        "descritivas": desc_df,
        "levene": levene_df,
        "normalidade": norm_df,
        "kruskal_wallis": kw_df,
        "anova": anova_df,
        "mann_whitney": mw_df,
        "wilcoxon": ws_df,
        "tamanho_efeito": efeito_df,
        "bootstrap_grupos": boot_df,
        "bootstrap_comparacoes": boot_comp_df
    }


# ==========================================================
# EXECUÇÃO
# ==========================================================

if __name__ == "__main__":
    
    # Usar dados já carregados na sessão
    resultados = executar_analise_completa(
        df_input=tabela_evento_1,
        group_col="consumers",
        value_col="variacao_s"
    )
    
    # Acessar resultados específicos
    print("\n\n✓ Análise concluída!")
    print("\nDataFrames disponíveis em 'resultados':")
    for chave in resultados.keys():
        print(f"  - resultados['{chave}']")
