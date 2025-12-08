
## AJUSTAR - o limiar ainda não está rodando muito bem para fornecer um diagnóstico confiável.

"""
Módulo para diagnóstico de doenças em plantas
Baseado em limiares calibrados das métricas
"""
from typing import Dict, Tuple


# Limiares calibrados (podem ser ajustados com base em validação)
# Valores baseados na análise do notebook
THRESHOLD_CIEDE = 15000.0   # Limiar para soma CIEDE2000
THRESHOLD_HSL = 0.15        # Limiar para erro HSL


def diagnose_ciede(metrics: Dict[str, float]) -> Tuple[str, str]:
    """
    Diagnóstico baseado na métrica CIEDE2000
    
    Args:
        metrics: dicionário com as métricas calculadas
        
    Returns:
        tupla com (diagnóstico, explicação)
    """
    ciede_sum = metrics.get("ciede2000_sum", 0.0)
    
    if ciede_sum > THRESHOLD_CIEDE:
        diagnosis = "DOENTE"
        explanation = (
            f"A soma de diferenças de cor (CIEDE2000) é {ciede_sum:.2f}, "
            f"acima do limiar de {THRESHOLD_CIEDE:.2f}. "
            "Isso indica anomalias significativas na coloração da folha, "
            "possivelmente indicando a presença de doença."
        )
    else:
        diagnosis = "SAUDÁVEL"
        explanation = (
            f"A soma de diferenças de cor (CIEDE2000) é {ciede_sum:.2f}, "
            f"abaixo do limiar de {THRESHOLD_CIEDE:.2f}. "
            "A folha apresenta padrão de coloração normal, "
            "sem indícios claros de doença."
        )
    
    return diagnosis, explanation


def diagnose_hsl(metrics: Dict[str, float]) -> Tuple[str, str]:
    """
    Diagnóstico baseado na métrica HSL (Hue, Saturation, Lightness)
    
    Args:
        metrics: dicionário com as métricas calculadas
        
    Returns:
        tupla com (diagnóstico, explicação)
    """
    hsl_error = metrics.get("hsl_error", 0.0)
    
    if hsl_error > THRESHOLD_HSL:
        diagnosis = "DOENTE"
        explanation = (
            f"O erro HSL é {hsl_error:.4f}, "
            f"acima do limiar de {THRESHOLD_HSL:.4f}. "
            "Detectadas mudanças na tonalidade (Hue) e saturação da folha, "
            "características comuns de doenças como clorose ou necrose."
        )
    else:
        diagnosis = "SAUDÁVEL"
        explanation = (
            f"O erro HSL é {hsl_error:.4f}, "
            f"abaixo do limiar de {THRESHOLD_HSL:.4f}. "
            "A tonalidade e saturação da folha estão dentro do esperado, "
            "sem indícios de alterações patológicas."
        )
    
    return diagnosis, explanation


def diagnose_combined(metrics: Dict[str, float]) -> Dict[str, any]:
    """
    Diagnóstico combinando múltiplas métricas
    
    Args:
        metrics: dicionário com as métricas calculadas
        
    Returns:
        dicionário com diagnósticos detalhados
    """
    # Diagnósticos individuais
    diag_ciede, expl_ciede = diagnose_ciede(metrics)
    diag_hsl, expl_hsl = diagnose_hsl(metrics)
    
    # Diagnóstico final: considera ambos os critérios
    # Se QUALQUER método detectar doença, considera doente
    if diag_ciede == "DOENTE" or diag_hsl == "DOENTE":
        final_diagnosis = "DOENTE"
        confidence = "ALTA" if (diag_ciede == "DOENTE" and diag_hsl == "DOENTE") else "MÉDIA"
    else:
        final_diagnosis = "SAUDÁVEL"
        confidence = "ALTA"
    
    # Análise detalhada
    top2_mean = metrics.get("top2pct_mean_deltaE", 0.0)
    top1_energy = metrics.get("top1pct_energy_fraction", 0.0)
    
    # Indicadores de localização de anomalia
    if top1_energy > 0.15:
        localization = "CONCENTRADA"
        loc_detail = "As anomalias estão concentradas em regiões específicas da folha."
    elif top1_energy > 0.05:
        localization = "MODERADA"
        loc_detail = "As anomalias estão moderadamente distribuídas na folha."
    else:
        localization = "DIFUSA"
        loc_detail = "As variações de cor estão distribuídas uniformemente."
    
    return {
        "diagnosis": final_diagnosis,
        "confidence": confidence,
        "localization": localization,
        "methods": {
            "ciede2000": {
                "result": diag_ciede,
                "explanation": expl_ciede
            },
            "hsl_analysis": {
                "result": diag_hsl,
                "explanation": expl_hsl
            }
        },
        "metrics": metrics,
        "details": {
            "localization": loc_detail,
            "severity_indicator": f"Top 2% mean ΔE: {top2_mean:.2f}",
            "concentration_indicator": f"Top 1% energy fraction: {top1_energy:.4f}"
        }
    }


def get_simple_diagnosis(metrics: Dict[str, float]) -> str:
    """
    Retorna um diagnóstico simples (SAUDÁVEL ou DOENTE)
    
    Args:
        metrics: dicionário com as métricas calculadas
        
    Returns:
        string com o diagnóstico
    """
    result = diagnose_combined(metrics)
    return result["diagnosis"]


def get_diagnosis_summary(metrics: Dict[str, float]) -> str:
    """
    Retorna um resumo textual do diagnóstico
    
    Args:
        metrics: dicionário com as métricas calculadas
        
    Returns:
        string com resumo do diagnóstico
    """
    result = diagnose_combined(metrics)
    
    summary = f"**Diagnóstico: {result['diagnosis']}**\n\n"
    summary += f"**Confiança: {result['confidence']}**\n\n"
    summary += f"**Padrão de distribuição: {result['localization']}**\n\n"
    
    summary += "**Análise por método:**\n\n"
    for method, data in result['methods'].items():
        summary += f"- {method}: {data['result']}\n"
        summary += f"  {data['explanation']}\n\n"
    
    summary += "**Detalhes adicionais:**\n\n"
    for key, value in result['details'].items():
        summary += f"- {value}\n"
    
    return summary
