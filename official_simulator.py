import pandas as pd
import numpy as np
from scipy.stats import poisson

print("Iniciando Escáner de Apuestas de Valor (Dixon-Coles + Kelly Criterion)...")

# Lectura del Ranking FIFA
try:
    df_ranking = pd.read_csv('ranking_fifa.csv')
    df_ranking = df_ranking.dropna(subset=['team', 'points'])
    puntos_fifa = dict(zip(df_ranking['team'], df_ranking['points']))
except FileNotFoundError:
    raise Exception("ERROR CRÍTICO: No se encontró 'ranking_fifa.csv'.")

# Ingesta de resultados históricos y ajuste de goles basado en la dificultad del rival
try:
    df_resultados = pd.read_csv('results.csv')
    df_resultados['date'] = pd.to_datetime(df_resultados['date'], format='mixed', errors='coerce')
    df_resultados['home_team'] = df_resultados['home_team'].astype(str).str.strip()
    df_resultados['away_team'] = df_resultados['away_team'].astype(str).str.strip()
except FileNotFoundError:
    raise Exception("ERROR: No se encontró 'results.csv'.")

df_oficiales = df_resultados[(df_resultados['date'] >= '2021-01-01') & (df_resultados['tournament'] != 'Friendly')].copy()
df_oficiales = df_oficiales.dropna(subset=['date', 'home_score', 'away_score'])

df_oficiales['home_pts'] = df_oficiales['home_team'].map(puntos_fifa).fillna(1400)
df_oficiales['away_pts'] = df_oficiales['away_team'].map(puntos_fifa).fillna(1400)

promedio_fifa_global = 1500.0

df_oficiales['dificultad_para_local'] = df_oficiales['away_pts'] / promedio_fifa_global
df_oficiales['dificultad_para_visita'] = df_oficiales['home_pts'] / promedio_fifa_global

df_oficiales['adj_home_score'] = df_oficiales['home_score'] * df_oficiales['dificultad_para_local']
df_oficiales['adj_away_score'] = df_oficiales['away_score'] * df_oficiales['dificultad_para_visita']

# Aplicar decaimiento exponencial temporal
fecha_actual = pd.to_datetime('today')
df_oficiales['dias_pasados'] = (fecha_actual - df_oficiales['date']).dt.days

df_oficiales['peso'] = (0.5) ** (df_oficiales['dias_pasados'] / 730.0)
df_oficiales.loc[df_oficiales['tournament'] == 'FIFA World Cup', 'peso'] *= 3.0

def weighted_avg_local(df):
    if df['peso'].sum() == 0: return pd.Series({'gf_local': 0.0, 'gc_local': 0.0})
    gf = np.average(df['adj_home_score'], weights=df['peso'])
    gc = np.average(df['adj_away_score'], weights=df['peso'])
    return pd.Series({'gf_local': gf, 'gc_local': gc})

def weighted_avg_visita(df):
    if df['peso'].sum() == 0: return pd.Series({'gf_visita': 0.0, 'gc_visita': 0.0})
    gf = np.average(df['adj_away_score'], weights=df['peso'])
    gc = np.average(df['adj_home_score'], weights=df['peso'])
    return pd.Series({'gf_visita': gf, 'gc_visita': gc})

local_stats = df_oficiales.groupby('home_team').apply(weighted_avg_local, include_groups=False)
visitante_stats = df_oficiales.groupby('away_team').apply(weighted_avg_visita, include_groups=False)

promedio_goles_local = np.average(df_oficiales['adj_home_score'], weights=df_oficiales['peso'])
promedio_goles_visitante = np.average(df_oficiales['adj_away_score'], weights=df_oficiales['peso'])
promedio_goles_neutral = (promedio_goles_local + promedio_goles_visitante) / 2

fuerzas = pd.concat([local_stats, visitante_stats], axis=1).fillna(0)
fuerzas['Ataque_Neutral'] = ((fuerzas['gf_local'] / promedio_goles_local) + (fuerzas['gf_visita'] / promedio_goles_visitante)) / 2
fuerzas['Defensa_Neutral'] = ((fuerzas['gc_local'] / promedio_goles_visitante) + (fuerzas['gc_visita'] / promedio_goles_local)) / 2

fuerzas['Ataque_Neutral'] = fuerzas['Ataque_Neutral'].fillna(1.0)
fuerzas['Defensa_Neutral'] = fuerzas['Defensa_Neutral'].fillna(1.0)

# Diccionario de Localía Dinámica
anfitriones = ['Mexico', 'United States', 'Canada']
pesos_localia = {'Mexico': 1.20, 'United States': 1.05, 'Canada': 1.03}

def obtener_bono_local(equipo):
    return pesos_localia.get(equipo, 1.10)

# Escáner Matricial con Ajuste de Dixon-Coles
def simular_partido_mundial(equipo_a, equipo_b):
    try:
        stats_a = fuerzas.loc[equipo_a]
    except KeyError:
        stats_a = {'Ataque_Neutral': 1.0, 'Defensa_Neutral': 1.0}
        
    try:
        stats_b = fuerzas.loc[equipo_b]
    except KeyError:
        stats_b = {'Ataque_Neutral': 1.0, 'Defensa_Neutral': 1.0}

    if equipo_a in anfitriones and equipo_b not in anfitriones:
        base_a, base_b = promedio_goles_local, promedio_goles_visitante
        bono_a, bono_b = obtener_bono_local(equipo_a), 1.0
        escenario = f"{equipo_a} (Anfitrión) vs {equipo_b} (Visitante)"
    elif equipo_b in anfitriones and equipo_a not in anfitriones:
        base_a, base_b = promedio_goles_visitante, promedio_goles_local
        bono_a, bono_b = 1.0, obtener_bono_local(equipo_b)
        escenario = f"{equipo_a} (Visitante) vs {equipo_b} (Anfitrión)"
    else:
        base_a = base_b = promedio_goles_neutral
        bono_a = bono_b = 1.0
        escenario = f"{equipo_a} (Neutral) vs {equipo_b} (Neutral)"

    lambda_a = stats_a['Ataque_Neutral'] * stats_b['Defensa_Neutral'] * base_a * bono_a
    lambda_b = stats_b['Ataque_Neutral'] * stats_a['Defensa_Neutral'] * base_b * bono_b

    if equipo_a not in puntos_fifa or equipo_b not in puntos_fifa:
        print(f"ADVERTENCIA: {equipo_a} o {equipo_b} no están en ranking_fifa.csv.")
        return None

    lambda_a *= (puntos_fifa[equipo_a] / puntos_fifa[equipo_b]) ** 2
    lambda_b *= (puntos_fifa[equipo_b] / puntos_fifa[equipo_a]) ** 2

    max_goles = 10
    x = np.arange(max_goles)
    y = np.arange(max_goles)
    
    P_A = poisson.pmf(x, lambda_a)
    P_B = poisson.pmf(y, lambda_b)
    
    matriz_prob = np.outer(P_A, P_B)
    
    rho = -0.13
    matriz_prob[0,0] *= (1 - lambda_a * lambda_b * rho)
    matriz_prob[1,0] *= (1 + lambda_a * rho)
    matriz_prob[0,1] *= (1 + lambda_b * rho)
    matriz_prob[1,1] *= (1 - rho)
    
    matriz_prob /= np.sum(matriz_prob)
    
    victorias_a = np.tril(matriz_prob, -1).sum() * 100
    empates = np.trace(matriz_prob) * 100
    victorias_b = np.triu(matriz_prob, 1).sum() * 100

    goles_totales = np.add.outer(x, y)
    over_1_5 = np.sum(matriz_prob[goles_totales > 1.5]) * 100
    over_2_5 = np.sum(matriz_prob[goles_totales > 2.5]) * 100
    
    ambos_anotan = np.sum(matriz_prob[1:, 1:]) * 100
    
    porteria_cero_a = np.sum(matriz_prob[:, 0]) * 100
    porteria_cero_b = np.sum(matriz_prob[0, :]) * 100

    flat_indices = np.argsort(matriz_prob.flatten())[::-1][:10]
    top_marcadores = [np.unravel_index(idx, matriz_prob.shape) for idx in flat_indices]

    print(f"\n[ ESCÁNER EXACTO: {escenario} ]")
    print(f"1X2: [1] {victorias_a:.1f}% | [X] {empates:.1f}% | [2] {victorias_b:.1f}%")
    print(f"Mercados: >1.5 Goles ({over_1_5:.1f}%) | >2.5 Goles ({over_2_5:.1f}%) | A/A ({ambos_anotan:.1f}%)")
    print(f"Portería a Cero: {equipo_a} ({porteria_cero_a:.1f}%) | {equipo_b} ({porteria_cero_b:.1f}%)")
    
    print("\nTop 5 Marcadores Exactos:")
    for i, (m_a, m_b) in enumerate(top_marcadores[:5], 1):
        prob = matriz_prob[m_a, m_b] * 100
        print(f"   {i}. {m_a} - {m_b} (Prob: {prob:.1f}%)")
        
    return {
        'prob_1': victorias_a,
        'prob_x': empates,
        'prob_2': victorias_b,
        'over_1_5': over_1_5,
        'over_2_5': over_2_5,
        'ambos_anotan': ambos_anotan
    }

# Calculadora Value Betting (Criterio de Kelly)
def analizar_apuesta(prob_modelo_pct, cuota_casino, bankroll_total, fraccion_kelly=0.5):
    print("\n[ ANÁLISIS FINANCIERO (VALUE BETTING) ]")
    
    prob = prob_modelo_pct / 100
    prob_fracaso = 1 - prob
    cuota_justa = 1 / prob if prob > 0 else 0
    
    print(f"Probabilidad de tu modelo: {prob_modelo_pct:.1f}%")
    print(f"Cuota Justa (Fair Odd): {cuota_justa:.2f}")
    print(f"Cuota de la Casa de Apuestas: {cuota_casino}")
    
    ev = (prob * cuota_casino) - 1
    
    if ev > 0:
        b = cuota_casino - 1
        kelly_base = ((b * prob) - prob_fracaso) / b
        apuesta_pct = kelly_base * fraccion_kelly
        apuesta_dinero = bankroll_total * apuesta_pct
        
        print(f"VALUE BET DETECTADA (Edge: +{ev*100:.1f}%)")
        print(f"Recomendación Kelly {fraccion_kelly}x: Destinar {apuesta_pct*100:.1f}% del bankroll.")
        print(f"Monto a apostar: ${apuesta_dinero:.2f}")
    else:
        print(f"SIN VALOR MATEMÁTICO (Edge negativo: {ev*100:.1f}%)")
        print("El casino tiene la ventaja. NO APOSTAR.")

# Ejecución desde la terminal
if __name__ == "__main__":
    # Puedes modificar los equipos aquí para analizar la Fase de Grupos
    resultados = simular_partido_mundial('Netherlands', 'Morocco')

    if resultados:
        analizar_apuesta(
            prob_modelo_pct=resultados['prob_1'], 
            cuota_casino=1.85, 
            bankroll_total=500 
        )