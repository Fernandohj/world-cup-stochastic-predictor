import pandas as pd
import numpy as np
from scipy.stats import poisson

print("Iniciando Escáner Matricial Individual (Dixon-Coles + Penales + Top 10)...")

# Lectura del Ranking FIFA
try:
    df_ranking = pd.read_csv('ranking_fifa.csv')
    df_ranking = df_ranking.dropna(subset=['team', 'points'])
    puntos_fifa = dict(zip(df_ranking['team'], df_ranking['points']))
except FileNotFoundError:
    raise Exception("ERROR CRÍTICO: No se encontró 'ranking_fifa.csv'.")

# Ingesta y ajuste de goles por rival
try:
    df_resultados = pd.read_csv('results.csv')
    df_resultados['date'] = pd.to_datetime(df_resultados['date'], format='mixed', errors='coerce')
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

# Ingesta de historial de penales
try:
    df_penales = pd.read_csv('shootouts.csv')
    participaciones = df_penales['home_team'].value_counts().add(df_penales['away_team'].value_counts(), fill_value=0)
    victorias_penales = df_penales['winner'].value_counts()
    efectividad_penales = (victorias_penales / participaciones).fillna(0).to_dict()
except FileNotFoundError:
    print("ADVERTENCIA: No se encontró 'shootouts.csv'. Se asumirá una probabilidad de 50/50 en penales.")
    efectividad_penales = {}

def obtener_prob_penales(equipo_a, equipo_b):
    efectividad_a = efectividad_penales.get(equipo_a, 0.5)
    efectividad_b = efectividad_penales.get(equipo_b, 0.5)
    prob_ganar_a = (efectividad_a + (1 - efectividad_b)) / 2
    prob_ganar_b = 1 - prob_ganar_a
    return prob_ganar_a, prob_ganar_b

# Motor de ponderación matemática (Time Decay)
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

# Diccionario de localía dinámica
anfitriones = ['Mexico', 'United States', 'Canada']
pesos_localia = {'Mexico': 1.20, 'United States': 1.05, 'Canada': 1.03}

def obtener_bono_local(equipo):
    return pesos_localia.get(equipo, 1.10)

# Escáner matricial detallado con factor Dixon-Coles
def analizar_partido_individual(equipo_a, equipo_b, es_eliminatoria=True):
    try:
        stats_a = fuerzas.loc[equipo_a]
    except KeyError:
        stats_a = {'Ataque_Neutral': 1.0, 'Defensa_Neutral': 1.0}
        
    try:
        stats_b = fuerzas.loc[equipo_b]
    except KeyError:
        stats_b = {'Ataque_Neutral': 1.0, 'Defensa_Neutral': 1.0}

    # Ajuste de Localía
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

    pts_a = puntos_fifa.get(equipo_a, 1400)
    pts_b = puntos_fifa.get(equipo_b, 1400)
    
    lambda_a *= (pts_a / pts_b) ** 2
    lambda_b *= (pts_b / pts_a) ** 2

    # Generación de la Matriz Bivariada
    max_goles = 10
    x, y = np.arange(max_goles), np.arange(max_goles)
    matriz_prob = np.outer(poisson.pmf(x, lambda_a), poisson.pmf(y, lambda_b))
    
    # Factor de corrección Dixon-Coles
    rho = -0.13
    matriz_prob[0,0] *= (1 - lambda_a * lambda_b * rho)
    matriz_prob[1,0] *= (1 + lambda_a * rho)
    matriz_prob[0,1] *= (1 + lambda_b * rho)
    matriz_prob[1,1] *= (1 - rho)
    matriz_prob /= np.sum(matriz_prob)
    
    # Extracción de mercados
    victorias_a = np.tril(matriz_prob, -1).sum() * 100
    empates = np.trace(matriz_prob) * 100
    victorias_b = np.triu(matriz_prob, 1).sum() * 100

    goles_totales = np.add.outer(x, y)
    over_1_5 = np.sum(matriz_prob[goles_totales > 1.5]) * 100
    over_2_5 = np.sum(matriz_prob[goles_totales > 2.5]) * 100
    ambos_anotan = np.sum(matriz_prob[1:, 1:]) * 100
    porteria_cero_a = np.sum(matriz_prob[:, 0]) * 100
    porteria_cero_b = np.sum(matriz_prob[0, :]) * 100

    # Top 10 Marcadores Exactos
    flat_indices = np.argsort(matriz_prob.flatten())[::-1][:10]
    top_marcadores = [np.unravel_index(idx, matriz_prob.shape) for idx in flat_indices]

    # Impresión del reporte analítico
    print(f"\n--- RADIOGRAFÍA DEL PARTIDO: {escenario} ---")
    
    print("\nRESULTADO EN 90 MINUTOS (1X2):")
    print(f"   [1] Gana {equipo_a}: {victorias_a:.1f}%")
    print(f"   [X] Empate: {empates:.1f}%")
    print(f"   [2] Gana {equipo_b}: {victorias_b:.1f}%")

    if es_eliminatoria:
        prob_penal_a, prob_penal_b = obtener_prob_penales(equipo_a, equipo_b)
        avanza_a = victorias_a + (empates * prob_penal_a)
        avanza_b = victorias_b + (empates * prob_penal_b)
        
        print("\nDEFINICIÓN EN ELIMINATORIA (INCLUYE PRÓRROGA/PENALES):")
        print(f"   Efectividad Histórica Penales: {equipo_a} ({prob_penal_a*100:.1f}%) vs {equipo_b} ({prob_penal_b*100:.1f}%)")
        print(f"   PROBABILIDAD REAL DE CLASIFICAR:")
        print(f"      {equipo_a} Avanza: {avanza_a:.1f}%")
        print(f"      {equipo_b} Avanza: {avanza_b:.1f}%")
    else:
        avanza_a = victorias_a
        avanza_b = victorias_b

    print("\nMERCADO DE GOLES (90 Min):")
    print(f"   Más de 1.5 Goles: {over_1_5:.1f}%")
    print(f"   Más de 2.5 Goles: {over_2_5:.1f}%")
    print(f"   Ambos Equipos Anotan: {ambos_anotan:.1f}%")
    print(f"   Portería a Cero ({equipo_a}): {porteria_cero_a:.1f}%")
    print(f"   Portería a Cero ({equipo_b}): {porteria_cero_b:.1f}%")

    print("\nTOP 10 MARCADORES EXACTOS (90 Min):")
    for i, (m_a, m_b) in enumerate(top_marcadores, 1):
        prob = matriz_prob[m_a, m_b] * 100
        print(f"   {i}.  {m_a} - {m_b} (Probabilidad: {prob:.1f}%)")
        
    print("---\n")
    
    return {
        'prob_1': victorias_a,
        'prob_x': empates,
        'prob_2': victorias_b,
        'avanza_a': avanza_a,
        'avanza_b': avanza_b
    }

# Calculadora de Value Betting (Criterio de Kelly)
def analizar_apuesta(prob_modelo_pct, cuota_casino, bankroll_total, fraccion_kelly=0.5):
    print("\n--- ANÁLISIS FINANCIERO (VALUE BETTING) ---")
    
    prob = prob_modelo_pct / 100
    prob_fracaso = 1 - prob
    cuota_justa = 1 / prob if prob > 0 else 0
    
    print(f"Probabilidad de tu modelo: {prob_modelo_pct:.1f}%")
    print(f"Cuota Justa (Fair Odd): {cuota_justa:.2f}")
    print(f"Cuota de la Casa de Apuestas: {cuota_casino}")
    
    ev = (prob * cuota_casino) - 1
    
    if ev > 0:
        print("\nVALUE BET DETECTADA")
        print(f"Ventaja matemática (Edge): +{ev*100:.1f}%")
        
        b = cuota_casino - 1
        kelly_base = ((b * prob) - prob_fracaso) / b
        apuesta_pct = kelly_base * fraccion_kelly
        apuesta_dinero = bankroll_total * apuesta_pct
        
        print(f"\nRecomendación de Gestión (Kelly {fraccion_kelly}x):")
        print(f"Destinar el {apuesta_pct*100:.1f}% del bankroll.")
        print(f"APOSTAR EXACTAMENTE: ${apuesta_dinero:.2f}")
    else:
        print("\nSIN VALOR MATEMÁTICO")
        print(f"Desventaja (Edge negativo): {ev*100:.1f}%")
        print("El casino tiene la ventaja. NO APOSTAR.")
    print("---")

# Ejecución del escáner individual
resultados = analizar_partido_individual('Portugal', 'Croatia', es_eliminatoria=True)

if resultados:
    analizar_apuesta(
        prob_modelo_pct=resultados['avanza_a'], 
        cuota_casino=1.85, 
        bankroll_total=500 
    )