import pandas as pd
import numpy as np
import itertools
from scipy.stats import poisson

print("Iniciando Simulador MASTER LIVE (Dixon-Coles + Escáner Top 3 + Lectura Mundial Real)...")

# Fase 0: Lectura del Ranking FIFA
try:
    df_ranking = pd.read_csv('ranking_fifa.csv')
    df_ranking = df_ranking.dropna(subset=['team', 'points'])
    puntos_fifa = dict(zip(df_ranking['team'], df_ranking['points']))
except FileNotFoundError:
    raise Exception("ERROR CRÍTICO: No se encontró 'ranking_fifa.csv'.")

# Fase 1: Ingesta y ajuste de goles por rival
try:
    df_resultados = pd.read_csv('results.csv')
    df_resultados['date'] = pd.to_datetime(df_resultados['date'], format='mixed', errors='coerce')
    # Limpieza preventiva de espacios vacíos
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

# Fase 1.5: Ingesta de historial de penales (Con auto-limpieza)
try:
    df_penales = pd.read_csv('shootouts.csv')
    df_penales['home_team'] = df_penales['home_team'].astype(str).str.strip()
    df_penales['away_team'] = df_penales['away_team'].astype(str).str.strip()
    df_penales['winner'] = df_penales['winner'].astype(str).str.strip()
    
    participaciones = df_penales['home_team'].value_counts().add(df_penales['away_team'].value_counts(), fill_value=0)
    victorias_penales = df_penales['winner'].value_counts()
    efectividad_penales = (victorias_penales / participaciones).fillna(0).to_dict()
except FileNotFoundError:
    print("ADVERTENCIA: No se encontró 'shootouts.csv'. Se asumirá una probabilidad de 50/50 en penales.")
    efectividad_penales = {}
    df_penales = pd.DataFrame()

def obtener_prob_penales(equipo_a, equipo_b):
    efectividad_a = efectividad_penales.get(equipo_a, 0.5)
    efectividad_b = efectividad_penales.get(equipo_b, 0.5)
    prob_ganar_a = (efectividad_a + (1 - efectividad_b)) / 2
    prob_ganar_b = 1 - prob_ganar_a
    return prob_ganar_a, prob_ganar_b

# Fase 2: Motor de ponderación matemática (Time Decay)
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

# Fase 3: Grupos oficiales y localía
grupos_mundial = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}

anfitriones = ['Mexico', 'United States', 'Canada']
pesos_localia = {'Mexico': 1.20, 'United States': 1.05, 'Canada': 1.03}

def obtener_bono_local(equipo):
    return pesos_localia.get(equipo, 1.10)

# Extracción de resultados reales (Filtrando directamente por el año 2026 para evadir el error de meses invertidos)
df_mundial_2026 = df_resultados[(df_resultados['tournament'] == 'FIFA World Cup') & (df_resultados['date'].dt.year >= 2026)].copy()
jugados_mundial = df_mundial_2026.dropna(subset=['home_score', 'away_score'])

resultados_reales_mundial = {}
for _, row in jugados_mundial.iterrows():
    resultados_reales_mundial[(row['home_team'], row['away_team'])] = (row['home_score'], row['away_score'])

# Fase 4: Cerebro matricial predictivo e interceptor inteligente
def simular_partido(equipo_a, equipo_b, es_eliminatoria=False, ronda=""):
    
    if es_eliminatoria:
        partido_real = False
        goles_a = goles_b = 0
        ganador_penales_real = None
        
        # 1. Buscamos en la matriz de resultados reales de 2026
        if (equipo_a, equipo_b) in resultados_reales_mundial:
            goles_a, goles_b = resultados_reales_mundial[(equipo_a, equipo_b)]
            partido_real = True
        elif (equipo_b, equipo_a) in resultados_reales_mundial:
            goles_b, goles_a = resultados_reales_mundial[(equipo_b, equipo_a)]
            partido_real = True

        # 2. Mecanismo de rescate para tanda de penales
        if not df_penales.empty:
            mask = (((df_penales['home_team'] == equipo_a) & (df_penales['away_team'] == equipo_b)) | 
                    ((df_penales['home_team'] == equipo_b) & (df_penales['away_team'] == equipo_a)))
            
            try:
                fechas = pd.to_datetime(df_penales['date'], errors='coerce')
                mask = mask & (fechas.dt.year >= 2026)
            except:
                pass
            
            hist_penales_2026 = df_penales[mask]
            
            if not hist_penales_2026.empty:
                ganador_penales_real = hist_penales_2026.iloc[-1]['winner']
                
                # Si el resultado se omitió en results.csv pero está en shootouts.csv, lo recuperamos
                if not partido_real:
                    partido_real = True
                    goles_a = goles_b = 0

        # FAST-FORWARD: Si el partido ya ocurrió
        if partido_real:
            if goles_a > goles_b:
                ganador, perdedor = equipo_a, equipo_b
                mensaje = f"{equipo_a} {int(goles_a)} - {int(goles_b)} {equipo_b}"
            elif goles_b > goles_a:
                ganador, perdedor = equipo_b, equipo_a
                mensaje = f"{equipo_a} {int(goles_a)} - {int(goles_b)} {equipo_b}"
            else:
                if ganador_penales_real:
                    ganador = ganador_penales_real
                    perdedor = equipo_b if ganador == equipo_a else equipo_a
                    mensaje = f"{equipo_a} {int(goles_a)} - {int(goles_b)} {equipo_b} | Penales: {ganador}"
                else:
                    prob_a, prob_b = obtener_prob_penales(equipo_a, equipo_b)
                    ganador = equipo_a if prob_a > prob_b else equipo_b
                    perdedor = equipo_b if ganador == equipo_a else equipo_a
                    mensaje = f"{equipo_a} {int(goles_a)} - {int(goles_b)} {equipo_b} | SIMULACIÓN PENALES: {ganador}"
            
            print(f"[ ✓ Completado ] {ronda}: {mensaje} -> Avanzó: {ganador}")
            return ganador, perdedor

    # SIMULACIÓN: El partido no ha ocurrido, se activa el motor matemático
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
    elif equipo_b in anfitriones and equipo_a not in anfitriones:
        base_a, base_b = promedio_goles_visitante, promedio_goles_local
        bono_a, bono_b = 1.0, obtener_bono_local(equipo_b)
    else:
        base_a = base_b = promedio_goles_neutral
        bono_a = bono_b = 1.0

    lambda_a = stats_a['Ataque_Neutral'] * stats_b['Defensa_Neutral'] * base_a * bono_a
    lambda_b = stats_b['Ataque_Neutral'] * stats_a['Defensa_Neutral'] * base_b * bono_b

    pts_a = puntos_fifa.get(equipo_a, 1400)
    pts_b = puntos_fifa.get(equipo_b, 1400)
    lambda_a *= (pts_a / pts_b) ** 2
    lambda_b *= (pts_b / pts_a) ** 2

    max_goles = 10
    x, y = np.arange(max_goles), np.arange(max_goles)
    matriz_prob = np.outer(poisson.pmf(x, lambda_a), poisson.pmf(y, lambda_b))
    
    rho = -0.13
    matriz_prob[0,0] *= (1 - lambda_a * lambda_b * rho)
    matriz_prob[1,0] *= (1 + lambda_a * rho)
    matriz_prob[0,1] *= (1 + lambda_b * rho)
    matriz_prob[1,1] *= (1 - rho)
    matriz_prob /= np.sum(matriz_prob)
    
    prob_a = np.tril(matriz_prob, -1).sum()
    prob_emp = np.trace(matriz_prob)
    prob_b = np.triu(matriz_prob, 1).sum()

    if es_eliminatoria:
        goles_totales = np.add.outer(x, y)
        over_1_5 = np.sum(matriz_prob[goles_totales > 1.5]) * 100
        over_2_5 = np.sum(matriz_prob[goles_totales > 2.5]) * 100
        ambos_anotan = np.sum(matriz_prob[1:, 1:]) * 100
        
        flat_indices = np.argsort(matriz_prob.flatten())[::-1][:3]
        top_marcadores = [np.unravel_index(idx, matriz_prob.shape) for idx in flat_indices]

        prob_penal_a, prob_penal_b = obtener_prob_penales(equipo_a, equipo_b)
        avanza_a = prob_a + (prob_emp * prob_penal_a)
        avanza_b = prob_b + (prob_emp * prob_penal_b)
        
        ganador = equipo_a if avanza_a > avanza_b else equipo_b
        perdedor = equipo_b if avanza_a > avanza_b else equipo_a
        
        print(f"\n[ SIMULACIÓN PENDIENTE: {ronda} | {equipo_a} vs {equipo_b} ]")
        print(f"90 Min: [1] {prob_a*100:.1f}% | [X] {prob_emp*100:.1f}% | [2] {prob_b*100:.1f}%")
        print(f"Clasificar: {equipo_a} ({avanza_a*100:.1f}%) vs {equipo_b} ({avanza_b*100:.1f}%)")
        print(f"Avanza a la siguiente ronda: {ganador}")
        print(f"Goles: >1.5 ({over_1_5:.1f}%) | >2.5 ({over_2_5:.1f}%) | A/A ({ambos_anotan:.1f}%)")
        print("Top 3 Marcadores Exactos:")
        for i, (m_a, m_b) in enumerate(top_marcadores, 1):
            prob = matriz_prob[m_a, m_b] * 100
            print(f"   {i}. {m_a} - {m_b} (Prob: {prob:.1f}%)")
        print("---")
        
        return ganador, perdedor
    else:
        return prob_a, prob_emp, prob_b, lambda_a, lambda_b

# Fase de grupos
def jugar_fase_grupos():
    tabla_posiciones = {equipo: {'xPts': 0.0, 'xDG': 0.0, 'Grupo': grupo} for grupo, equipos in grupos_mundial.items() for equipo in equipos}

    for letra, equipos in grupos_mundial.items():
        partidos = list(itertools.combinations(equipos, 2))
        for eq_a, eq_b in partidos:
            partido_jugado = False
            goles_a, goles_b = 0, 0
            if (eq_a, eq_b) in resultados_reales_mundial:
                goles_a, goles_b = resultados_reales_mundial[(eq_a, eq_b)]
                partido_jugado = True
            elif (eq_b, eq_a) in resultados_reales_mundial:
                goles_b, goles_a = resultados_reales_mundial[(eq_b, eq_a)]
                partido_jugado = True
                
            if partido_jugado:
                pts_a = 3 if goles_a > goles_b else (1 if goles_a == goles_b else 0)
                pts_b = 3 if goles_b > goles_a else (1 if goles_a == goles_b else 0)
                tabla_posiciones[eq_a]['xPts'] += pts_a
                tabla_posiciones[eq_b]['xPts'] += pts_b
                tabla_posiciones[eq_a]['xDG'] += (goles_a - goles_b)
                tabla_posiciones[eq_b]['xDG'] += (goles_b - goles_a)
            else:
                prob_a, prob_emp, prob_b, xg_a, xg_b = simular_partido(eq_a, eq_b, es_eliminatoria=False)
                tabla_posiciones[eq_a]['xPts'] += (prob_a * 3) + (prob_emp * 1)
                tabla_posiciones[eq_b]['xPts'] += (prob_b * 3) + (prob_emp * 1)
                tabla_posiciones[eq_a]['xDG'] += (xg_a - xg_b)
                tabla_posiciones[eq_b]['xDG'] += (xg_b - xg_a)

    df_posiciones = pd.DataFrame.from_dict(tabla_posiciones, orient='index')
    primeros, segundos, terceros = {}, {}, {}

    for letra in grupos_mundial.keys():
        grupo = df_posiciones[df_posiciones['Grupo'] == letra].sort_values(by=['xPts', 'xDG'], ascending=[False, False])
        primeros[letra] = grupo.index[0]
        segundos[letra] = grupo.index[1]
        terceros[letra] = grupo.index[2]

    return primeros, segundos, terceros

# Bracket del Mundial
def simular_eliminatorias_reales():
    jugar_fase_grupos()

    print("\n--- INICIANDO ELIMINATORIAS (CUADRO REAL) ---")
    
    w1, _ = simular_partido('Germany', 'Paraguay', es_eliminatoria=True, ronda="16vos M1")
    w2, _ = simular_partido('France', 'Sweden', es_eliminatoria=True, ronda="16vos M2")
    w3, _ = simular_partido('Canada', 'South Africa', es_eliminatoria=True, ronda="16vos M3")
    w4, _ = simular_partido('Netherlands', 'Morocco', es_eliminatoria=True, ronda="16vos M4")
    w5, _ = simular_partido('Portugal', 'Croatia', es_eliminatoria=True, ronda="16vos M5")
    w6, _ = simular_partido('Spain', 'Austria', es_eliminatoria=True, ronda="16vos M6")
    w7, _ = simular_partido('United States', 'Bosnia and Herzegovina', es_eliminatoria=True, ronda="16vos M7")
    w8, _ = simular_partido('Belgium', 'Senegal', es_eliminatoria=True, ronda="16vos M8")
    w9, _ = simular_partido('Brazil', 'Japan', es_eliminatoria=True, ronda="16vos M9")
    w10, _ = simular_partido('Ivory Coast', 'Norway', es_eliminatoria=True, ronda="16vos M10")
    w11, _ = simular_partido('Mexico', 'Ecuador', es_eliminatoria=True, ronda="16vos M11")
    w12, _ = simular_partido('England', 'DR Congo', es_eliminatoria=True, ronda="16vos M12")
    w13, _ = simular_partido('Argentina', 'Cape Verde', es_eliminatoria=True, ronda="16vos M13")
    w14, _ = simular_partido('Australia', 'Egypt', es_eliminatoria=True, ronda="16vos M14")
    w15, _ = simular_partido('Switzerland', 'Algeria', es_eliminatoria=True, ronda="16vos M15")
    w16, _ = simular_partido('Colombia', 'Ghana', es_eliminatoria=True, ronda="16vos M16")

    print("\n--- OCTAVOS DE FINAL ---")
    o1, _ = simular_partido(w1, w2, es_eliminatoria=True, ronda="Octavos O1")
    o2, _ = simular_partido(w3, w4, es_eliminatoria=True, ronda="Octavos O2")
    o3, _ = simular_partido(w5, w6, es_eliminatoria=True, ronda="Octavos O3")
    o4, _ = simular_partido(w7, w8, es_eliminatoria=True, ronda="Octavos O4")
    o5, _ = simular_partido(w9, w10, es_eliminatoria=True, ronda="Octavos O5")
    o6, _ = simular_partido(w11, w12, es_eliminatoria=True, ronda="Octavos O6")
    o7, _ = simular_partido(w13, w14, es_eliminatoria=True, ronda="Octavos O7")
    o8, _ = simular_partido(w15, w16, es_eliminatoria=True, ronda="Octavos O8")

    print("\n--- CUARTOS DE FINAL ---")
    c1, _ = simular_partido(o1, o2, es_eliminatoria=True, ronda="Cuartos C1")
    c2, _ = simular_partido(o3, o4, es_eliminatoria=True, ronda="Cuartos C2")
    c3, _ = simular_partido(o5, o6, es_eliminatoria=True, ronda="Cuartos C3")
    c4, _ = simular_partido(o7, o8, es_eliminatoria=True, ronda="Cuartos C4")

    print("\n--- SEMIFINALES ---")
    s1, _ = simular_partido(c1, c2, es_eliminatoria=True, ronda="Semifinal S1")
    s2, _ = simular_partido(c3, c4, es_eliminatoria=True, ronda="Semifinal S2")

    print("\n--- LA GRAN FINAL ---")
    campeon, _ = simular_partido(s1, s2, es_eliminatoria=True, ronda="GRAN FINAL")
    print(f"\nCAMPEÓN DEL MUNDO: {campeon.upper()}\n")

if __name__ == "__main__":
    simular_eliminatorias_reales()