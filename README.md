# 🏆 World Cup 2026 Stochastic Predictor

Motor predictivo de Inteligencia Artificial desarrollado en Python para simular y proyectar los resultados de la Copa del Mundo 2026. El modelo abandona las heurísticas tradicionales en favor de una arquitectura estocástica rigurosa basada en estadística bayesiana, distribuciones bivariadas y factores de decaimiento temporal.

## 🧠 Arquitectura Matemática

El núcleo del ecosistema se basa en los siguientes principios estadísticos:
* **Matriz de Poisson Bivariada:** Cálculo de probabilidades de Goles Esperados (xG) aislando la fuerza de ataque y defensa neutral de cada selección frente al promedio global histórico de la FIFA.
* **Factor de Corrección Dixon-Coles (ρ = -0.13):** Ajuste algorítmico crítico para corregir la subestimación de empates (0-0, 1-1) inherente a los modelos de Poisson puros en el fútbol de selecciones.
* **Time Decay (Decaimiento Exponencial):** Ponderación dinámica que otorga mayor relevancia matemática a los resultados recientes para capturar el *momentum* real de las selecciones, penalizando datos antiguos.
* **Localía Dinámica (HFA):** Ajuste de ventaja de campo (Home Field Advantage) específico inyectado a los anfitriones (México, Estados Unidos, Canadá).

## 📂 Ecosistema de Módulos

El proyecto está modularizado en 5 scripts ejecutables, diseñados para diferentes etapas del análisis y consumo de datos:

1. **`tournament_simulator.py`**: Motor de rastreo en vivo (*Live Tracker*). Lee resultados reales dinámicamente desde la base de datos y simula el resto del torneo en tiempo real, adaptándose a llaves predefinidas.
2. **`official_simulator.py`**: Simulador puro desde cero. Ejecuta el torneo completo utilizando un algoritmo de *Backtracking* para realizar la asignación perfecta de los mejores terceros lugares, cumpliendo estrictamente con el reglamento de la FIFA.
3. **`knockout_scanner.py`**: Escáner de auditoría táctica de terminal. Analiza partidos individuales de eliminación directa, calculando probabilidades de prórroga, cruzando la efectividad histórica de penales y arrojando el Top 10 de marcadores exactos.
4. **`group_stage_scanner.py`**: Analizador financiero para Fase de Grupos. Escanea cruces y compara las probabilidades del modelo contra las cuotas de las casas de apuestas para detectar el *Edge* matemático y recomendar gestión de banca mediante el **Criterio de Kelly**.
5. **`match_scanner.py`**: Módulo base refactorizado. Actúa como una función pura que devuelve objetos JSON/Diccionarios, diseñada específicamente para ser consumida por un *Frontend* o una API web en el futuro.

## 📊 Fuentes de Datos (Data Pipeline)
* `results.csv`: Histórico de partidos internacionales oficiales (limpio de amistosos).
* `shootouts.csv`: Registro histórico de tandas de penales para la resolución de empates en fases KO.
* `ranking_fifa.csv`: Puntuación global utilizada como factor multiplicador de fuerza base.

## 🎯 Precisión y Validación (Edge)
El objetivo de este modelo no es la clarividencia determinista, sino la detección de ineficiencias en los mercados de pronósticos (*Value Betting*). Al aislar la fuerza neutral y aplicar el factor Dixon-Coles, el modelo busca de manera consistente un **Valor Esperado Positivo (+EV)**. Las métricas de éxito se evalúan iterando el Criterio de Kelly sobre el diferencial entre la cuota justa del modelo (Fair Odd) y la cuota del mercado comercial.

## 🚀 Stack Tecnológico
* **Lenguaje:** Python 3.x
* **Librerías Core:** `pandas` (Manipulación de DataFrames), `numpy` (Cálculo matricial), `scipy.stats` (Funciones de masa de probabilidad estocástica).3.x
* **Librerías Core:** `pandas` (Manipulación de DataFrames), `numpy` (Cálculo matricial), `scipy.stats` (Funciones de masa de probabilidad).
