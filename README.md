# World Cup Stochastic Predictor (2026)

Motor predictivo de Inteligencia Artificial desarrollado en Python para simular y proyectar los resultados de la Copa del Mundo 2026. El modelo abandona las heurísticas tradicionales en favor de una arquitectura estocástica rigurosa basada en estadística bayesiana, distribuciones bivariadas y factores de decaimiento.

## Arquitectura Matemática

El núcleo del ecosistema se basa en los siguientes principios:
* **Matriz de Poisson Bivariada:** Cálculo de probabilidades de Goles Esperados (xG) aislando la fuerza de ataque y defensa neutral de cada selección frente al promedio global de la FIFA.
* **Factor de Corrección Dixon-Coles (ρ = -0.13):** Ajuste algorítmico crítico para corregir la subestimación de empates (0-0, 1-1) inherente a los modelos de Poisson puros en el fútbol.
* **Time Decay (Decaimiento Exponencial):** Ponderación dinámica que otorga mayor relevancia a los resultados recientes para capturar el *momentum* real de las selecciones.
* **Localía Dinámica (HFA):** Ajuste de ventaja de campo (Home Field Advantage) específico para los anfitriones (México, Estados Unidos, Canadá).

## Ecosistema de Módulos

El proyecto está modularizado en 5 scripts ejecutables diseñados para diferentes etapas del análisis:

1. **`tournament_simulator.py`**: Rastreador en vivo. Lee resultados reales dinámicamente y simula el resto del torneo en tiempo real.
2. **`official_simulator.py`**: Simulador puro desde cero. Ejecuta el torneo completo utilizando un algoritmo de *Backtracking* para realizar la asignación perfecta de los mejores terceros lugares según el reglamento oficial de la FIFA.
3. **`fc26_live_tracker.py`**: Interceptor especializado. Realiza un *override* de las reglas de emparejamiento oficiales para simular escenarios alternativos de Fase Eliminatoria (Modo Carrera).
4. **`knockout_scanner.py`**: Escáner de auditoría táctica. Analiza partidos de eliminación directa arrojando probabilidades de prórroga, efectividad histórica de penales y Top 3 de marcadores exactos.
5. **`value_bet_calculator.py`**: Analizador financiero. Escanea cruces de Fase de Grupos cruzando las probabilidades del modelo contra cuotas de casas de apuestas para detectar el *Edge* matemático y recomendar gestión de banca mediante el Criterio de Kelly.

## Stack Tecnológico
* **Lenguaje:** Python 3.x
* **Librerías Core:** `pandas` (Manipulación de DataFrames), `numpy` (Cálculo matricial), `scipy.stats` (Funciones de masa de probabilidad).
