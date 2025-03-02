"""Librerias generales"""
import pandas as pd
import numpy as np

"""Librerias de visualizacion"""
import matplotlib.pyplot as plt
import seaborn as sns

"""Librerias para manejar o importar archivos y directorios"""
import os
import pickle
import sys

"""Librerias de sklearn"""
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV




""" 
---------------------------------- Funciones de visualizacion ----------------------------------
"""



def matriz_de_confusion(
    directorio_modelo,
    conf_matrix):
    print("Matriz de Confusión:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='viridis', fmt='d', cbar=False,
                xticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'],
                yticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'])
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')

    ruta=os.path.join(directorio_modelo, "matriz_de_confusion.png")
    plt.savefig(ruta)


def lineplot_partido_entero_pkl(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    directorio_modelo,  # directorio donde está el modelo
    gol_n_ticks,
    custom_match_id,  # id del partido a analizar en la sección de análisis puntual de todos los momentos de un partido
    best_model
    ):
    
    
    # Seleccionar un partido específico
    match_id_sample = custom_match_id  # Cambia esto al ID del partido que deseas graficar
    partido_data = data_posiciones_filtered_original[data_posiciones_filtered_original['match_id'] == match_id_sample]

    # Crear un DataFrame con los momentos del partido
    momentos_partido = partido_data[['match_time']].drop_duplicates().sort_values(by='match_time')

    # Preparar los datos para la predicción
    X_partido = []
    indices_partido = []

    for match_time in momentos_partido['match_time']:
        subset = partido_data[partido_data['match_time'] == match_time]
        if len(subset) == 7:  # Asegurarse de tener las 7 filas
            input_data = subset[['x', 'y', 'velocity_x', 'velocity_y', 'team']].values.flatten()
            X_partido.append(input_data)
            indices_partido.append(match_time)

    X_partido = np.array(X_partido)

    # Predecir las probabilidades para cada momento del partido
    probabilidades_partido = best_model.predict_proba(X_partido)

    # Convertir las probabilidades a un DataFrame
    probabilidades_partido_df = pd.DataFrame(probabilidades_partido, columns=['%Blue', '%Red', '%None'])
    probabilidades_partido_df['match_time'] = indices_partido

    # Graficar las predicciones a lo largo del tiempo
    plt.figure(figsize=(12, 6))

    # Graficar las probabilidades de cada clase
    plt.plot(probabilidades_partido_df['match_time'], probabilidades_partido_df['%Blue'], label='%Blue', color='blue')
    plt.plot(probabilidades_partido_df['match_time'], probabilidades_partido_df['%Red'], label='%Red', color='red')
    plt.plot(probabilidades_partido_df['match_time'], probabilidades_partido_df['%None'], label='%None', color='gray')

    # Añadir etiquetas y título
    plt.xlabel('Tiempo del Partido')
    plt.ylabel('Probabilidad')
    plt.title(f'Predicciones del Modelo para el Partido {match_id_sample}')
    plt.legend(loc='upper right')
    plt.grid()

    # Guardar el gráfico
    ruta = os.path.join(directorio_modelo, f"lineplot_predicciones_partido_{match_id_sample}.png")
    plt.savefig(ruta)

    # Mostrar el gráfico


def partido_entero_pkl(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    directorio_modelo,  # directorio donde está el modelo
    gol_n_ticks,
    custom_match_id,  # id del partido a analizar en la sección de análisis puntual de todos los momentos de un partido
    best_model,
    tipo_de_modelo
    ):
    
    print(best_model)
    
    # Creación de la clave combinada
    data_posiciones_grouped_filtered_labeled['match_time'] = data_posiciones_grouped_filtered_labeled['match_id'].astype(str) + '_' + data_posiciones_grouped_filtered_labeled['time'].astype(str)
    data_posiciones_filtered['match_time'] = data_posiciones_filtered['match_id'].astype(str) + '_' + data_posiciones_filtered['time'].astype(str)
    
    
    # Filtrar el partido 'custom_match_id'
    match_id_seleccionado = custom_match_id
    partido_custom = data_posiciones_filtered_original[data_posiciones_filtered_original['match_id'] == match_id_seleccionado]
    #print('partido_custom:\n', partido_custom)
    
    
    # Obtener todos los momentos (match_time) únicos del partido
    momentos_partido_custom = partido_custom[['match_time']].drop_duplicates().sort_values(by='match_time')
    #print('momentos_partido_custom:\n', momentos_partido_custom)
    
    
    # Construir los vectores de características (X) y etiquetas (y)
    X = []
    y = []
    indices_partido_custom = []
    data = data_posiciones_grouped_filtered_labeled[data_posiciones_grouped_filtered_labeled['match_id'] == custom_match_id]

    for row in data.itertuples():
        match_time = row.match_time
        gol_N_ticks = getattr(row, gol_n_ticks)
        
        # Seleccionar las 7 filas correspondientes
        subset = data_posiciones_filtered[data_posiciones_filtered['match_time'] == match_time]
        if len(subset) == 7:  # Asegurarse de tener las 7 filas
            input_data = subset[['x', 'y', 'velocity_x', 'velocity_y', 'team']].values.flatten()
            X.append(input_data)
            y.append(gol_N_ticks)
            indices_partido_custom.append(match_time)
        else:
            print(f"Advertencia: {match_time} no tiene 7 filas.")

    
    
    # Obtener el scaler del modelo cargado
    if tipo_de_modelo == 'SVC':
        scaler_path = os.path.join(directorio_modelo, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler_cargado = pickle.load(f)

        # Escalar los datos de test con el scaler cargado
        X_partido_custom_scaled = scaler_cargado.transform(X)
        
    else:
        X_partido_custom_scaled = np.array(X)
    
    #print('X_partido_custom_scaled:\n', X_partido_custom_scaled)
    print(type(X))
    print(X_partido_custom_scaled.shape)  # Si es un array de NumPy, muestra las dimensiones
    print(X_partido_custom_scaled.dtype)  # Si es un array de NumPy, muestra el tipo de datos

    

    # Predecir probabilidades para cada momento del partido
    X_partido_custom_scaled = np.array(X)
    
    
    
    probabilidades_partido_custom = best_model.predict_proba(X_partido_custom_scaled)
    print('probabilidades_partido_custom:\n', probabilidades_partido_custom)
    print(type(probabilidades_partido_custom))
    
    
    # Obtener las etiquetas predichas
    y_pred = np.argmax(probabilidades_partido_custom, axis=1)

    # Obtener las etiquetas reales
    etiquetas_reales = []
    for match_time in indices_partido_custom:
        etiqueta_real = data_posiciones_grouped_filtered_labeled[
            data_posiciones_grouped_filtered_labeled['match_time'] == match_time
        ][gol_n_ticks].values[0]
        etiquetas_reales.append(etiqueta_real)

    # Crear DataFrame con resultados
    resultados_partido_custom = pd.DataFrame({
        'match_time': indices_partido_custom,
        'valor_real': etiquetas_reales,
        '%Blue': probabilidades_partido_custom[:, 0],
        '%Red': probabilidades_partido_custom[:, 1],
        '%None': probabilidades_partido_custom[:, 2]
    })

    # Formatear porcentajes
    resultados_partido_custom['%Blue'] = resultados_partido_custom['%Blue'].apply(lambda x: f"{x * 100:.2f}%")
    resultados_partido_custom['%Red'] = resultados_partido_custom['%Red'].apply(lambda x: f"{x * 100:.2f}%")
    resultados_partido_custom['%None'] = resultados_partido_custom['%None'].apply(lambda x: f"{x * 100:.2f}%")

    #print('resultados_partido_custom:\n', resultados_partido_custom)
    
    

   
   
    # Crear subplots dinámicos según el número de momentos
    n_momentos = len(indices_partido_custom)
    n_cols = 2
    n_rows = (n_momentos + n_cols - 1) // n_cols  # Cálculo dinámico de filas

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.ravel()

    for i, match_time in enumerate(indices_partido_custom):
        ax = axs[i]
        # Filtrar datos para el momento actual
        moment_data = data_posiciones_filtered_original[data_posiciones_filtered_original['match_time'] == match_time]
    
        # Graficar el campo de juego
        ax.imshow(background_image, extent=[-605, 605, -255, 255], aspect='auto')
            
        # Asignar colores y tamaños según el equipo
        colors = moment_data['team'].astype(str).map({'0': 'white', '1': 'red', '2': 'blue'})
        sizes = moment_data['team'].astype(str).map({'0': 25, '1': 100, '2': 100})
           
        # Graficar posiciones
        scatter = ax.scatter(moment_data['x'], moment_data['y'], c=colors, s=sizes, edgecolors='black')
            
        # Agregar flechas de velocidad
        for _, row in moment_data.iterrows():
            ax.quiver(
                row['x'], row['y'],
                row['velocity_x'], row['velocity_y'],
                angles='xy', scale_units='xy', scale=2, 
                color='#242424', width=0.004
            )
            
        # Obtener valores reales y predichos
        fila_resultado = resultados_partido_custom[resultados_partido_custom['match_time'] == match_time].iloc[0]
        valor_real = fila_resultado['valor_real']
        pred_blue = fila_resultado['%Blue']
        pred_red = fila_resultado['%Red']
        pred_none = fila_resultado['%None']
            
        # Configurar título
        ax.set_title(
            f'Partido {match_id_seleccionado} - Momento: {match_time}\n'
            f'Pred: Blue {pred_blue}, Red {pred_red}, None {pred_none}\n'
            f'Real: {valor_real}'
        )
        ax.set_xlim(-605, 605)
        ax.set_ylim(-255, 255)
        ax.axis('off')

    # Ocultar subplots vacíos si los hay
    for j in range(i + 1, n_rows * n_cols):
        axs[j].axis('off')

    plt.tight_layout()

    # Guardar el gráfico
    ruta = os.path.join(directorio_modelo, f"predicciones_partido_{match_id_seleccionado}.png")
    plt.savefig(ruta)
    
    return
    
    
    
    
    
    

    