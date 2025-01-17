# Importo librerías ----------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.image as mpimg
from sklearn.preprocessing import MinMaxScaler


# Funciones para analisis de datos -------------------------------------------------------------------------

def print_value_counts(df, df_name, cant, columns=None):
    if columns is None:
        columns = df.columns

    for column in columns:
        if column in df.columns:
            print(f"Value counts para {df_name} - Columna: {column}")
            print(df[column].value_counts().head(cant))
            print("\n")
        else:
            print(f"La columna '{column}' no existe en el DataFrame {df_name}.\n")


def plot_ball_heatmaps_for_team_goal(equipo, title_prefix, data_pos, data_grouped):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs = axs.ravel()  # Aplanar para iterar fácilmente

    background_image = mpimg.imread('stadium.PNG') # Mapa de la cancha en formato '.png'
    
    for ticks in range(1, 7):
        column_name = f'gol_{ticks}_ticks'
        title_suffix = f'gol del {equipo} en menos de {ticks} ticks'
        
        # Filtrar data_grouped según la columna correspondiente
        filtered_times = data_grouped[
            data_grouped[column_name] == equipo
        ]['time']
        
        # Filtrar posiciones de la pelota
        ball_data_filtered = data_pos[
            (data_pos['player_id'] == 0) & (data_pos['time'].isin(filtered_times))
        ]
        
        # Crear los mapas de calor
        ax = axs[ticks - 1]
        ax.imshow(background_image, extent=[-605, 605, -255, 255], aspect='auto')
        
        sns.kdeplot(
            x=ball_data_filtered['x'], y=ball_data_filtered['y'], 
            fill=True, cmap='Blues', bw_adjust=0.75, levels=20, thresh=0.1, alpha=0.7, ax=ax
        )
        
        ax.set_xlim([-605, 605])
        ax.set_ylim([-255, 255])
        ax.axis('off')
        ax.set_title(f'{title_prefix}\n{title_suffix}', fontsize=12)
    
    plt.tight_layout()
    plt.show()


# Funciones para el filtrado -------------------------------------------------------------------------------

def analizar_filtrado(nombre_del_filtro, data_filtros, data_posiciones_rows, data_posiciones_filtered_rows, previous_shape_rows):
    print("Cantidad de filas que quedaron luego del filtrado:", data_posiciones_filtered_rows)
    print("Porcentaje de filas que quedan del dataset original:", round((data_posiciones_filtered_rows/data_posiciones_rows*100), 2),"%")
    print("Porcentaje de filas que quedan del dataset filtrado anterior:", round((data_posiciones_filtered_rows/previous_shape_rows*100), 2),"%")
    
     # Guardo la informacion en el dataframe de filtrado
    fila = {
        'filtro': nombre_del_filtro,
        'cant_de_momentos': data_posiciones_filtered_rows,
        '%_respecto_a_original': round((data_posiciones_filtered_rows/data_posiciones_rows*100), 2),
        '%_respecto_a_anterior': round((data_posiciones_filtered_rows/previous_shape_rows*100), 2)
    }
    fila = pd.DataFrame([fila])
    data_filtros =  pd.concat([data_filtros, fila], ignore_index=True)
    return data_filtros

# Funciones para el modelado -------------------------------------------------------------------------------

def normalizar_datos(data):
    # Normalizar las columnas de posiciones y velocidades
    columns_to_normalize = ['x', 'y', 'velocity_x', 'velocity_y']
    scaler = MinMaxScaler()
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    
    # Normalizar la columna del equipo
    data['team'] = data['team'].map({1: -1, 2: 1, 0: 0})
    
    return data