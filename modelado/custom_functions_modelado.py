from sklearn.utils import resample
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def modelar_red_neuronal_001(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    gol_n_ticks
    ):
    
    # ------------------------------ PREPARACION Y BALANCEO DE CLASES ----------------------------------------------
    
    
    # Creación de la clave combinada
    data_posiciones_grouped_filtered_labeled['match_time'] = data_posiciones_grouped_filtered_labeled['match_id'].astype(str) + '_' + data_posiciones_grouped_filtered_labeled['time'].astype(str)
    data_posiciones_filtered['match_time'] = data_posiciones_filtered['match_id'].astype(str) + '_' + data_posiciones_filtered['time'].astype(str)
    
    # Separación de clases para balanceo
    indices_nada = data_posiciones_grouped_filtered_labeled[data_posiciones_grouped_filtered_labeled[gol_n_ticks] == 'none'].index
    indices_gol_rojo = data_posiciones_grouped_filtered_labeled[data_posiciones_grouped_filtered_labeled[gol_n_ticks] == 'Red'].index
    indices_gol_azul = data_posiciones_grouped_filtered_labeled[data_posiciones_grouped_filtered_labeled[gol_n_ticks] == 'Blue'].index

    # Determinar la cantidad a muestrear para balancear
    n = 2 # proporcion de None vs Red(Blue)
    num_muestras_gol = min(len(indices_gol_rojo), len(indices_gol_azul), len(indices_nada)/n)
    num_muestras_nada = num_muestras_gol * n

    # Muestreo balanceado de combinaciones match_time
    indices_nada_balanceados = resample(indices_nada, n_samples=num_muestras_nada, replace=False, random_state=random_state)
    indices_gol_rojo_balanceados = resample(indices_gol_rojo, n_samples=num_muestras_gol, replace=False, random_state=random_state)
    indices_gol_azul_balanceados = resample(indices_gol_azul, n_samples=num_muestras_gol, replace=False, random_state=random_state)
    
    
    # Combinar los índices balanceados
    indices_balanceados = np.concatenate([indices_nada_balanceados, indices_gol_rojo_balanceados, indices_gol_azul_balanceados])
    rng  = np.random.default_rng(random_state)
    rng.shuffle(indices_balanceados)

    # Seleccionar los datos balanceados
    data_balanceada = data_posiciones_grouped_filtered_labeled.loc[indices_balanceados]


    # Construir los vectores de características (X) y etiquetas (y)
    X = []
    y = []
    indices_X_y = []

    for row in data_balanceada.itertuples():
        match_time = row.match_time
        gol_N_ticks = getattr(row, gol_n_ticks)
        
        # Seleccionar las 7 filas correspondientes
        subset = data_posiciones_filtered[data_posiciones_filtered['match_time'] == match_time]
        if len(subset) == 7:  # Asegurarse de tener las 7 filas
            input_data = subset[['x', 'y', 'velocity_x', 'velocity_y', 'team']].values.flatten()
            X.append(input_data)
            y.append(gol_N_ticks)
            indices_X_y.append(match_time)
        else:
            print(f"Advertencia: {match_time} no tiene 7 filas.")

    X = np.array(X)
    y = np.array(y)
    indices_X_y = np.array(indices_X_y)
    
    print('')
    print('Algunas datos de X:\n', X[0:2])
    print('')
    print('Algunos datos de y:\n',y[0:2])
    print('')
    print("Distribución balanceada de clases:\n", Counter(y))
    print('')
    print("Algunos indices de X e y:\n", indices_X_y[:5])
    
    # ------------------------- DIVISION DE DATOS DE ENTRENAMIENTO Y TESTEO ----------------------------------------------
    
    # Dividir los datos para el entrenamiento y testeo del modelo
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices_X_y, test_size=0.3, random_state=random_state, stratify=y) # stratify=y hace que se mantenga las proporciones
    
    print('')
    print(Counter(y_train))
    print('')
    print(Counter(y_test))
    print('')
    print("Índices correspondientes a y_test:\n", indices_test[:5])
    
    # --------------------------------- ENTRENAMIENTO DEL MODELO ----------------------------------------------
    
    # Crear el modelo
    modelo = MLPClassifier(alpha=0.001, hidden_layer_sizes=(256, 128, 64), learning_rate_init=0.001, activation='relu', solver='adam', max_iter=500, random_state=random_state)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    
    y_pred = modelo.predict(X_test)
    
    # Dataframe para graficar posteriormente
    resultados = pd.DataFrame({
        'indice_original': indices_test,
        'valor_real': y_test,
        'valor_predicho': y_pred
    })
    print('')
    print('Conteo de predicciones de y:\n', Counter(y_pred))
    print('')
    print('Reporte de la clasificacion:\n', classification_report(y_test, y_pred, target_names=['Gol Blue', 'Gol Red', 'Ningún gol']))
    
    # -------------------------------- MATRIZ DE CONFUSION ----------------------------------------------

    print('')
    print('MATRIZ DE CONFUSION:')
    print('')
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='viridis', fmt='d', cbar=False,
                xticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'],
                yticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'])
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.show()
    
    # ---------------------------------- PROBABILIDADES ----------------------------------------------

    
    # Obtener las probabilidades de las predicciones en el conjunto de prueba
    probabilidades = modelo.predict_proba(X_test)

    # Convertir las probabilidades a un DataFrame
    probabilidades_df = pd.DataFrame(probabilidades, columns=['%Blue', '%Red', '%None'])
    probabilidades_df = probabilidades_df.round(5)

    # Agregar las columnas de valor real
    probabilidades_df['Valor real'] = y_test

    # Reorganizar las columnas
    probabilidades_df = probabilidades_df[['Valor real', '%Blue', '%Red', '%None']]

    # Mostrar todas las filas (en Jupyter Notebook, muestra todas)
    pd.set_option('display.max_rows', None)
    probabilidades_df_sample_10 = probabilidades_df.sample(10, random_state=random_state).reset_index()
    
    
    # ------- GRAFICO DE DISTRIBUCION DE PROBABILIDAD DE CLASES DE 10 MOMENTOS AL AZAR -------------------

    print('')
    print('GRAFICO DE DISTRIBUCION DE PROBABILIDAD DE CLASES DE 10 MOMENTOS AL AZAR:')
    print('')
    # Crear gráfico de barras apiladas horizontal
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear barras apiladas para cada fila
    for i, row in probabilidades_df_sample_10.iterrows():
        ax.barh(i, row['%Blue']*100, color='blue')
        ax.barh(i, row['%Red']*100, left=row['%Blue']*100, color='red')
        ax.barh(i, row['%None']*100, left=(row['%Blue'] + row['%Red'])*100, color='gray')

    # Etiquetas y título
    ax.set_xlabel('Porcentaje de Probabilidades predichas por el modelo')
    ax.set_ylabel('Valor Real')
    ax.set_title('Distribución de Probabilidades de Clases')



    # Asignar las etiquetas del eje Y con los valores de la columna 'Valor real'
    ax.set_yticks(range(len(probabilidades_df_sample_10)))
    ax.set_yticklabels(probabilidades_df_sample_10['Valor real'].values)

    # Mostrar el gráfico
    plt.show()
    
    
    # ------------ GRAFICO DE 10 MOMENTOS AL AZAR Y LAS PREDICCIONES DEL MODELO -------------------
    
    print('')
    print('GRAFICO DE 10 MOMENTOS AL AZAR Y LAS PREDICCIONES DEL MODELO:')
    print('')
   
    probabilidades_df_filtered = probabilidades_df.drop(columns=['Valor real'])
    resultados_concat = pd.concat([resultados, probabilidades_df_filtered], axis=1)

    # Formateo valores para plotear
    resultados_concat['%Blue'] = resultados_concat['%Blue'].apply(lambda x: f"{round(x,4)*100:.2f}")
    resultados_concat['%None'] = resultados_concat['%None'].apply(lambda x: f"{round(x, 4)*100:.2f}")
    resultados_concat['%Red'] = resultados_concat['%Red'].apply(lambda x: f"{round(x, 4)*100:.2f}")
    
    data_posiciones_filtered_original_sample = data_posiciones_filtered_original[data_posiciones_filtered_original['match_time'].isin(indices_test)]
    unique_moments = data_posiciones_filtered_original_sample[['match_time']].drop_duplicates()
    random_moments = unique_moments.sample(n=10, random_state=random_state)

    # Crear subplots
    fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    axs = axs.ravel()  # Aplanar el array de ejes para facilitar el acceso

    # Filtrar y graficar los datos para cada momento seleccionado
    for i, (index, moment) in enumerate(random_moments.iterrows()):
        match_time = moment['match_time']
        
        # Filtrar datos para el momento actual
        moment_data = data_posiciones_filtered_original[(data_posiciones_filtered_original['match_time'] == match_time)]
        
        # Graficar los datos en el subplot correspondiente
        ax = axs[i]
        ax.imshow(background_image, extent=[-605, 605, -255, 255], aspect='auto')
        
        # Asignar colores y tamaños según el equipo
        colors = moment_data['team'].astype(str).map({'0': 'white', '1': 'red', '2': 'blue'})
        sizes = moment_data['team'].astype(str).map({'0': 25, '1': 100, '2': 100})

        scatter = ax.scatter(moment_data['x'], moment_data['y'], c=colors, s=sizes, edgecolors='black')

        # Agregar flechas que indican la velocidad de cada jugador/pelota
        for j, row in moment_data.iterrows():
            x_pos = row['x']
            y_pos = row['y']
            v_x = row['velocity_x']
            v_y = row['velocity_y']
            
            # Calcular la longitud de la flecha (norma de la velocidad)
            speed = np.sqrt(v_x**2 + v_y**2) * 0.001
            
            # La escala de las flechas (ajustar este factor según sea necesario)
            arrow_scale = 2
            
            # Agregar la flecha con 'quiver'
            ax.quiver(x_pos, y_pos, v_x, v_y, angles='xy', scale_units='xy', scale=arrow_scale, color='#242424',
                     width=0.004)

        valor_real = resultados.loc[(resultados['indice_original'] == match_time), 'valor_real'].values[0]
        valor_predicho_red = resultados_concat.loc[(resultados['indice_original'] == match_time), '%Red'].values[0]
        valor_predicho_none = resultados_concat.loc[(resultados['indice_original'] == match_time), '%None'].values[0]
        valor_predicho_blue = resultados_concat.loc[(resultados['indice_original'] == match_time), '%Blue'].values[0]
        
        ax.set_title(f'Match - Time: {match_time}\n Predicted: Red:{valor_predicho_red}%, none:{valor_predicho_none}%, Blue:{valor_predicho_blue}%\n Real: {valor_real}')
        ax.set_xlim([-605, 605])
        ax.set_ylim([-255, 255])
        ax.axis('off')  # Opcional: para ocultar los ejes

    plt.tight_layout()
    plt.show()