"""Librerias generales"""
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import uniform, loguniform

"""Librerias de visualizacion"""
import seaborn as sns
import matplotlib.pyplot as plt

"""Librerias customizadas"""
import custom_functions_contador as cf_c
import custom_functions_visualizacion as cf_v

"""Librerias para manejar o importar archivos y directorios"""
import os
import pickle
import sys

"""Librerias de sklearn"""
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


"""


    Nota: las librerias Tensorflow y Keras se importan en cada modelo,
    asi como otras mas especificas no importadas anteriormente.
    
    
"""

""" 
---------------------------------- Funciones de modelado y analisis ----------------------------------
"""



def modelar_red_neuronal_001(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    gol_n_ticks,
    #redNeuronalSimple_bool, # Para indicar si se quiere hacer una red neuronal sencilla de prueba
    gridsearchCv_bool # Para indicar si se quiere hacer el gridsearch + cross validation
    ):

    if True:
        
        # ---------------------------- CARGO EL ID DEL MODELO Y CREO SU CARPETA ----------------------------------------
    
        id_modelo = cf_c.leer_contador("contador_de_modelos.txt")
        tipo_de_modelo = "_RedNeuronal_"
        variable_carpeta = f"models/{id_modelo}{tipo_de_modelo}{gol_n_ticks}"
        
        # ----------------- DEFINO CARPETA DONDE SE GUARDARAN LOS GRAFICOS Y RESULTADOS --------------------------------
    
        os.makedirs(variable_carpeta, exist_ok=True)
        carpeta_resultados = f"{variable_carpeta}/"
    
        print(f"El id del modelo es {id_modelo}.\n Sus resultados se guardarán en la carpeta: {carpeta_resultados}")
        
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
        modelo = MLPClassifier(alpha=1.0, hidden_layer_sizes=(128, 128), learning_rate_init=0.001, activation='relu', solver='adam', max_iter=500, random_state=random_state)
    
        # Entrenar el modelo
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        
        # Guardar el modelo con pickle
        modelo_path = f"{carpeta_resultados}modelo.pkl"
    
        try:
            with open(modelo_path, 'wb') as file:
                pickle.dump(modelo, file)
                print(f"Modelo guardado con éxito en: {modelo_path}")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")


        
        # Redirigir los prints a un .out
        output_path = f"{carpeta_resultados}.out"
        with open(output_path, 'w') as f:
            sys.stdout = f  # Redirige la salida estándar
            
            todos_los_parametros = modelo.get_params()
            print("\nParámetros del modelo:")
            for parametro, valor in todos_los_parametros.items():
                print(f"{parametro}: {valor}")
            
            print("\nReporte de clasificación con el mejor modelo:\n", classification_report(y_test, y_pred))
            
            sys.stdout = sys.__stdout__  # Restaurar la salida estándar
        
        print(f"Resultados guardados en {output_path}")
        


        
        # Actualizar contador de modelos
        cf_c.incrementar_contador("contador_de_modelos.txt")
        print("Contador de modelos actualizado.")
        
        
        
        
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
        ruta=os.path.join(variable_carpeta, "matriz_de_confusion.png")
        plt.savefig(ruta)
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
    
        ruta=os.path.join(variable_carpeta, "distr_de_probs_10_momentos.png")
        plt.savefig(ruta)
        
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
    
        ruta=os.path.join(variable_carpeta, "10_momentos_vs_modelo.png")
        plt.savefig(ruta)
        
        plt.tight_layout()
        plt.show()
        
        # --------------------- GRAFICO DE CURVAS DE APRENDIZAJE DEL MODELO ------------------------------
    
    
        # Generar curvas de aprendizaje
        train_sizes, train_scores, test_scores = learning_curve(
            modelo, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
    
        # Calcular promedios y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
    
        # Graficar
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Precisión en Entrenamiento")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Precisión en Validación")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.title("Curvas de Aprendizaje")
        plt.xlabel("Tamaño del conjunto de entrenamiento")
        plt.ylabel("Precisión")
        plt.legend(loc="best")
        plt.grid()
    
        ruta=os.path.join(variable_carpeta, "curvas_de_aprendizaje.png")
        plt.savefig(ruta)
        
        plt.show()

    
    # --------------------------------- GRIDSEARCH CV ----------------------------------------
    
    if gridsearchCv_bool:


        # ---------------------------- CARGO EL ID DEL MODELO Y CREO SU CARPETA ----------------------------------------
        
        id_modelo = cf_c.leer_contador("contador_de_modelos.txt")
        tipo_de_modelo = "_GridSearchCV"
        variable_carpeta = f"models/{id_modelo}{tipo_de_modelo}{gol_n_ticks}"
        
        # ----------------- DEFINO CARPETA DONDE SE GUARDARAN LOS GRAFICOS Y RESULTADOS --------------------------------
        
        os.makedirs(variable_carpeta, exist_ok=True)
        carpeta_resultados = f"{variable_carpeta}/"
    
        print(f"El id del modelo es {id_modelo}.\n Sus resultados se guardarán en la carpeta: {carpeta_resultados}")


        # ------------------------------------- ENTRENAMIENTO Y RESULTADO------------------------------------------------
        
        custom_class_weight = {
            -1: 2,  # Gol del equipo Red (favorecido)
            0: 1,  # Ningún gol
            1: 2   # Gol del equipo Blue (favorecido)
        }
        
        # Definir los parámetros a probar
        param_grid = {
            'hidden_layer_sizes': [(32, 16), (32, 16, 8), (32, 32), (256, 128, 64), (256, 256), (128, 128),(128, 64, 32), (128, 64), (64, 32, 16)],
            'alpha': [0.01, 0.05, 0.1, 0.3, 1.0, 3.5, 10],
            'learning_rate_init': [0.001, 0.005, 0.01, 0.03, 0.1, 0.3, 1.0]
        }

        # Configurar la búsqueda en cuadrícula con validación cruzada
        grid_search = GridSearchCV(MLPClassifier(max_iter=1000, random_state=random_state, early_stopping=True), param_grid, n_jobs=3, cv=5)

        # Ajustar el modelo con los datos
        grid_search.fit(X, y)
        

        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Guardar el modelo con pickle
        modelo_path = f"{carpeta_resultados}modelo.pkl"
    
        try:
            with open(modelo_path, 'wb') as file:
                pickle.dump(grid_search, file)
                print(f"Modelo guardado con éxito en: {modelo_path}")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")


        
        # Actualizar contador de modelos
        cf_c.incrementar_contador("contador_de_modelos.txt")
        print("Contador de modelos actualizado.")


        
        

        results_df = pd.DataFrame(grid_search.cv_results_)

        # Seleccionar columnas relevantes para mostrar los resultados
        results_df = results_df[['params', 'mean_test_score', 'rank_test_score']]

        # Ordenar por la puntuación media en orden descendente
        results_df = results_df.sort_values(by='mean_test_score', ascending=False)

        # Redirigir los prints a un .out
        output_path = f"{carpeta_resultados}.out"

      
        print("\nTop 10 combinaciones de parámetros:")
        print(results_df.head(10))
            
        print("\nTodas las combinaciones de parámetros ordenadas por puntuación:")
        print(results_df)
            
        print("\nMejor puntuación:", grid_search.best_score_)
        mejores_parametros = grid_search.best_params_
        print("\nMejores parámetros encontrados:", mejores_parametros)
        todos_los_parametros = best_model.get_params()
        print("\nTodos los parámetros del mejor modelo:")
        for parametro, valor in todos_los_parametros.items():
            print(f"{parametro}: {valor}")
            
        print("\nReporte de clasificación con el mejor modelo:\n", classification_report(y_test, y_pred))
        
        #print(f"Resultados guardados en {output_path}")

        
        
        # ----------------- MATRIZ DE CONFUSION (Grid Search + CV) --------------------
        
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

        ruta=os.path.join(variable_carpeta, "matriz_de_confusion.png")
        plt.savefig(ruta)
        
        plt.show()
    
        
        # --- GRAFICO DE 20 MOMENTOS AL AZAR Y LAS PREDICCIONES DEL MODELO (Grid Search + CV) ---
    
        print('')
        print('GRAFICO DE 20 MOMENTOS AL AZAR Y LAS PREDICCIONES DEL MODELO:')
        print('')
       
        # Dataframe para graficar posteriormente
        resultados_gs = pd.DataFrame({
            'indice_original': indices_test,
            'valor_real': y_test,
            'valor_predicho': y_pred
        })
        
        # Obtener las probabilidades de las predicciones en el conjunto de prueba
        probabilidades_gs = best_model.predict_proba(X_test)

        # Convertir las probabilidades a un DataFrame
        probabilidades_gs_df = pd.DataFrame(probabilidades_gs, columns=['%Blue', '%Red', '%None'])
        probabilidades_gs_df = probabilidades_gs_df.round(5)

        # Agregar las columnas de valor real
        probabilidades_gs_df['Valor real'] = y_test

        # Reorganizar las columnas
        probabilidades_gs_df = probabilidades_gs_df[['Valor real', '%Blue', '%Red', '%None']]

        # Mostrar todas las filas (en Jupyter Notebook, muestra todas)
        pd.set_option('display.max_rows', None)
        probabilidades_gs_df_sample_20 = probabilidades_gs_df.sample(20, random_state=random_state).reset_index()
        
       
        probabilidades_gs_df_filtered = probabilidades_gs_df.drop(columns=['Valor real'])
        resultados_gs_concat = pd.concat([resultados_gs, probabilidades_gs_df_filtered], axis=1)

        # Formateo valores para plotear
        resultados_gs_concat['%Blue'] = resultados_gs_concat['%Blue'].apply(lambda x: f"{round(x,4)*100:.2f}")
        resultados_gs_concat['%None'] = resultados_gs_concat['%None'].apply(lambda x: f"{round(x, 4)*100:.2f}")
        resultados_gs_concat['%Red'] = resultados_gs_concat['%Red'].apply(lambda x: f"{round(x, 4)*100:.2f}")
        
        data_posiciones_filtered_original_sample = data_posiciones_filtered_original[data_posiciones_filtered_original['match_time'].isin(indices_test)]
        unique_moments = data_posiciones_filtered_original_sample[['match_time']].drop_duplicates()
        random_moments = unique_moments.sample(n=20, random_state=random_state)

        # Crear subplots
        fig, axs = plt.subplots(10, 2, figsize=(12, 30))
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

            valor_real = resultados_gs.loc[(resultados_gs['indice_original'] == match_time), 'valor_real'].values[0]
            valor_predicho_red = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%Red'].values[0]
            valor_predicho_none = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%None'].values[0]
            valor_predicho_blue = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%Blue'].values[0]
            
            ax.set_title(f'Match - Time: {match_time}\n Predicted: Red:{valor_predicho_red}%, none:{valor_predicho_none}%, Blue:{valor_predicho_blue}%\n Real: {valor_real}')
            ax.set_xlim([-605, 605])
            ax.set_ylim([-255, 255])
            ax.axis('off')  # Opcional: para ocultar los ejes

        plt.tight_layout()

        ruta=os.path.join(variable_carpeta, "20_momentos_vs_modelo.png")
        plt.savefig(ruta)
        
        plt.show()
        
        # --------------------- GRAFICO DE CURVAS DE APRENDIZAJE DEL MODELO ------------------------------


        # Generar curvas de aprendizaje
        train_sizes, train_scores, test_scores = learning_curve(
            best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        # Calcular promedios y desviaciones estándar
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Graficar
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Precisión en Entrenamiento")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Precisión en Validación")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.title("Curvas de Aprendizaje")
        plt.xlabel("Tamaño del conjunto de entrenamiento")
        plt.ylabel("Precisión")
        plt.legend(loc="best")
        plt.grid()
        
        ruta=os.path.join(variable_carpeta, "curvas_de_aprendizaje.png")
        plt.savefig(ruta)
        
        plt.show()
    
        # --- GRAFICO DE TODOS LOS MOMENTOS DE UN PARTIDO AL AZAR Y LAS PREDICCIONES DEL MODELO (Grid Search + CV) ---
        
        #(HACER)
        
    print('Listo.')
    
   
    
    

def modelar_tensorflow_003(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    gol_n_ticks
    
    ):
    
    
    import pandas as pd
    import numpy as np
    from sklearn import model_selection, ensemble, linear_model, neural_network, metrics, inspection

    import matplotlib.pyplot as plt
    from pickle import load, dump
    import os

    if True:
            # ---------------------------- CARGO EL ID DEL MODELO Y CREO SU CARPETA ----------------------------------------
        
            id_modelo = cf_c.leer_contador("contador_de_modelos.txt")
            tipo_de_modelo = "_TensorFlow003_"
            variable_carpeta = f"models/{id_modelo}{tipo_de_modelo}{gol_n_ticks}"
            
            # ----------------- DEFINO CARPETA DONDE SE GUARDARAN LOS GRAFICOS Y RESULTADOS --------------------------------
        
            os.makedirs(variable_carpeta, exist_ok=True)
            carpeta_resultados = f"{variable_carpeta}/"
        
            print(f"El id del modelo es {id_modelo}.\n Sus resultados se guardarán en la carpeta: {carpeta_resultados}")
            
            
            # Actualizar contador de modelos
            cf_c.incrementar_contador("contador_de_modelos.txt")
            print("Contador de modelos actualizado.")
            
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
            

            # --- Librerias del modelo ---
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, regularizers
            import pandas as pd
            import numpy as np
            import os
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            import keras_tuner as kt


            # Verificar que TensorFlow está utilizando solamente la CPU
            if not tf.config.list_physical_devices('GPU'):
                print("TensorFlow está configurado para usar únicamente la CPU.")
            else:
                print("Advertencia: TensorFlow está utilizando la GPU.")
            
            tf.config.threading.set_intra_op_parallelism_threads(3)

            
            # Dividir los datos para el entrenamiento y testeo del modelo
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices_X_y, test_size=0.3, random_state=random_state, stratify=y) # stratify=y hace que se mantenga las proporciones
            
            print('')
            print(Counter(y_train))
            print('')
            print(Counter(y_test))
            print('')
            print("Índices correspondientes a y_test:\n", indices_test[:5])


            # Codificar etiquetas
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)


            # Normalizar datos
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


            # Guardar encoder y scaler
            ruta_encoder = os.path.join(carpeta_resultados, "encoder.pkl")
            ruta_scaler = os.path.join(carpeta_resultados, "scaler.pkl")
            
            with open(ruta_encoder, "wb") as f:
                pickle.dump(encoder, f)

            with open(ruta_scaler, "wb") as f:
                pickle.dump(scaler, f)

            
            # --- Defino el modelo ---
            from tensorflow.keras.optimizers.schedules import ExponentialDecay

            def build_model(hp):
                model = keras.Sequential()
                model.add(layers.Input(shape=(X_train.shape[1],)))
                
                # Número de capas ocultas
                for i in range(hp.Int('num_layers', 2, 5)):
                    model.add(layers.Dense(
                        hp.Int(f'units_{i}', 64, 128, step=32),
                        activation=hp.Choice('activation', ['relu', 'selu', 'tanh']),
                        kernel_regularizer=regularizers.l2(hp.Float('l2_reg', 1e-5, 1e-2, sampling='LOG'))
                    ))
                    model.add(layers.BatchNormalization())
                    model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)))
                
                # Capa de salida
                model.add(layers.Dense(len(np.unique(y)), activation='softmax'))

                
                # Selección de optimizador
                optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
                learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])

                # Programador de tasa de aprendizaje
                lr_schedule = ExponentialDecay(
                    initial_learning_rate=learning_rate,
                    decay_steps=1000,
                    decay_rate=0.9,
                    staircase=True
                )

                if optimizer_name == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
                elif optimizer_name == 'rmsprop':
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
                else:
                    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=hp.Float('momentum_sgd', 0.0, 0.9), nesterov=True)

                # Función de pérdida adaptativa
                loss_fn = 'sparse_categorical_crossentropy' if len(np.unique(y)) > 2 else 'binary_crossentropy'

                model.compile(
                    optimizer=optimizer,
                    loss=loss_fn,
                    metrics=['accuracy']
                )

                return model
            
            
            # Configuración de búsqueda de hiperparámetros
            tuner = kt.Hyperband(
                build_model,
                objective='val_accuracy',
                max_epochs=100,
                factor=3,
                directory=carpeta_resultados,
                project_name='hyperparam_tuning'
            )
            
            

            # Ejecución de la búsqueda de hiperparámetros
            tuner.search(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100
                )
            
            
            # Seleccionar el mejor modelo
            best_hps = tuner.get_best_hyperparameters()[0]
            best_model = tuner.hypermodel.build(best_hps)
            history = best_model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                class_weight = {0: 1.0, 1: 1.0, 2: 1.0},
                epochs=100,  # Más entrenamiento para el modelo final
                workers=3, 
                use_multiprocessing=True
            )
            
            
            # Guardar el mejor modelo
            modelo_path = f"{carpeta_resultados}best_model.h5"
            best_model.save(modelo_path)
            print("Entrenamiento finalizado y mejor modelo guardado.")





            # ---------------------------- Resultados ---------------------------------


            # Obtener predicciones
            y_pred_prob = best_model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = y_test
             
            # Matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            ruta=os.path.join(variable_carpeta, "matriz_de_confusion.png")
            plt.savefig(ruta)
               
            # Reporte de clasificación

            from sklearn.metrics import classification_report

            print("Classification Report:")
            print(classification_report(y_true, y_pred))
                
                
            # ---------------------- Curvas de aprendizaje -----------------------

            plt.figure(figsize=(12,5))

            # Pérdida
            plt.subplot(1,2,1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history.get('val_loss', []), label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curve')

            # Precisión
            plt.subplot(1,2,2)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Accuracy Curve')

            ruta=os.path.join(variable_carpeta, "curvas_de_aprendizaje.png")
            plt.savefig(ruta)
            
            
            
            # --- GRAFICO DE 20 MOMENTOS AL AZAR Y LAS PREDICCIONES DEL MODELO (Grid Search + CV) ---
    
            print('')
            print('GRAFICO DE 20 MOMENTOS AL AZAR Y LAS PREDICCIONES DEL MODELO:')
            print('')
           
            # Dataframe para graficar posteriormente
            resultados_gs = pd.DataFrame({
                'indice_original': indices_test,
                'valor_real': y_test,
                'valor_predicho': y_pred
            })
            
            # Obtener las probabilidades de las predicciones en el conjunto de prueba
            probabilidades_gs = best_model.predict_proba(X_test)

            # Convertir las probabilidades a un DataFrame
            probabilidades_gs_df = pd.DataFrame(probabilidades_gs, columns=['%Blue', '%Red', '%None'])
            probabilidades_gs_df = probabilidades_gs_df.round(5)

            # Agregar las columnas de valor real
            probabilidades_gs_df['Valor real'] = y_test

            # Reorganizar las columnas
            probabilidades_gs_df = probabilidades_gs_df[['Valor real', '%Blue', '%Red', '%None']]

            # Mostrar todas las filas (en Jupyter Notebook, muestra todas)
            pd.set_option('display.max_rows', None)
            probabilidades_gs_df_sample_20 = probabilidades_gs_df.sample(20, random_state=random_state).reset_index()
            
           
            probabilidades_gs_df_filtered = probabilidades_gs_df.drop(columns=['Valor real'])
            resultados_gs_concat = pd.concat([resultados_gs, probabilidades_gs_df_filtered], axis=1)

            # Formateo valores para plotear
            resultados_gs_concat['%Blue'] = resultados_gs_concat['%Blue'].apply(lambda x: f"{round(x,4)*100:.2f}")
            resultados_gs_concat['%None'] = resultados_gs_concat['%None'].apply(lambda x: f"{round(x, 4)*100:.2f}")
            resultados_gs_concat['%Red'] = resultados_gs_concat['%Red'].apply(lambda x: f"{round(x, 4)*100:.2f}")
            
            data_posiciones_filtered_original_sample = data_posiciones_filtered_original[data_posiciones_filtered_original['match_time'].isin(indices_test)]
            unique_moments = data_posiciones_filtered_original_sample[['match_time']].drop_duplicates()
            random_moments = unique_moments.sample(n=20, random_state=random_state)

            # Crear subplots
            fig, axs = plt.subplots(10, 2, figsize=(12, 30))
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

                valor_real = resultados_gs.loc[(resultados_gs['indice_original'] == match_time), 'valor_real'].values[0]
                valor_predicho_red = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%Red'].values[0]
                valor_predicho_none = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%None'].values[0]
                valor_predicho_blue = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%Blue'].values[0]
                
                ax.set_title(f'Match - Time: {match_time}\n Predicted: Red:{valor_predicho_red}%, none:{valor_predicho_none}%, Blue:{valor_predicho_blue}%\n Real: {valor_real}')
                ax.set_xlim([-605, 605])
                ax.set_ylim([-255, 255])
                ax.axis('off')  # Opcional: para ocultar los ejes

            plt.tight_layout()

            ruta=os.path.join(variable_carpeta, "20_momentos_vs_modelo.png")
            plt.savefig(ruta)
            
            plt.show()
            
            
            
            
    print('Listo!')





def modelar_randomsearch_rf_004(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    gol_n_ticks,
    #custom_class_weight, # Pesos de las clases customizados para el random forest.
    custom_param_grid_rf, # Parametros customizados para el grid search.
    combinaciones
    ):
    
    if True:
        # ---------------------------- CARGO EL ID DEL MODELO Y CREO SU CARPETA ----------------------------------------
    
        id_modelo = cf_c.leer_contador("contador_de_modelos.txt")
        tipo_de_modelo = "_RandomSearch_RF_"
        variable_carpeta = f"models/{id_modelo}{tipo_de_modelo}{gol_n_ticks}"
        
        # ----------------- DEFINO CARPETA DONDE SE GUARDARAN LOS GRAFICOS Y RESULTADOS --------------------------------
    
        os.makedirs(variable_carpeta, exist_ok=True)
        carpeta_resultados = f"{variable_carpeta}/"
    
        print(f"El id del modelo es {id_modelo}.\n Sus resultados se guardarán en la carpeta: {carpeta_resultados}")
        
        # Actualizar contador de modelos
        cf_c.incrementar_contador("contador_de_modelos.txt")
        print("Contador de modelos actualizado.")
       
        
        
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
        num_muestras_gol = round(min(len(indices_gol_rojo), len(indices_gol_azul), len(indices_nada)/n) / 2)
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
        
        # Dividir los datos en entrenamiento (70%), validación (20%) y prueba (10%)
        X_train, X_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(X, y, indices_X_y, test_size=0.3, random_state=random_state, stratify=y)
        X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_temp, y_temp, indices_temp, test_size=1/3, random_state=42, stratify=y_temp)

        
        print('')
        print(Counter(y_train))
        print('')
        print(Counter(y_test))
        print('')
        print("Índices correspondientes a y_test:\n", indices_test[:5])
        
        
        
        # ------------------------------------- ENTRENAMIENTO Y RESULTADO------------------------------------------------
        

        
        randomized_search_rf = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=random_state),  # Modelo base
            param_distributions=custom_param_grid_rf,  # Espacio de búsqueda
            n_iter=combinaciones,  # Número de combinaciones aleatorias a probar
            cv=10,
            n_jobs=6,  # Usar 3 CPUs
            verbose=3,  # Mostrar progreso
            random_state=random_state
        )

        # Ajustar el modelo con los datos
        randomized_search_rf.fit(X_train, y_train)
     
        print(np.unique(y_train))
        
        # Obtener el mejor modelo
        best_model = randomized_search_rf.best_estimator_
        y_pred = best_model.predict(X_test)
        
        
        
        
        # ------------------------- GUARDAR EL MODELO ------------------------------

        # Guardar todos los modelos
        modelos_path = f"{carpeta_resultados}modelos.pkl"
        
        try:
            with open(modelos_path, 'wb') as file:
                pickle.dump(randomized_search_rf, file)
                print(f"Modelo guardado con éxito en: {modelos_path}")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
        
        
        # Guardar el modelo con pickle
        best_model_path = f"{carpeta_resultados}best_model.pkl"
        
        try:
            with open(best_model_path, 'wb') as file:
                pickle.dump(best_model, file)
                print(f"Modelo guardado con éxito en: {best_model_path}")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
        
        
        # Guardo variables de train, test
        # Guardo data para el análisis
        carpeta_X_y_split_data = f"{carpeta_resultados}X_y_split_data/"
        os.makedirs(carpeta_X_y_split_data, exist_ok=True)

        np.save(os.path.join(carpeta_X_y_split_data, "X_val.npy"), X_val)
        np.save(os.path.join(carpeta_X_y_split_data, "y_val.npy"), y_val)
        np.save(os.path.join(carpeta_X_y_split_data, "X_test.npy"), X_test)
        np.save(os.path.join(carpeta_X_y_split_data, "y_test.npy"), y_test)
        np.save(os.path.join(carpeta_X_y_split_data, "X_train.npy"), X_train)
        np.save(os.path.join(carpeta_X_y_split_data, "y_train.npy"), y_train)
        np.save(os.path.join(carpeta_X_y_split_data, "indices_val.npy"), indices_val)
        
        tipo_de_modelo = 'randomforest'
        
        if tipo_de_modelo == 'SVC':
            with open(os.path.join(carpeta_resultados, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
        
        print('Listo')






def custom_evaluation_10_03_01(y_true, y_pred, class_probs):
    """
    Función de evaluación personalizada.
    
    Parámetros:
    - y_true: Etiquetas reales (array de forma [n_samples]).
    - y_pred: Etiquetas predichas (array de forma [n_samples]).
    - class_probs: Probabilidades de clase (array de forma [n_samples, n_classes]).
    
    Retorna:
    - Costo promedio por muestra.
    """
    # Definir la matriz de costos
    cost_matrix = {
        ('Red', 'Blue'): 10.0,  # Penalización alta por confundir Gol rojo con Gol azul
        ('Blue', 'Red'): 10.0,  # Penalización alta por confundir Gol azul con Gol rojo
        ('none', 'Red'): 1.0,   # Penalización baja por confundir Nada con Gol rojo
        ('none', 'Blue'): 1.0,  # Penalización baja por confundir Nada con Gol azul
        ('Red', 'none'): 3.0,   # Penalización media por confundir Gol rojo con Nada
        ('Blue', 'none'): 3.0   # Penalización media por confundir Gol azul con Nada
    }
    
    # Inicializar el costo total
    total_cost = 0.0
    
    # Calcular el costo para cada predicción
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            # Obtener la penalización de la matriz de costos
            penalty = cost_matrix.get((true_label, pred_label), 0.0)
            # Aplicar la penalización al costo total
            total_cost += penalty
    
    # Calcular el costo promedio por muestra
    average_cost = total_cost / len(y_true)
    return average_cost 





def modelar_svm_007(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    gol_n_ticks,
    custom_param_grid_svm, # Parametros customizados para el grid search.
    cant_combinaciones
    ):

    
    
    id_modelo = cf_c.leer_contador("contador_de_modelos.txt")
    tipo_de_modelo = "_SVM_GridSearch_"
    variable_carpeta = f"models/{id_modelo}{tipo_de_modelo}{gol_n_ticks}"
    
    # ----------------- DEFINO CARPETA DONDE SE GUARDARAN LOS GRAFICOS Y RESULTADOS --------------------------------

    os.makedirs(variable_carpeta, exist_ok=True)
    carpeta_resultados = f"{variable_carpeta}/"

    print(f"El id del modelo es {id_modelo}.\n Sus resultados se guardarán en la carpeta: {carpeta_resultados}")
    
   # Actualizar contador de modelos
    cf_c.incrementar_contador("contador_de_modelos.txt")
    print("Contador de modelos actualizado.")
    
    
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
    num_muestras_gol = round(min(len(indices_gol_rojo), len(indices_gol_azul), len(indices_nada)/n) / 2)
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



    # ------------------------- PREPARACIÓN DE DATOS ------------------------------
    
    
    # Dividir los datos en entrenamiento (70%), validación (20%) y prueba (10%)
    X_train, X_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(X, y, indices_X_y, test_size=0.3, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_temp, y_temp, indices_temp, test_size=1/3, random_state=42, stratify=y_temp)

    # Normalizar los datos (los SVMs son sensibles a la escala)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Guardo data para el análisis
    carpeta_X_y_split_data = f"{carpeta_resultados}X_y_split_data/"
    os.makedirs(carpeta_X_y_split_data, exist_ok=True)

    np.save(os.path.join(carpeta_X_y_split_data, "X_val.npy"), X_val)
    np.save(os.path.join(carpeta_X_y_split_data, "y_val.npy"), y_val)
    np.save(os.path.join(carpeta_X_y_split_data, "X_test.npy"), X_test)
    np.save(os.path.join(carpeta_X_y_split_data, "y_test.npy"), y_test)
    np.save(os.path.join(carpeta_X_y_split_data, "X_train.npy"), X_train)
    np.save(os.path.join(carpeta_X_y_split_data, "y_train.npy"), y_train)
    np.save(os.path.join(carpeta_X_y_split_data, "indices_val.npy"), indices_val)

    
    with open(os.path.join(carpeta_resultados, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
        

    # ------------------ ENTRENAMIENTO CON GRIDSEARCH + CV -----------------------

    
    
    pipeline_svm = Pipeline([
        ('scaler', StandardScaler()),  # Escalado para mejorar el rendimiento del SVM
        ('svm', SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=random_state))
    ])
    
    #pipeline_svm.fit(X_train, y_train)
    
    # Configurar la búsqueda aleatoria
    random_search = RandomizedSearchCV(
        pipeline_svm, custom_param_grid_svm, n_iter=cant_combinaciones, cv=5, scoring='accuracy', random_state=random_state, n_jobs=6, verbose=3
    )

    # Entrenar el modelo con búsqueda de hiperparámetros
    random_search.fit(X_train, y_train)

    # Obtener el mejor modelo
    best_model = random_search.best_estimator_
        
    
    # ------------------------- GUARDAR EL MODELO ------------------------------
    
    # Guardar el mejor modelo
    modelo_path = f"{carpeta_resultados}best_model.pkl"
    
    try:
        with open(modelo_path, 'wb') as file:
            pickle.dump(best_model, file)
            print(f"Modelo guardado con éxito en: {modelo_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
        
        
        
    # Guardar todos los modelos
    modelo_path = f"{carpeta_resultados}modelos.pkl"
    
    try:
        with open(modelo_path, 'wb') as file:
            pickle.dump(random_search, file)
            print(f"todos_los_modelos guardado con éxito en: {modelo_path}")
    except Exception as e:
        print(f"Error al guardar todos_los_modelos: {e}")




# FUNCIONA PARA: randomforest (004)
# EN PROCESO: svc (006)
# NO FUNCIONA PARA: redneuronal
def analizar_modelo_pkl(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    directorio_modelo,  # directorio donde está el modelo
    gol_n_ticks,
    custom_list_match_ids,  # lista con ids de partidos a analizar en la sección de análisis puntual de todos los momentos de un partido
    custom_eval_function, # Llama a la funcion evaluacion customizada
    tipo_de_busqueda, # 'multiple' para analizar Gridsearch/Randomsearch; 'simple' para modelos unicos.
    tipo_de_modelo # 'SVC, randomforest, redneuronal'
    ):
    
    
    

    """ ------------------------------- Importo el modelo ---------------------------------- """
    
    # Cargar el modelo
    model_path = os.path.join(directorio_modelo, "best_model.pkl")
    with open(model_path, "rb") as f:
        best_model = pickle.load(f)
        
    if tipo_de_busqueda == 'multiple':
        # Cargar el modelo
        models_path = os.path.join(directorio_modelo, "modelos.pkl")
        with open(models_path, "rb") as f:
            modelos = pickle.load(f)
    
    
    
    
    
    """ --------------------- Importo los datos del split en el modelado ------------------------- """
    
    carpeta_X_y_split_data = f"{directorio_modelo}X_y_split_data/"

    # Cargo los datos del split
    X_val = np.load(os.path.join(carpeta_X_y_split_data, "X_val.npy"))
    y_val = np.load(os.path.join(carpeta_X_y_split_data, "y_val.npy"))
    X_test = np.load(os.path.join(carpeta_X_y_split_data, "X_test.npy"))
    y_test = np.load(os.path.join(carpeta_X_y_split_data, "y_test.npy"))
    X_train = np.load(os.path.join(carpeta_X_y_split_data, "X_train.npy"))
    y_train = np.load(os.path.join(carpeta_X_y_split_data, "y_train.npy"))
    indices_val = np.load(os.path.join(carpeta_X_y_split_data, "indices_val.npy"))

    
    """ -------------------------- Metricas tipicas (en un .out) ----------------------------- """
    
    #best_model = modelos.best_estimator_
    #best_model = modelos
    
    #print('DATA DE MODELO:\n')
    #print(best_model.named_steps['svm'])

    
    # Predecir en el conjunto de validación
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)
    
    y_pred = best_model.predict(X_val)
    
    if False: # Cosas viejas de debugging
        #print('X_val:\n', X_val)
        #print(type(X_val))
        #print(X_val.shape)  # Si es un array de NumPy, muestra las dimensiones
        #print(X_val.dtype)  # Si es un array de NumPy, muestra el tipo de datos
        
        #print('y_val:\n', y_val)
        #print(type(y_val))
        #print(y_val.shape)  # Si es un array de NumPy, muestra las dimensiones
        #print(y_val.dtype)  # Si es un array de NumPy, muestra el tipo de datos


        #print('y_val_pred:\n', y_val_pred)
        #print(type(y_val_pred))
        #print(y_val_pred.shape)  # Si es un array de NumPy, muestra las dimensiones
        #print(y_val_pred.dtype)  # Si es un array de NumPy, muestra el tipo de datos
        
        
        #print('y_val_proba:\n', y_val_proba)
        #print(type(y_val_proba))
        #print(y_val_proba.shape)  # Si es un array de NumPy, muestra las dimensiones
        #print(y_val_proba.dtype)  # Si es un array de NumPy, muestra el tipo de datos
        
        
        
        # Configurar pandas para mostrar todos los parámetros completos
        #pd.set_option('display.max_colwidth', None)
        
        #results_df = pd.DataFrame(modelos.cv_results_)
        #results_df = results_df[['params', 'mean_test_score', 'rank_test_score']]
        #results_df = results_df.sort_values(by='mean_test_score', ascending=False)
        print('')


    # Redirigir los prints a un .out
    output_path = f"{directorio_modelo}.out"
     
    # Crear el archivo .out y escribir los resultados
    try:
        with open(output_path, "w") as f:
            
            sys.stdout = f
             
            # Métricas típicas
            f.write("\nMétricas en Validación del mejor modelo:\n")
            f.write(f"\nAccuracy: {accuracy_score(y_val, y_val_pred)}")
            f.write(f"\nPrecision: {precision_score(y_val, y_val_pred, average='macro')}")
            f.write(f"\nRecall: {recall_score(y_val, y_val_pred, average='macro')}")
            f.write(f"\nF1-Score: {f1_score(y_val, y_val_pred, average='macro')}")
            
            f.write("\n\nClassification Report:\n")
            f.write(classification_report(y_val, y_val_pred))
            
            # Restaurar la salida estándar
            sys.stdout = sys.__stdout__  
            
        # Confirmación de guardado
        print(f"Resultados guardados en {output_path}")
        
        
    except Exception as e:
        print(f"Error al intentar guardar los resultados: {e}")
        
    
    
    
    if tipo_de_busqueda == 'multiple':

        # Configurar pandas para mostrar todos los parámetros completos
        pd.set_option('display.max_colwidth', None)
        
        results_df = pd.DataFrame(modelos.cv_results_)

        # Seleccionar columnas relevantes para mostrar los resultados
        results_df = results_df[['params', 'mean_test_score', 'rank_test_score']]

        # Ordenar por la puntuación media en orden descendente
        results_df = results_df.sort_values(by='mean_test_score', ascending=False)

        # Redirigir los prints a un .tst
        output_models_path = f"{directorio_modelo}models.out"
        

        # Crear el archivo y escribir los resultados
        try:
            with open(output_models_path, "w") as f:
                
                sys.stdout = f

                # Top 10 combinaciones de parámetros
                f.write("\nTop 10 combinaciones de parámetros:\n")
                f.write(results_df.head(10).to_string(index=False))  # Mostrar parámetros completos

                # Todas las combinaciones de parámetros ordenadas
                f.write("\n\nTodas las combinaciones de parámetros ordenadas por puntuación:\n")
                f.write(results_df.to_string(index=False))

                # Mejor puntuación
                f.write(f"\n\nMejor puntuación: {modelos.best_score_}\n")

                # Mejores parámetros
                mejores_parametros = modelos.best_params_
                f.write("\nMejores parámetros encontrados:\n")
                for parametro, valor in mejores_parametros.items():
                    f.write(f"{parametro}: {valor}\n")

                # Todos los parámetros del mejor modelo
                f.write("\nTodos los parámetros del mejor modelo:\n")
                todos_los_parametros = best_model.get_params()
                for parametro, valor in todos_los_parametros.items():
                    f.write(f"{parametro}: {valor}\n")
                
                sys.stdout = sys.__stdout__  # Restaurar la salida estándar

            # Confirmación de guardado
            print(f"Resultados guardados en {output_models_path}")
            
            
        except Exception as e:
            print(f"Error al intentar guardar los resultados: {e}")

        
    print("---- Metricas en .out terminadas --------------------------------------")

    
    """ --- Curvas de aprendizaje (beta) ---------------------------------------------------------- """
    from sklearn.model_selection import learning_curve, StratifiedKFold

    # Definir validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Definir tamaños de entrenamiento en escala logarítmica
    train_sizes = np.linspace(0.05, 1.0, 8)  # Más precisión en datos pequeños

    # Obtener curvas de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_train, y_train, cv=cv, scoring="accuracy", 
        n_jobs=4, train_sizes=train_sizes, shuffle=True
    )

    # Calcular promedios y desviaciones estándar
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Gráfica mejorada
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Precisión en Entrenamiento")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")

    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Precisión en Validación")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")

    plt.axhline(y=np.max(test_mean), color='r', linestyle='--', label="Máx. Precisión de Validación")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Precisión")
    plt.title("Curvas de Aprendizaje Mejoradas")
    plt.legend(loc="best")
    plt.grid(True)

    # Guardar imagen
    ruta = os.path.join(directorio_modelo, "curvas_de_aprendizaje_mejoradas.png")
    plt.savefig(ruta)
    plt.show()

    print("---- Curvas de aprendizaje terminadas y guardadas en:", ruta)
    
    

    """ ------------------------------ Matriz de confusion ------------------------------- """
    
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    
    cf_v.matriz_de_confusion(
        directorio_modelo,
        conf_matrix
    )
    
    print("---- Matriz de confusion terminada --------------------------------------")

    
    
    """ --------------- 20 momentos al azar vs las predicciones del modelo ----------------- """
    
    print('')
    print('GRAFICO DE 20 MOMENTOS AL AZAR Y LAS PREDICCIONES DEL MODELO:')
    print('')
   

   
    # Dataframe para graficar posteriormente
    resultados_gs = pd.DataFrame({
        'indice_original': indices_val,
        'valor_real': y_val,
        'valor_predicho': y_pred
    })
    
    
    # Obtener las probabilidades de las predicciones en el conjunto de prueba
    probabilidades_gs = best_model.predict_proba(X_val)

    # Convertir las probabilidades a un DataFrame
    probabilidades_gs_df = pd.DataFrame(probabilidades_gs, columns=['%Blue', '%Red', '%None'])
    probabilidades_gs_df = probabilidades_gs_df.round(5)

    # Agregar las columnas de valor real
    probabilidades_gs_df['Valor real'] = y_val

    # Reorganizar las columnas
    probabilidades_gs_df = probabilidades_gs_df[['Valor real', '%Blue', '%Red', '%None']]

    # Mostrar todas las filas (en Jupyter Notebook, muestra todas)
    pd.set_option('display.max_rows', None)
    probabilidades_gs_df_sample_20 = probabilidades_gs_df.sample(20, random_state=random_state).reset_index()
    
   
    probabilidades_gs_df_filtered = probabilidades_gs_df.drop(columns=['Valor real'])
    resultados_gs_concat = pd.concat([resultados_gs, probabilidades_gs_df_filtered], axis=1)

    # Formateo valores para plotear
    resultados_gs_concat['%Blue'] = resultados_gs_concat['%Blue'].apply(lambda x: f"{round(x,4)*100:.2f}")
    resultados_gs_concat['%None'] = resultados_gs_concat['%None'].apply(lambda x: f"{round(x, 4)*100:.2f}")
    resultados_gs_concat['%Red'] = resultados_gs_concat['%Red'].apply(lambda x: f"{round(x, 4)*100:.2f}")
    
    data_posiciones_filtered_original_sample = data_posiciones_filtered_original[data_posiciones_filtered_original['match_time'].isin(indices_val)]
    unique_moments = data_posiciones_filtered_original_sample[['match_time']].drop_duplicates()
    random_moments = unique_moments.sample(n=20, random_state=random_state)

    # Crear subplots
    fig, axs = plt.subplots(10, 2, figsize=(12, 30))
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

        valor_real = resultados_gs.loc[(resultados_gs['indice_original'] == match_time), 'valor_real'].values[0]
        valor_predicho_red = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%Red'].values[0]
        valor_predicho_none = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%None'].values[0]
        valor_predicho_blue = resultados_gs_concat.loc[(resultados_gs['indice_original'] == match_time), '%Blue'].values[0]
        
        ax.set_title(f'Match - Time: {match_time}\n Predicted: Red:{valor_predicho_red}%, none:{valor_predicho_none}%, Blue:{valor_predicho_blue}%\n Real: {valor_real}')
        ax.set_xlim([-605, 605])
        ax.set_ylim([-255, 255])
        ax.axis('off')  # Opcional: para ocultar los ejes

    plt.tight_layout()

    ruta=os.path.join(directorio_modelo, "20_momentos_vs_predicciones.png")
    plt.savefig(ruta)
    

    print("---- 20 momentos al azar terminados --------------------------------------")



    """ -------------------------------- Partidos enteros ------------------------------- """
    
    for custom_match_id in custom_list_match_ids:
        cf_v.partido_entero_pkl(
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
        )
        print(f"El partido con id {custom_match_id} ha sido graficado.")
        
    print("---- Partidos enteros terminados --------------------------------------")


    
    """ ------------- Probabilidades a lo largo del tiempo (por partido entero) --------------- """
    
    for custom_match_id in custom_list_match_ids: 
        cf_v.lineplot_partido_entero_pkl(
            random_state,
            background_image,
            data_posiciones_grouped_filtered_labeled,
            data_posiciones_filtered,
            data_posiciones_filtered_original,
            directorio_modelo,  # directorio donde está el modelo
            gol_n_ticks,
            custom_match_id,  # id del partido a analizar en la sección de análisis puntual de todos los momentos de un partido
            best_model
        )
        print(f"El partido con id {custom_match_id} ha sido graficado.")
        
    print("---- Lineplot de partidos enteros terminados --------------------------------------")

    
    
    
    
    """ ------------------------------- Curvas de aprendizaje ---------------------------------- """
    
    # -------- ¡¡¡FALTA CHEQUEAR!!! ----------------------------------------------

    # -------- ¡¡¡FALTA CHEQUEAR!!! ----------------------------------------------

    # -------- ¡¡¡FALTA CHEQUEAR!!! ----------------------------------------------

    # -------- ¡¡¡FALTA CHEQUEAR!!! ----------------------------------------------

    # -------- ¡¡¡FALTA CHEQUEAR!!! ----------------------------------------------

    # -------- ¡¡¡FALTA CHEQUEAR!!! ----------------------------------------------
    
    
    
    
    # Generar curvas de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=6,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Calcular promedios y desviaciones estándar
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Precisión en Entrenamiento")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Precisión en Validación")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.title("Curvas de Aprendizaje")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Precisión")
    plt.legend(loc="best")
    plt.grid()
    
    ruta=os.path.join(directorio_modelo, "curvas_de_aprendizaje.png")
    plt.savefig(ruta)
    
    
    
    
     
    print("---- Curvas de aprendizaje terminadas --------------------------------------")

    
    """ Listo """
    print('')
    print('¡¡¡Analisis finalizado!!!!')









# EN CONSTRUCCION
def modelar_xgboost(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    gol_n_ticks,
    #custom_class_weight, # Pesos de las clases customizados para el random forest.
    custom_param_grid_rf, # Parametros customizados para el grid search.
    combinaciones
    ):
    
    from xgboost import XGBClassifier

    X = df.drop(columns=['goal_next_3s'])
    y = df['goal_next_3s']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300, 400],  # Número de árboles
        'max_depth': [3, 5, 7],  # Profundidad de los árboles
        'learning_rate': [0.01, 0.05, 0.1],  # Tasa de aprendizaje
        'subsample': [0.7, 0.8, 0.9],  # Proporción de datos usada por árbol
        'colsample_bytree': [0.7, 0.8, 0.9],  # Proporción de features usadas por árbol
        'gamma': [0, 0.5, 1, 5],  # Poda de ramas
        'reg_lambda': [0.1, 1, 10],  # Regularización L2
        'reg_alpha': [0, 0.1, 1]  # Regularización L1
    }

    # Creamos el clasificador
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Randomized Search con validación cruzada
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,  # Número de combinaciones aleatorias a probar
        scoring='accuracy',  # Métrica de evaluación
        cv=5,  # 5 folds en validación cruzada
        verbose=3,
        n_jobs=6,  # Usa todos los núcleos disponibles
        random_state=42
    )

    # Ejecutamos la búsqueda
    random_search.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')





# EN CONSTRUCCION
# Funcion para usar metricas creadas para el xgBoost (y posible RedNeuronal).
def usar_metricas():
    return



def preparar_datos(
    random_state,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    gol_n_ticks,
    estrategia, # La estrategia es para ver como preparar o muestrear los datos.
                # valores posibles: 'original', no hace cambios.
                #                   'shuffle', mezcla el orden de las 7 entidades en cancha.
                #                   'metrics', usa las metricas creadas como datos de entrenamiento (para xgBoost y posible RedNeuronal).           
    proporcion_none # Proporcion de None vs otros equipos (2 significa 50% None, 25% Red, 25% Blue).
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
    num_muestras_gol = round(min(len(indices_gol_rojo), len(indices_gol_azul), len(indices_nada)/proporcion_none))
    num_muestras_nada = round(num_muestras_gol * proporcion_none)

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
    
    
    estrategias_validas = {"original", "shuffle", "metrics"}
    
    if estrategia in estrategias_validas:
        print(f'Usando la estrategia de muestreo: {estrategia}')
    else:
        print('Has incertado una estrategia de muestreo invalida.\nPor defecto se usara la estrategia: original.')
        
        


    for row in data_balanceada.itertuples():
        match_time = row.match_time
        gol_N_ticks = getattr(row, gol_n_ticks)
        
        # Seleccionar las 7 filas correspondientes
        subset = data_posiciones_filtered[data_posiciones_filtered['match_time'] == match_time]
        
        # Seleccion de estrategia
        if estrategia == 'shuffle':
            subset = subset.sample(frac=1, random_state=random_state).reset_index(drop=True)  # Mezclar filas
        elif estrategia == 'metrics':
            usar_metricas()
        if estrategia != 'metrics':
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
    
    # Dividir los datos en entrenamiento (70%), validación (20%) y prueba (10%)
    X_train, X_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(X, y, indices_X_y, test_size=0.3, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_temp, y_temp, indices_temp, test_size=1/3, random_state=42, stratify=y_temp)

    print('')
    print(Counter(y_train))
    print('')
    print(Counter(y_test))
    print('')
    print(Counter(y_val))
    print('')
    
    return X_train, X_test, X_val, y_train, y_test, y_val, indices_train, indices_test, indices_val

    
    
def guardar_split_data(carpeta_resultados, X_train, X_test, X_val, y_train, y_test, y_val, indices_val):

    # Guardo data para el análisis
    carpeta_X_y_split_data = f"{carpeta_resultados}X_y_split_data/"
    os.makedirs(carpeta_X_y_split_data, exist_ok=True)

    np.save(os.path.join(carpeta_X_y_split_data, "X_val.npy"), X_val)
    np.save(os.path.join(carpeta_X_y_split_data, "y_val.npy"), y_val)
    np.save(os.path.join(carpeta_X_y_split_data, "X_test.npy"), X_test)
    np.save(os.path.join(carpeta_X_y_split_data, "y_test.npy"), y_test)
    np.save(os.path.join(carpeta_X_y_split_data, "X_train.npy"), X_train)
    np.save(os.path.join(carpeta_X_y_split_data, "y_train.npy"), y_train)
    np.save(os.path.join(carpeta_X_y_split_data, "indices_val.npy"), indices_val)
    
    return



def guardar_modelos_randomsearch(
    carpeta_resultados,
    random_search,
    best_model
    ):
            
    # Guardar todos los modelos
    modelos_path = f"{carpeta_resultados}modelos.pkl"
    
    try:
        with open(modelos_path, 'wb') as file:
            pickle.dump(random_search, file)
            print(f"Modelo guardado con éxito en: {modelos_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
    
    
    # Guardar el modelo con pickle
    best_model_path = f"{carpeta_resultados}best_model.pkl"
    
    try:
        with open(best_model_path, 'wb') as file:
            pickle.dump(best_model, file)
            print(f"Modelo guardado con éxito en: {best_model_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
    
    return



# EN CONSTRUCCION
def modelar_redneuronal(
    random_state,
    background_image,
    data_posiciones_grouped_filtered_labeled,
    data_posiciones_filtered,
    data_posiciones_filtered_original,
    gol_n_ticks,
    custom_param_grid,
    estrategia,
    proporcion_none,
    combinaciones
    ):
    # ---------------------------- CARGO EL ID DEL MODELO Y CREO SU CARPETA ----------------------------------------
    
    id_modelo = cf_c.leer_contador("contador_de_modelos.txt")
    tipo_de_modelo = "_RedNeuronalGS_"
    variable_carpeta = f"models/{id_modelo}{tipo_de_modelo}{gol_n_ticks}"
    
    # Actualizar contador de modelos
    cf_c.incrementar_contador("contador_de_modelos.txt")
    print("Contador de modelos actualizado.")
    
    
    # ----------------- DEFINO CARPETA DONDE SE GUARDARAN LOS GRAFICOS Y RESULTADOS --------------------------------
    
    os.makedirs(variable_carpeta, exist_ok=True)
    carpeta_resultados = f"{variable_carpeta}/"

    print(f"El id del modelo es {id_modelo}.\n Sus resultados se guardarán en la carpeta: {carpeta_resultados}")
    
    
    # Redirigir los prints a un .out
    output_path = f"{carpeta_resultados}params.out"
     
    # Crear el archivo .out y escribir los resultados
    try:
        with open(output_path, "w") as f:
            
            sys.stdout = f
            
            f.write(f"\nproporcion_none: {proporcion_none}")
            f.write(f"\nestrategia: {estrategia}")
            f.write(f"\ncombinaciones: {combinaciones}")
            f.write(f"\nrandom_state: {random_state}\n")
            
            # Restaurar la salida estándar
            sys.stdout = sys.__stdout__  
            
        # Confirmación de guardado
        print(f"Resultados guardados en {output_path}")
        
        
    except Exception as e:
        print(f"Error al intentar guardar los resultados: {e}")
        
    
    
    # --------------------------------- PREPARACION Y MUESTREO ----------------------------------------------------------

    X_train, X_test, X_val, y_train, y_test, y_val, indices_train, indices_test, indices_val = preparar_datos(
        random_state,
        data_posiciones_grouped_filtered_labeled,
        data_posiciones_filtered,
        gol_n_ticks,
        estrategia, # La estrategia es para ver como preparar o muestrear los datos.
                    # valores posibles: 'original', no hace cambios.
                    #                   'shuffle', mezcla el orden de las 7 entidades en cancha.
                    #                   'metrics', usa las metricas creadas como datos de entrenamiento (para xgBoost y posible RedNeuronal).           
        proporcion_none # Proporcion de None vs otros equipos (2 significa 50% None, 25% Red, 25% Blue).
    )
    
    guardar_split_data(carpeta_resultados, X_train, X_test, X_val, y_train, y_test, y_val, indices_val) 
    
    print('La data del split ha sido guardada')
    
    
    # ------------------------------------- ENTRENAMIENTO ----------------------------------------------------------
    
    # Configurar la búsqueda en cuadrícula con validación cruzada
    random_search = RandomizedSearchCV(
        estimator=MLPClassifier(random_state=random_state),
        param_distributions=custom_param_grid,
        n_iter=combinaciones,  # Número de combinaciones aleatorias a probar
        cv=5,  # Validación cruzada con 5 folds
        n_jobs=6,  # Usar 6 CPUs
        verbose=3,  # Mostrar progreso
        random_state=random_state
    )

    # Ajustar el modelo con los datos
    random_search.fit(X_train, y_train)    

    # Obtener el mejor modelo
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    
    # ---------------------------------- GUARDADO DE MODELOS -------------------------------------------------------

    guardar_modelos_randomsearch(carpeta_resultados, random_search, best_model)
    
    
    print('Listo!')

    

