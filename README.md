# Recopilación de Datos de Partidos de HaxBall, Análisis de Datos y Modelado

Este proyecto incluye la recopilación de datos de partidos de HaxBall, su análisis y modelado. 
Los datos fueron generados en partidos que ocurrieron en mi servidor, donde usuarios desconocidos entraron a jugar partidos 3vs3, a 3 goles o 180 segundos (o tiempo extra en caso de empate).
En los notebooks en la carpeta `/modelado` se proveen mas detalles de los datos y archivos **.csv** con los datos que se utilizaron, en las carpetas `/modelado/filtered_data` y `/modelado/raw_data`.

## Estructura del Proyecto

### /backend
Archivo JavaScript que envía datos a una base de datos en PostgreSQL.

### /maps
Archivo del mapa (estadio) utilizado en el servidor.

### /modelado
Notebooks donde se obtienen, analizan y filtran los datos y donde se crean los modelos.

#### /filtered_data
Archivos .csv con todos los datos filtrados en el notebook `analisis_y_filtrado`.

#### /raw_data
Archivos .csv con todos los datos crudos, directo de la base de datos, sin ser filtrados.

### /servidor_online
Archivo JavaScript que genera el servidor de HaxBall. Este archivo debe ejecutarse en la consola del navegador.

## Recomendaciones

- Usar **Microsoft Edge** para ejecutar el archivo `server.js` del servidor de HaxBall en la consola del navegador.

## Instalación y Configuración (reescribir esta seccion con lo del Notion, esto es un vago resumen)

1. **Backend**:
   - Asegúrate de tener Node.js instalado.
   - Instala las dependencias necesarias ejecutando `npm install` en la carpeta `/backend`.
   - Configura las variables de entorno en un archivo `.env` en la carpeta `/backend`.
   - Inicia el servidor con `node server.js`.

2. **Base de Datos**:
   - Configura y asegúrate de tener una base de datos PostgreSQL.
   - Actualiza las credenciales en el archivo `.env`.

3. **Modelado**:
   - Usa Jupyter Notebooks para abrir y ejecutar los notebooks en la carpeta `/modelado`.

## Uso

1. **Recopilación de Datos**:
   - Ejecuta el archivo del servidor de HaxBall desde la consola del navegador para recopilar datos.
   
2. **Análisis y Filtrado**:
   - Utiliza los notebooks en la carpeta `/modelado` para analizar y filtrar los datos.
   - Los datos filtrados se guardarán en la subcarpeta `/filtered_data`.

3. **Modelado**:
   - Crea y entrena modelos utilizando los datos filtrados en los notebooks dentro de la carpeta `/modelado`.

## Licencia
