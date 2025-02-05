# Recopilación de Datos de Partidos de HaxBall, Análisis de Datos y Modelado

Este proyecto incluye la recopilación de datos de partidos del juego online [HaxBall](https://www.haxball.com/play), su análisis exploratorio y modelado. 
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

## Instalación y Configuración

1. **Backend**:
   - Instalar Node.js.
   - Instalar las dependencias necesarias con el comando `npm install` en la carpeta `/backend` desde tu consola.
   - Declarar las variables de entorno en un archivo `.env` dentro de la carpeta `/backend`.
   - Iniciar el servidor con `node server.js` desde tu consola.

2. **Base de Datos**:
   - Configurar una base de datos PostgreSQL.
   - Actualizar las credenciales en el archivo `.env`.

3. **Modelado**:
   - Usar Jupyter Notebooks para ejecutar los notebooks de la carpeta `/modelado`.

## Uso

1. **Recopilación de Datos**:
   - Ejecuta el archivo del servidor de HaxBall desde la consola del navegador para recopilar datos.
   
2. **Análisis y Filtrado**:
   - Utiliza los notebooks en la carpeta `/modelado` para analizar y filtrar los datos.
   - Los datos filtrados se guardarán en la subcarpeta `/filtered_data`.

3. **Modelado**:
   - Crea y entrena modelos utilizando los datos filtrados en los notebooks dentro de la carpeta `/modelado`.

## Estructura de los datos
1. **Servidor Haxball Headless (JavaScript)**:
   - Captura los eventos del juego (`onGameTick`, `onPlayerChat`, etc.).
   - Recolecta los datos necesarios: posiciones de jugadores, pelota, y otros eventos.
   - Envía estos datos a un servidor backend mediante una solicitud HTTP o WebSocket.

2. **Servidor Backend (Node.js + SQL)**:
   - Escucha los datos que envía el servidor de Haxball.
   - Inserta los datos en una base de datos SQL.

3. **Base de Datos SQL**:
   - Guarda los datos del partido, jugadores, equipos y posiciones.
  








