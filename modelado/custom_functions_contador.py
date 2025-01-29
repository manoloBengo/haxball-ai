import os

# --------------- Para leer el contador desde el archivo ----------------------

def leer_contador(archivo):
    if os.path.exists(archivo):
        with open(archivo, "r") as f:
            return int(f.read().strip())
    else:
        return 0



# --------------- Para incrementar y guardar el contador ----------------------

def incrementar_contador(archivo):
    contador = leer_contador(archivo) + 1
    with open(archivo, "w") as f:
        f.write(str(contador))
    return contador


