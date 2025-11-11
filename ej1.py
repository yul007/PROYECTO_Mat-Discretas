import numpy as np
import pandas as pd
from typing import Tuple, List 

# --------------------- UTILIDADES ---------------------

def generar_matriz_booleana(filas: int, columnas: int, prob_uno: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Genera una matriz booleana aleatoria (0/1) de tamaño filas x columnas.
    - prob_uno: probabilidad de que una entrada sea 1.
    - seed: para reproducibilidad.
    """
    if seed is not None:
        np.random.seed(seed)
    return (np.random.rand(filas, columnas) < prob_uno).astype(int)


def es_subfila(r_q: np.ndarray, r_p: np.ndarray, fila_q: int, fila_p: int) -> Tuple[bool, List[int], List[int]]:
    """
    Determina si r_q es subfila de r_p según la definición exacta.
    Devuelve: (es_subfila, columnas_donde_q_menos_p, columnas_donde_p_mas_q)
    """
    # 1. r_q <= r_p en todas las posiciones (donde q=1 → p=1)
    columnas_donde_q_tiene_1 = np.where(r_q == 1)[0]
    columnas_donde_p_no_tiene_1 = np.where(r_p == 0)[0]
    
    # Verificar condición 1: donde q=1, p también debe tener 1
    violaciones_cond1 = np.intersect1d(columnas_donde_q_tiene_1, columnas_donde_p_no_tiene_1)
    if len(violaciones_cond1) > 0:
        return False, [], []
    
    # 2. Existe al menos una columna donde p=1 y q=0
    columnas_donde_p_tiene_1 = np.where(r_p == 1)[0]
    columnas_donde_q_tiene_0 = np.where(r_q == 0)[0]
    columnas_donde_p_mas_q = np.intersect1d(columnas_donde_p_tiene_1, columnas_donde_q_tiene_0)
    
    return len(columnas_donde_p_mas_q) > 0, [], list(columnas_donde_p_mas_q)


def analizar_subfilas(MD: np.ndarray) -> List[Tuple[int, int, str, List[int]]]:
    """
    Analiza todas las comparaciones de subfilas y devuelve explicaciones.
    Formato: (fila_q, fila_p, "es_subfila" o "no_es_subfila", columnas_explicativas)
    """
    m = MD.shape[0]
    comparaciones = []
    
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            es_sub, _, columnas_donde_p_mas_q = es_subfila(MD[i], MD[j], i, j)
            
            razon = "SÍ" if es_sub else "NO"
            if columnas_donde_p_mas_q:
                columnas_str = [str(c) for c in columnas_donde_p_mas_q]
                explicacion = f"Col {', '.join(columnas_str)}: F{j} tiene 1 donde F{i} tiene 0"
            else:
                explicacion = "F{i} no es estrictamente menor que F{j}"
            
            comparaciones.append((i, j, razon, explicacion))
    
    return comparaciones


def construir_matriz_basica_con_explicacion(MD: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, str, str]]]:
    """
    Construye la Matriz Básica (MB) y explica por qué cada fila se mantiene o elimina.
    """
    m = MD.shape[0]
    es_basica = np.ones(m, dtype=bool)
    eliminaciones = []  # (fila_eliminada, fila_dominante, explicación)
    
    for i in range(m):
        if not es_basica[i]:
            continue
        
        for j in range(m):
            if i == j or not es_basica[j]:
                continue
            
            # Si la fila i es subfila de la fila j → eliminar i
            es_sub, _, columnas_donde_j_mas_i = es_subfila(MD[i], MD[j], i, j)
            if es_sub:
                razon = f"Es subfila de F{j} (Col {', '.join([str(c) for c in columnas_donde_j_mas_i])}: F{j} tiene 1 donde F{i} tiene 0)"
                eliminaciones.append((i, j, razon))
                es_basica[i] = False
                break  # una sola razón es suficiente
    
    return MD[es_basica], eliminaciones


def mostrar_matriz_con_filas_numeradas(matriz: np.ndarray, titulo: str):
    """Muestra la matriz con números de fila para mejor legibilidad."""
    df = pd.DataFrame(matriz)
    df.index = [f"F{i}" for i in range(matriz.shape[0])]
    print(f"\n>>> {titulo}")
    print(f"Dimensiones: {matriz.shape[0]} filas × {matriz.shape[1]} columnas")
    print(df)
    print()


def mostrar_analisis_subfilas(eliminaciones: List[Tuple[int, str, str]]):
    """Muestra un análisis detallado de por qué se eliminaron filas."""
    if not eliminaciones:
        print("\n>>> ANÁLISIS DE SUBFILAS")
        print("✓ Todas las filas son básicas (ninguna es subfila de otra).")
        return
    
    print("\n>>> ANÁLISIS DE SUBFILAS")
    print("FILA | ¿Es básica? | Razón de eliminación")
    print("-" * 40)
    
    filas_procesadas = set()
    for fila_eliminada, fila_dominante, razon in eliminaciones:
        if fila_eliminada not in filas_procesadas:
            filas_procesadas.add(fila_eliminada)
            estado = "ELIMINADA" if not es_basica[fila_eliminada] else "BÁSICA"
            print(f"F{fila_eliminada:2d} | {estado:8s} | {razon}")
    
    filas_basicas = [i for i in range(MD.shape[0]) if es_basica[i]]
    print(f"\n✓ Filas básicas finales: {', '.join([f'F{i}' for i in filas_basicas])}")
    print(f"✗ Filas eliminadas: {', '.join([f'F{i}' for i in filas_procesadas])}")


def calcular_densidad(matriz: np.ndarray) -> float:
    """Densidad = número de 1's / total de entradas"""
    if matriz.size == 0:
        return 0.0
    return np.sum(matriz == 1) / matriz.size


# --------------------- ENTRADA DEL USUARIO ---------------------

print("=== EJERCICIO 1: GENERACIÓN DE MATRIZ BOOLEANA Y MATRIZ BÁSICA ===\n")

# Validación de filas
while True:
    try:
        filas = int(input("Ingresa el número de filas (1 a 100): "))
        if 1 <= filas <= 100:
            break
        else:
            print("Error: debe estar entre 1 y 100.")
    except ValueError:
        print("Error: ingresa un número entero.")

# Validación de columnas
while True:
    try:
        columnas = int(input("Ingresa el número de columnas (1 a 10): "))
        if 1 <= columnas <= 10:
            break
        else:
            print("Error: debe estar entre 1 y 10.")
    except ValueError:
        print("Error: ingresa un número entero.")

# Probabilidad de 1
while True:
    try:
        prob_input = input("Probabilidad de 1 (0.0 a 1.0, Enter para 0.5): ").strip()
        prob_uno = 0.5 if prob_input == "" else float(prob_input)
        if 0.0 <= prob_uno <= 1.0:
            break
        else:
            print("Error: debe estar entre 0.0 y 1.0.")
    except ValueError:
        print("Error: ingresa un número válido.")

# Seed opcional
seed_input = input("Seed para reproducibilidad (Enter para aleatorio): ").strip()
seed = int(seed_input) if seed_input.isdigit() else None

# --------------------- GENERACIÓN Y CÁLCULO ---------------------

print(f"\nGenerando Matriz de Diferencias (MD) de {filas}×{columnas}...")
print(f"Probabilidad de 1: {prob_uno}, Seed: {seed}\n")

MD = generar_matriz_booleana(filas, columnas, prob_uno, seed)
es_basica = np.ones(MD.shape[0], dtype=bool)  # Variable global para mostrar_analisis_subfilas

# --------------------- RESULTADOS ---------------------

mostrar_matriz_con_filas_numeradas(MD, "MATRIZ DE DIFERENCIAS (MD)")

MB, eliminaciones = construir_matriz_basica_con_explicacion(MD)
mostrar_matriz_con_filas_numeradas(MB, "MATRIZ BÁSICA (MB)")

# ANÁLISIS DETALLADO DE SUBFILAS
mostrar_analisis_subfilas(eliminaciones)

# DENSIDADES
densidad_MD = calcular_densidad(MD)
densidad_MB = calcular_densidad(MB)

print("\n>>> DENSIDADES")
print(f"Densidad de MD = {densidad_MD:.4f}  ({np.sum(MD == 1)} unos de {MD.size} entradas)")
print(f"Densidad de MB = {densidad_MB:.4f}  ({np.sum(MB == 1)} unos de {MB.size} entradas)")

# Guardar resultados
# import os
# os.makedirs("salidas_ejercicio1", exist_ok=True)
# pd.DataFrame(MD).to_csv("salidas_ejercicio1/MD.csv", index=False)
# pd.DataFrame(MB).to_csv("salidas_ejercicio1/MB.csv", index=False)

# Guardar análisis
# analisis_df = pd.DataFrame(eliminaciones, columns=["Fila eliminada", "Fila dominante", "Razón"])
# analisis_df.to_csv("salidas_ejercicio1/analisis_subfilas.csv", index=False)

# print(f"\nArchivos guardados en: salidas_ejercicio1/")
# print("  - MD.csv")
# print("  - MB.csv") 
# print("  - analisis_subfilas.csv")
