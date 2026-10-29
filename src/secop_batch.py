"""
Análisis batch de SECOP II - Contratos Electrónicos con Apache Spark.
Detección inicial de indicadores de comportamientos potencialmente sospechosos.
"""
import sys
import time
import re
import unicodedata

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ----------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------
RUTA_DEFECTO = "file:///home/vboxuser/secop_dev.csv"
RUTA_ENTRADA = sys.argv[1] if len(sys.argv) > 1 else RUTA_DEFECTO
HDFS_BASE = "hdfs://localhost:9000/secop"
SEPARADOR = ","   # cámbialo a "\t" si el archivo viene separado por tabulaciones
ANIO_ANALISIS = 2025   # año que se analiza ("el año pasado")
SUFIJO_SALIDA = sys.argv[2] if len(sys.argv) > 2 else ""   # ej: "_muestra" para pruebas

# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def titulo(texto):
    """Imprime un título de sección para que la consola quede ordenada."""
    print("\n" + "=" * 70)
    print(texto.upper())
    print("=" * 70)


def normalizar(nombre):
    """'Modalidad de Contratación ' -> 'modalidad_de_contratacion'"""
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip().lower())
    return s.strip("_")


# ----------------------------------------------------------------------
# Sesión de Spark
# ----------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("SECOP_II_Batch")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ----------------------------------------------------------------------
# 1. Carga del dataset
# ----------------------------------------------------------------------
titulo("1. Carga del dataset")
print(f"Ruta de entrada: {RUTA_ENTRADA}")

df_raw = (
    spark.read
    .option("header", True)
    .option("sep", SEPARADOR)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("encoding", "UTF-8")
    .csv(RUTA_ENTRADA)
)

titulo("1.1 Esquema")
df_raw.printSchema()
print(f"Número de columnas: {len(df_raw.columns)}")

titulo("1.2 Filas de ejemplo")
columnas_interes = [
    "nombre_entidad", "departamento", "modalidad_de_contratacion",
    "fecha_de_firma", "valor_del_contrato", "proveedor_adjudicado",
]
# Relaciona el nombre normalizado con el nombre original del CSV
mapa_columnas = {normalizar(c): c for c in df_raw.columns}
encontradas = [mapa_columnas[c] for c in columnas_interes if c in mapa_columnas]
faltantes = [c for c in columnas_interes if c not in mapa_columnas]

if encontradas:
    df_raw.select(encontradas).show(5, truncate=40)
if faltantes:
    print(f"Columnas no encontradas: {faltantes}")

titulo("1.3 Conteo de registros")
t0 = time.time()
total_registros = df_raw.count()
print(f"Total de registros: {total_registros:,}  (tiempo: {time.time() - t0:.1f} s)")





# ----------------------------------------------------------------------
# 2. Selección y normalización de columnas
# ----------------------------------------------------------------------
titulo("2. Selección de columnas")

# Renombra todas las columnas al formato normalizado (ej: "Valor del Contrato" -> valor_del_contrato)
df = df_raw.toDF(*[normalizar(c) for c in df_raw.columns])

COLUMNAS_ANALISIS = [
    "nombre_entidad", "nit_entidad", "departamento", "ciudad", "orden", "sector",
    "proceso_de_compra", "id_contrato", "estado_contrato",
    "codigo_de_categoria_principal", "tipo_de_contrato",
    "modalidad_de_contratacion", "justificacion_modalidad_de_contratacion",
    "fecha_de_firma", "fecha_de_inicio_del_contrato", "fecha_de_fin_del_contrato",
    "tipodocproveedor", "documento_proveedor", "proveedor_adjudicado",
    "es_pyme", "es_grupo", "origen_de_los_recursos",
    "valor_del_contrato", "valor_pagado", "dias_adicionados",
]
presentes = [c for c in COLUMNAS_ANALISIS if c in df.columns]
ausentes = [c for c in COLUMNAS_ANALISIS if c not in df.columns]

df = df.select(presentes)
print(f"Columnas originales: {len(df_raw.columns)}  ->  seleccionadas: {len(presentes)}")
if ausentes:
    print(f"Columnas no encontradas (se omiten): {ausentes}")

# ----------------------------------------------------------------------
# 3. Limpieza y transformación de tipos
# ----------------------------------------------------------------------
titulo("3. Limpieza y transformación")

# Evita errores del parser de fechas de Spark 3 con formatos antiguos
spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")

# Valores que en SECOP significan "vacío"
VALORES_VACIOS = ["", "no definido", "sin descripcion", "no aplica", "null"]


def limpiar_texto(c):
    """Quita espacios repetidos, entidades HTML y convierte 'No Definido' en nulo."""
    col = F.trim(F.regexp_replace(F.col(c), r"\s+", " "))
    col = F.regexp_replace(col, "&#39;", "'")
    col = F.regexp_replace(col, "&amp;", "&")
    return F.when(F.lower(col).isin(VALORES_VACIOS), None).otherwise(col)


def a_numero(c):
    """'$12,499,480' -> 12499480.0"""
    return F.regexp_replace(F.col(c), r"[$,\s]", "").cast("double")


# 3.1 Limpieza general de textos
for c in presentes:
    df = df.withColumn(c, limpiar_texto(c))

# 3.2 Mayúsculas en columnas categóricas y nombres (para agrupar sin duplicados)
COLS_MAYUS = [
    "nombre_entidad", "departamento", "ciudad", "orden", "sector",
    "estado_contrato", "tipo_de_contrato", "modalidad_de_contratacion",
    "proveedor_adjudicado", "origen_de_los_recursos",
]
for c in COLS_MAYUS:
    if c in df.columns:
        df = df.withColumn(c, F.upper(F.col(c)))

# 3.3 Identificadores: solo letras y números ("901275780-5" -> "9012757805")
for c in ["documento_proveedor", "nit_entidad"]:
    df = df.withColumn(c, F.upper(F.regexp_replace(F.col(c), r"[^0-9A-Za-z]", "")))

# 3.4 Valores monetarios y numéricos
df = (
    df.withColumn("valor_del_contrato", a_numero("valor_del_contrato"))
      .withColumn("valor_pagado", a_numero("valor_pagado"))
      .withColumn("dias_adicionados", a_numero("dias_adicionados").cast("int"))
)

# 3.5 Fechas
FORMATO_FECHA = "MM/dd/yyyy hh:mm:ss a"
for c in ["fecha_de_firma", "fecha_de_inicio_del_contrato", "fecha_de_fin_del_contrato"]:
    df = df.withColumn(c, F.to_timestamp(F.col(c), FORMATO_FECHA))

print("Tipos convertidos: valores -> double, días -> int, fechas -> timestamp")

# ----------------------------------------------------------------------
# 4. Filtrado: embudo de registros
# ----------------------------------------------------------------------
titulo(f"4. Filtrado de registros (año {ANIO_ANALISIS})")

cond_fecha = F.col("fecha_de_firma").isNotNull()
cond_anio = cond_fecha & (F.year("fecha_de_firma") == ANIO_ANALISIS)
cond_valido = cond_anio & F.col("id_contrato").isNotNull() & (F.col("valor_del_contrato") > 0)

# Una sola pasada sobre los datos calcula todas las etapas del embudo
t0 = time.time()
embudo = df.agg(
    F.count(F.lit(1)).alias("total"),
    F.sum(F.when(cond_fecha, 1).otherwise(0)).alias("con_fecha"),
    F.sum(F.when(cond_anio, 1).otherwise(0)).alias("del_anio"),
    F.sum(F.when(cond_valido, 1).otherwise(0)).alias("validos"),
).collect()[0]
print(f"(tiempo del embudo: {time.time() - t0:.1f} s)\n")

print(f"{'Registros totales en el CSV':48s}{embudo['total']:>14,}")
print(f"{'Con fecha de firma válida':48s}{embudo['con_fecha']:>14,}")
print(f"{'Firmados en ' + str(ANIO_ANALISIS):48s}{embudo['del_anio']:>14,}")
print(f"{'Con id de contrato y valor > 0':48s}{embudo['validos']:>14,}")

df_filtrado = df.filter(cond_valido).dropDuplicates(["id_contrato"])

# ----------------------------------------------------------------------
# 5. Variables derivadas (indicadores de riesgo)
# ----------------------------------------------------------------------
titulo("5. Variables derivadas")

df_final = (
    df_filtrado
    .withColumn("anio_firma", F.year("fecha_de_firma"))
    .withColumn("mes_firma", F.month("fecha_de_firma"))
    .withColumn("dia_firma", F.to_date("fecha_de_firma"))
    .withColumn("hora_firma", F.hour("fecha_de_firma"))
    .withColumn("dia_semana", F.dayofweek("fecha_de_firma"))  # 1=domingo ... 7=sábado
    .withColumn("es_contratacion_directa",
                F.when(F.col("modalidad_de_contratacion").contains("DIRECTA"), 1).otherwise(0))
    .withColumn("firma_posterior_inicio",
                F.when(F.to_date("fecha_de_firma") > F.to_date("fecha_de_inicio_del_contrato"), 1).otherwise(0))
    .withColumn("firma_fuera_horario",
                F.when((F.col("hora_firma") < 7) | (F.col("hora_firma") >= 19)
                       | F.col("dia_semana").isin(1, 7), 1).otherwise(0))
    .withColumn("firma_diciembre", F.when(F.col("mes_firma") == 12, 1).otherwise(0))
    .withColumn("pagado_mayor_valor",
                F.when(F.col("valor_pagado") > F.col("valor_del_contrato"), 1).otherwise(0))
    .withColumn("duracion_dias",
                F.datediff("fecha_de_fin_del_contrato", "fecha_de_inicio_del_contrato"))
    .withColumn("rango_valor",
                F.when(F.col("valor_del_contrato") < 50e6, "1. Menos de 50 M")
                 .when(F.col("valor_del_contrato") < 500e6, "2. 50 M - 500 M")
                 .when(F.col("valor_del_contrato") < 5e9, "3. 500 M - 5.000 M")
                 .otherwise("4. Más de 5.000 M"))
)
print("Indicadores creados: firma_posterior_inicio, firma_fuera_horario, firma_diciembre,")
print("es_contratacion_directa, pagado_mayor_valor, duracion_dias, rango_valor")

# ----------------------------------------------------------------------
# 6. Almacenamiento del dataset procesado en HDFS (Parquet)
# ----------------------------------------------------------------------
titulo("6. Almacenamiento en HDFS (Parquet)")

RUTA_PARQUET = f"{HDFS_BASE}/procesado/contratos_{ANIO_ANALISIS}{SUFIJO_SALIDA}"
t0 = time.time()
df_final.write.mode("overwrite").parquet(RUTA_PARQUET)
print(f"Guardado en: {RUTA_PARQUET}  (tiempo: {time.time() - t0:.1f} s)")

# A partir de aquí se trabaja sobre el Parquet (mucho más rápido que el CSV)
df = spark.read.parquet(RUTA_PARQUET).cache()
registros_finales = df.count()
print(f"{'Registros únicos (sin duplicados de contrato)':48s}{registros_finales:>14,}")

titulo("6.1 Esquema del dataset procesado")
df.printSchema()

titulo("6.2 Muestra del dataset procesado")
df.select(
    "nombre_entidad", "modalidad_de_contratacion", "fecha_de_firma",
    "valor_del_contrato", "proveedor_adjudicado", "firma_posterior_inicio",
    "firma_fuera_horario",
).show(5, truncate=30)

titulo("6.3 Verificación rápida de indicadores")
df.select(
    F.sum("es_contratacion_directa").alias("directa"),
    F.sum("firma_posterior_inicio").alias("firma_post_inicio"),
    F.sum("firma_fuera_horario").alias("fuera_horario"),
    F.sum("firma_diciembre").alias("diciembre"),
    F.sum("pagado_mayor_valor").alias("pagado_mayor"),
).show()

# ----------------------------------------------------------------------
# Cierre
# ----------------------------------------------------------------------
spark.stop()