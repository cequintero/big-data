"""
Análisis exploratorio (EDA) y visualización de SECOP II con Apache Spark.
Lee el dataset procesado (Parquet en HDFS) que genera secop_batch.py.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # la VM no tiene pantalla: las gráficas se guardan como PNG
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ----------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------
ANIO_ANALISIS = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
HDFS_BASE = "hdfs://localhost:9000/secop"
RUTA_PARQUET = f"{HDFS_BASE}/procesado/contratos_{ANIO_ANALISIS}"
RUTA_RESULTADOS = f"{HDFS_BASE}/resultados"
DIR_GRAFICAS = sys.argv[2] if len(sys.argv) > 2 else "/mnt/repo/output/graficas"

INDICADORES = [
    "es_contratacion_directa", "firma_posterior_inicio", "firma_fuera_horario",
    "firma_diciembre", "pagado_mayor_valor",
]


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def titulo(texto):
    print("\n" + "=" * 70)
    print(texto.upper())
    print("=" * 70)


def guardar_resultado(df_res, nombre):
    """Guarda una tabla de resultados en HDFS como CSV."""
    ruta = f"{RUTA_RESULTADOS}/{nombre}"
    df_res.coalesce(1).write.mode("overwrite").option("header", True).csv(ruta)
    print(f"Resultado guardado en HDFS: {ruta}")


def grafica_barras(pdf, col_x, col_y, titulo_g, archivo, horizontal=False, etiqueta=""):
    etiquetas = pdf[col_x].astype(str).str.slice(0, 45)
    fig, ax = plt.subplots(figsize=(11, 6))
    if horizontal:
        ax.barh(etiquetas, pdf[col_y])
        ax.invert_yaxis()
        ax.set_xlabel(etiqueta)
    else:
        ax.bar(etiquetas, pdf[col_y])
        ax.set_ylabel(etiqueta)
        plt.xticks(rotation=45, ha="right")
    ax.set_title(titulo_g)
    fig.tight_layout()
    ruta = os.path.join(DIR_GRAFICAS, archivo)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print(f"Gráfica guardada: {ruta}")


def grafica_linea(pdf, col_x, col_y, titulo_g, archivo, etiqueta_x="", etiqueta_y=""):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(pdf[col_x], pdf[col_y], marker="o")
    ax.set_title(titulo_g)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(DIR_GRAFICAS, archivo)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print(f"Gráfica guardada: {ruta}")


# ----------------------------------------------------------------------
# Sesión de Spark y carga del Parquet
# ----------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("SECOP_II_EDA")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
os.makedirs(DIR_GRAFICAS, exist_ok=True)

df = spark.read.parquet(RUTA_PARQUET).cache()
total = df.count()

# ----------------------------------------------------------------------
# 1. Resumen general
# ----------------------------------------------------------------------
titulo(f"1. Resumen general - contratos firmados en {ANIO_ANALISIS}")
res = df.agg(
    F.sum("valor_del_contrato").alias("valor_total"),
    F.countDistinct("nit_entidad").alias("entidades"),
    F.countDistinct("documento_proveedor").alias("proveedores"),
).collect()[0]
print(f"Contratos:            {total:,}")
print(f"Valor total:          ${res['valor_total']:,.0f}")
print(f"Entidades distintas:  {res['entidades']:,}")
print(f"Proveedores distintos:{res['proveedores']:>,}")

print("\nEstadísticas descriptivas:")
df.select("valor_del_contrato", "duracion_dias", "dias_adicionados") \
  .summary("count", "mean", "min", "25%", "50%", "75%", "max").show(truncate=False)

# ----------------------------------------------------------------------
# 2. Contratos por modalidad de contratación
# ----------------------------------------------------------------------
titulo("2. Contratos por modalidad de contratación")
por_modalidad = (
    df.groupBy("modalidad_de_contratacion")
    .agg(F.count("*").alias("contratos"), F.sum("valor_del_contrato").alias("valor_total"))
    .withColumn("pct_contratos", F.round(F.col("contratos") * 100 / total, 2))
    .orderBy(F.desc("contratos"))
)
por_modalidad.select(
    "modalidad_de_contratacion", "contratos", "pct_contratos",
    F.format_number("valor_total", 0).alias("valor_total"),
).show(truncate=False)
grafica_barras(por_modalidad.toPandas(), "modalidad_de_contratacion", "contratos",
               f"Contratos por modalidad de contratación ({ANIO_ANALISIS})",
               "01_contratos_por_modalidad.png", horizontal=True, etiqueta="Número de contratos")
guardar_resultado(por_modalidad, "por_modalidad")

# ----------------------------------------------------------------------
# 3. Contratos por mes
# ----------------------------------------------------------------------
titulo("3. Contratos firmados por mes")
por_mes = (
    df.groupBy("mes_firma")
    .agg(F.count("*").alias("contratos"), F.sum("valor_del_contrato").alias("valor_total"))
    .orderBy("mes_firma")
)
por_mes.select("mes_firma", "contratos", F.format_number("valor_total", 0).alias("valor_total")) \
       .show(12, truncate=False)
grafica_linea(por_mes.toPandas(), "mes_firma", "contratos",
              f"Contratos firmados por mes ({ANIO_ANALISIS})", "02_contratos_por_mes.png",
              "Mes", "Número de contratos")
guardar_resultado(por_mes, "por_mes")

# ----------------------------------------------------------------------
# 4. Top 10 departamentos
# ----------------------------------------------------------------------
titulo("4. Top 10 departamentos por número de contratos")
por_depto = (
    df.groupBy("departamento")
    .agg(F.count("*").alias("contratos"), F.sum("valor_del_contrato").alias("valor_total"))
    .orderBy(F.desc("contratos"))
)
por_depto.select("departamento", "contratos", F.format_number("valor_total", 0).alias("valor_total")) \
         .show(10, truncate=False)
guardar_resultado(por_depto, "por_departamento")

# ----------------------------------------------------------------------
# 5. Indicadores de riesgo
# ----------------------------------------------------------------------
titulo("5.1 Indicadores de riesgo (contratos y porcentaje)")
# Suma cada indicador y convierte las columnas en filas con stack (todo se ejecuta en Spark)
expr_stack = "stack({}, {}) as (indicador, contratos)".format(
    len(INDICADORES), ", ".join(f"'{c}', {c}" for c in INDICADORES)
)
indicadores = (
    df.agg(*[F.sum(c).cast("long").alias(c) for c in INDICADORES])
    .selectExpr(expr_stack)
    .withColumn("porcentaje", F.round(F.col("contratos") * 100 / total, 2))
)
indicadores.show(truncate=False)
grafica_barras(indicadores.toPandas(), "indicador", "porcentaje",
               f"Porcentaje de contratos por indicador ({ANIO_ANALISIS})",
               "03_indicadores_riesgo.png", horizontal=True, etiqueta="% de contratos")

titulo("5.2 Distribución de firmas por hora del día")
por_hora = df.groupBy("hora_firma").count().orderBy("hora_firma")
por_hora.show(24)
grafica_barras(por_hora.toPandas(), "hora_firma", "count",
               f"Contratos firmados por hora del día ({ANIO_ANALISIS})",
               "04_firmas_por_hora.png", etiqueta="Número de contratos")

titulo("5.3 Concentración: top 10 proveedores por número de contratos")
top_proveedores = (
    df.filter(F.col("documento_proveedor").isNotNull())
    .groupBy("documento_proveedor")
    .agg(
        F.first("proveedor_adjudicado", ignorenulls=True).alias("proveedor"),
        F.count("*").alias("contratos"),
        F.countDistinct("nit_entidad").alias("entidades"),
        F.sum("valor_del_contrato").alias("valor_total"),
    )
    .orderBy(F.desc("contratos"))
    .limit(10)
)
top_proveedores.select(
    "proveedor", "documento_proveedor", "contratos", "entidades",
    F.format_number("valor_total", 0).alias("valor_total"),
).show(truncate=40)
grafica_barras(top_proveedores.toPandas(), "proveedor", "contratos",
               f"Top 10 proveedores por número de contratos ({ANIO_ANALISIS})",
               "05_top_proveedores.png", horizontal=True, etiqueta="Número de contratos")
guardar_resultado(top_proveedores, "top_proveedores")

titulo("5.4 Concentración: proveedores con 30% o más de los contratos de una entidad")
por_entidad = df.groupBy("nit_entidad").agg(F.count("*").alias("contratos_entidad"))
pares = (
    df.filter(F.col("documento_proveedor").isNotNull())
    .groupBy("nit_entidad", "documento_proveedor")
    .agg(
        F.first("nombre_entidad", ignorenulls=True).alias("entidad"),
        F.first("proveedor_adjudicado", ignorenulls=True).alias("proveedor"),
        F.count("*").alias("contratos_par"),
        F.sum("valor_del_contrato").alias("valor_par"),
    )
    .join(por_entidad, "nit_entidad")
    .withColumn("pct_de_la_entidad",
                F.round(F.col("contratos_par") * 100 / F.col("contratos_entidad"), 1))
)
concentracion = (
    pares.filter((F.col("contratos_entidad") >= 20) & (F.col("pct_de_la_entidad") >= 30))
    .orderBy(F.desc("pct_de_la_entidad"), F.desc("contratos_par"))
)
print(f"Casos encontrados (entidades con 20+ contratos): {concentracion.count():,}")
concentracion.select(
    "entidad", "proveedor", "contratos_par", "contratos_entidad", "pct_de_la_entidad",
).show(10, truncate=35)
guardar_resultado(concentracion, "concentracion_entidad_proveedor")

titulo("5.5 Posible fraccionamiento: 3+ contratos misma entidad-proveedor el mismo día")
fraccionamiento = (
    df.filter(F.col("documento_proveedor").isNotNull())
    .groupBy("nit_entidad", "documento_proveedor", "dia_firma")
    .agg(
        F.first("nombre_entidad", ignorenulls=True).alias("entidad"),
        F.first("proveedor_adjudicado", ignorenulls=True).alias("proveedor"),
        F.count("*").alias("contratos_mismo_dia"),
        F.sum("valor_del_contrato").alias("valor_total"),
    )
    .filter(F.col("contratos_mismo_dia") >= 3)
    .orderBy(F.desc("contratos_mismo_dia"))
)
print(f"Casos encontrados: {fraccionamiento.count():,}")
fraccionamiento.select(
    "entidad", "proveedor", "dia_firma", "contratos_mismo_dia",
    F.format_number("valor_total", 0).alias("valor_total"),
).show(10, truncate=35)
guardar_resultado(fraccionamiento, "posible_fraccionamiento")

titulo("5.6 Contratos a revisar (2 o más alertas simultáneas)")
casos = (
    df.withColumn(
        "puntaje_riesgo",
        F.col("firma_posterior_inicio") + F.col("firma_fuera_horario") + F.col("pagado_mayor_valor")
        + F.when((F.col("es_contratacion_directa") == 1)
                 & (F.col("valor_del_contrato") >= 500e6), 1).otherwise(0),
    )
    .filter(F.col("puntaje_riesgo") >= 2)
    .orderBy(F.desc("puntaje_riesgo"), F.desc("valor_del_contrato"))
)
print(f"Contratos con 2 o más alertas: {casos.count():,}")
casos.select(
    "id_contrato", "nombre_entidad", "proveedor_adjudicado", "puntaje_riesgo",
    F.format_number("valor_del_contrato", 0).alias("valor"),
).show(10, truncate=30)
guardar_resultado(
    casos.select("id_contrato", "nombre_entidad", "proveedor_adjudicado", "modalidad_de_contratacion",
                 "fecha_de_firma", "valor_del_contrato", "puntaje_riesgo"),
    "contratos_a_revisar",
)

spark.stop()