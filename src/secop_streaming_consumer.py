"""
Consumidor Spark Structured Streaming: lee contratos SECOP II desde Kafka y calcula
en tiempo real estadísticas por hora y alertas de contratos de alto valor o fuera de horario.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

UMBRAL_ALTO_VALOR = 500_000_000  # 500 millones de pesos

spark = (
    SparkSession.builder
    .appName("SECOP_II_Streaming")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

esquema = StructType([
    StructField("id_contrato", StringType()),
    StructField("entidad", StringType()),
    StructField("departamento", StringType()),
    StructField("modalidad", StringType()),
    StructField("valor", DoubleType()),
    StructField("proveedor", StringType()),
    StructField("fecha_firma", StringType()),
])

# Lectura del topic de Kafka y conversión del JSON a columnas
eventos = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "secop_contratos")
    .option("startingOffsets", "latest")
    .load()
    .select(F.from_json(F.col("value").cast("string"), esquema).alias("e"))
    .select("e.*")
    .withColumn("fecha_firma", F.to_timestamp("fecha_firma"))
    .withColumn("hora", F.hour("fecha_firma"))
    .withColumn("fuera_horario", F.when((F.col("hora") < 7) | (F.col("hora") >= 19), 1).otherwise(0))
    .withColumn("directa", F.when(F.upper("modalidad").contains("DIRECTA"), 1).otherwise(0))
)

# Consulta 1: estadísticas acumuladas por hora del día simulado (ventanas de 1 hora)
por_hora = (
    eventos.groupBy(F.window("fecha_firma", "1 hour").alias("ventana"))
    .agg(
        F.count("*").alias("contratos"),
        F.round(F.sum("valor") / 1e6, 1).alias("valor_millones"),
        F.sum("directa").alias("directa"),
        F.sum("fuera_horario").alias("fuera_horario"),
    )
    .select(F.date_format("ventana.start", "HH:mm").alias("hora"),
            "contratos", "valor_millones", "directa", "fuera_horario")
    .orderBy("hora")
)

q_por_hora = (
    por_hora.writeStream.outputMode("complete").format("console")
    .option("truncate", False).option("numRows", 30)
    .trigger(processingTime="10 seconds").queryName("estadisticas_por_hora")
    .start()
)

# Consulta 2: alertas en tiempo real (alto valor o firma fuera de horario)
alertas = (
    eventos.filter((F.col("valor") >= UMBRAL_ALTO_VALOR) | (F.col("fuera_horario") == 1))
    .withColumn("motivo", F.when(F.col("valor") >= UMBRAL_ALTO_VALOR, "ALTO VALOR")
                           .otherwise("FUERA DE HORARIO"))
    .select(F.date_format("fecha_firma", "HH:mm:ss").alias("hora"), "motivo",
            F.substring("entidad", 1, 40).alias("entidad"),
            F.format_number("valor", 0).alias("valor"))
)

q_alertas = (
    alertas.writeStream.outputMode("append").format("console")
    .option("truncate", False)
    .trigger(processingTime="10 seconds").queryName("alertas")
    .start()
)

spark.streams.awaitAnyTermination()