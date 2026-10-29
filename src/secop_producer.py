"""
Productor Kafka: simula la llegada en tiempo real de los contratos SECOP II
firmados en un día de 2025, respetando el orden y la hora de firma (tiempo acelerado).
Uso: python3 secop_producer.py <archivo_csv> <factor_aceleracion>
"""
import csv
import json
import sys
import time
from datetime import datetime

from kafka import KafkaProducer

ARCHIVO = sys.argv[1] if len(sys.argv) > 1 else "/home/vboxuser/secop_streaming/contratos_dia.csv"
FACTOR = float(sys.argv[2]) if len(sys.argv) > 2 else 600  # 1 s real = 600 s simulados (10 min)
TOPIC = "secop_contratos"

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode("utf-8"),
)

with open(ARCHIVO, encoding="utf-8", newline="") as f:
    contratos = list(csv.DictReader(f))

print(f"Contratos a enviar: {len(contratos)}  |  Aceleración: x{FACTOR:.0f}")

anterior = None
for c in contratos:
    momento = datetime.strptime(c["fecha_firma"], "%Y-%m-%d %H:%M:%S")
    # Espera proporcional al tiempo real entre una firma y la siguiente
    if anterior is not None:
        espera = (momento - anterior).total_seconds() / FACTOR
        if espera > 0:
            time.sleep(espera)
    anterior = momento

    evento = {
        "id_contrato": c["id_contrato"],
        "entidad": c["nombre_entidad"],
        "departamento": c["departamento"],
        "modalidad": c["modalidad_de_contratacion"],
        "valor": float(c["valor_del_contrato"] or 0),
        "proveedor": c["proveedor_adjudicado"],
        "fecha_firma": c["fecha_firma"],
    }
    producer.send(TOPIC, value=evento)
    print(f"[{c['fecha_firma'][11:]}] {evento['entidad'][:40]:40s} ${evento['valor']:>16,.0f}")

producer.flush()
print("Envío terminado.")