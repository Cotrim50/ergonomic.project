import json
import os
from datetime import datetime
import numpy as np


def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return round(obj)
    else:
        return obj

class RelatorioJSON:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        self.data = {
            "session_id": timestamp,
            "duration_sec": 0,
            "sitting_stats": {},  # Por ID
            "timeline": {},        # Por ID: lista de {"timestamp": x, "estado": "sentado"/"de_pe"}
            "angles_tronco": {},
            "posicao_bracos": {}
        }

    def adicionar_estado(self, person_id, timestamp, estado):
        if person_id not in self.data["timeline"]:
            self.data["timeline"][person_id] = []
        self.data["timeline"][person_id].append({
            "timestamp": timestamp,
            "estado": estado
        })
    
    def adicionar_angulo_tronco(self, body_id, timestamp, angulo):
        if body_id not in self.data["angles_tronco"]:
            self.data["angles_tronco"][body_id] = []
        self.data["angles_tronco"][body_id].append({"timestamp": timestamp, "angle": angulo})
    
    def adicionar_posicao_bracos(self, body_id, timestamp, ombro_dir, cotovelo_dir, ombro_esq, cotovelo_esq):
        if body_id not in self.data["posicao_bracos"]:
            self.data["posicao_bracos"][body_id] = []
        self.data["posicao_bracos"][body_id].append({
            "timestamp": timestamp,
            "ombro_dir": ombro_dir,
            "cotovelo_dir": cotovelo_dir,
            "ombro_esq": ombro_esq,
            "cotovelo_esq": cotovelo_esq
        })

    def calcular_estatisticas(self):
        for person_id, timeline in self.data["timeline"].items():
            total = len(timeline)
            if total == 0:
                continue
            sentados = sum(1 for item in timeline if item["estado"] == "sentado")
            percent_sentado = (sentados / total) * 100
            self.data["sitting_stats"][person_id] = {
                "sitting_percent": round(percent_sentado, 1),
                "standing_percent": round(100 - percent_sentado, 1),
                "samples": total
            }

    def set_duracao(self, duracao):
        self.data["duration_sec"] = duracao

    def guardar_json(self, output_dir="relatorios_json"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.session_id}.json")
        clean_data = convert_np_types(self.data)  # converte os np.float32 para float
        with open(path, "w") as f:
            json.dump(clean_data, f, indent=4)
        return path
