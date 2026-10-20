import numpy as np

def detectar_sentado(body, threshold_altura=0.25, threshold_angulo=120):
    """
    Detecta se a pessoa está sentada baseado na altura relativa dos quadris e joelhos 
    e no ângulo entre tronco e coxa.
    
    Parâmetros:
    - body: objeto do ZED com keypoints.
    - threshold_altura: altura mínima hip-joelho para considerar em pé.
    - threshold_angulo: ângulo mínimo entre tronco e coxa para considerar em pé.

    Retorna:
    - True se sentado, False caso contrário.
    """
    keypoints = body.keypoint

    left_hip = keypoints[11]
    left_knee = keypoints[12]
    right_hip = keypoints[8]
    right_knee = keypoints[9]
    neck = keypoints[1]

    # Centro dos quadris e joelhos
    hip_center = (left_hip + right_hip) / 2
    knee_center = (left_knee + right_knee) / 2

    # Diferença de altura (eixo Y)
    altura_hip_joelho = abs(hip_center[1] - knee_center[1])

    # Vetores para cálculo do ângulo tronco-coxa (lado direito)
    trunk_vec = neck - hip_center
    thigh_vec = knee_center - hip_center

    cos_theta = np.dot(trunk_vec, thigh_vec) / (np.linalg.norm(trunk_vec) * np.linalg.norm(thigh_vec))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angulo = np.degrees(np.arccos(cos_theta))

    if altura_hip_joelho < threshold_altura or angulo < threshold_angulo:
        print(f"Pessoa sentada detectada! Altura hip-joelho: {altura_hip_joelho:.2f}, Ângulo tronco-coxa: {angulo:.1f}°")
        return True
    return False
