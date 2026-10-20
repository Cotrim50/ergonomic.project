import numpy as np

from config import LIMIAR_INCLINACAO_TRONCO, LIMIAR_ROTACAO_TRONCO, LIMIAR_FLEXAO_TRONCO, LIMIAR_INCLINACAO_CABECA

def verificar_postura_tronco(body):
    """
    Verifica inclinação do tronco no plano vertical (ângulo entre tronco e vetor vertical).
    Retorna tuple: (ma_postura: bool, angulo: float)
    """
    keypoints = body.keypoint
    neck = keypoints[1]
    hip_right = keypoints[8]
    hip_left = keypoints[11]

    hip_center = (hip_right + hip_left) / 2
    trunk_vec = neck - hip_center
    vertical_vec = np.array([0, 1, 0])

    cos_theta = np.dot(trunk_vec, vertical_vec) / (np.linalg.norm(trunk_vec) * np.linalg.norm(vertical_vec))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    direcao = "frente" if trunk_vec[1] < 0 else "trás"

    print(f"Inclinação do tronco: {angle_deg:.2f}° para {direcao.upper()}")

    ma_postura = angle_deg > LIMIAR_INCLINACAO_TRONCO
    return ma_postura, angle_deg

def calcular_rotacao_axial_tronco(body):
    """
    Calcula a rotação axial do tronco com base na orientação dos ombros.
    Retorna True se a rotação exceder o limiar (em graus).
    """
    keypoints = body.keypoint
    shoulder_right = keypoints[2]
    shoulder_left = keypoints[5]

    shoulder_vec = shoulder_right - shoulder_left
    shoulder_vec_proj = np.array([shoulder_vec[0], 0, shoulder_vec[2]])
    horizontal_vec = np.array([1, 0, 0])

    cos_theta = np.dot(shoulder_vec_proj, horizontal_vec) / (np.linalg.norm(shoulder_vec_proj) * np.linalg.norm(horizontal_vec))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    direcao = "DIREITA" if shoulder_vec_proj[2] > 0 else "ESQUERDA"

    print(f"Rotação axial do tronco: {angle_deg:.2f}° para {direcao}")

    return angle_deg > LIMIAR_ROTACAO_TRONCO

def calcular_flexao_lateral_tronco(body):
    """
    Calcula a flexão lateral do tronco (desvio lateral).
    Retorna True se a flexão exceder o limiar.
    """
    keypoints = body.keypoint
    neck = keypoints[1]
    hip_right = keypoints[8]
    hip_left = keypoints[11]

    hip_center = (hip_right + hip_left) / 2
    trunk_vec = neck - hip_center
    vertical_vec = np.array([0, 1, 0])

    trunk_vec_proj = np.array([0, trunk_vec[1], trunk_vec[2]])

    cos_theta = np.dot(trunk_vec_proj, vertical_vec) / (np.linalg.norm(trunk_vec_proj) * np.linalg.norm(vertical_vec))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    direcao = "DIREITA" if trunk_vec_proj[2] > 0 else "ESQUERDA"

    print(f"Flexão lateral do tronco: {angle_deg:.2f}° para {direcao}")

    return angle_deg > LIMIAR_FLEXAO_TRONCO

def verificar_inclinacao_vertical_cabeca(body):
    """
    Verifica inclinação vertical da cabeça baseado na orientação dos olhos e orelhas.
    Retorna True se a inclinação exceder o limiar.
    """
    keypoints = body.keypoint
    right_eye = keypoints[14]
    left_eye = keypoints[15]
    right_ear = keypoints[16]
    left_ear = keypoints[17]

    mid_eyes = (right_eye + left_eye) / 2
    mid_ears = (right_ear + left_ear) / 2

    ear_to_eye_vec = mid_eyes - mid_ears
    horizontal_vec = np.array([1, 0, 0])

    cos_theta = np.dot(ear_to_eye_vec, horizontal_vec) / (np.linalg.norm(ear_to_eye_vec) * np.linalg.norm(horizontal_vec))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))

    direcao = "cima" if ear_to_eye_vec[1] > 0 else "baixo"

    print(f"Inclinação vertical da cabeça: {angle_deg:.2f}° para {direcao.upper()}")

    return angle_deg > LIMIAR_INCLINACAO_CABECA

def avaliar_posicao_bracos(body):
    """
    Verifica se os cotovelos estão acima dos ombros, indicando má postura dos braços.
    Retorna tuple: (ma_postura: bool, ombro_dir: float, cotovelo_dir: float, ombro_esq: float, cotovelo_esq: float)
    """
    keypoints = body.keypoint

    shoulder_right = keypoints[2]
    elbow_right = keypoints[3]

    shoulder_left = keypoints[5]
    elbow_left = keypoints[6]

    mensagens = []

    if elbow_right[1] > shoulder_right[1]:
        mensagens.append("Braço DIREITO está numa má posição (cotovelo acima do ombro).")

    if elbow_left[1] > shoulder_left[1]:
        mensagens.append("Braço ESQUERDO está numa má posição (cotovelo acima do ombro).")

    for msg in mensagens:
        print(f"{msg}")

    ma_postura = len(mensagens) > 0
    return ma_postura, shoulder_right[1], elbow_right[1], shoulder_left[1], elbow_left[1]