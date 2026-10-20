# @jcotrim

import cv2
import numpy as np
from cv_viewer.utils import *
import pyzed.sl as sl
import time

import mediapipe as mp

from scipy.spatial.transform import Rotation as R


print("MediaPipe Hands inicializado com sucesso")
hands = None
mp_hands = None
mp_drawing = None
mp_drawing_styles = None


#converte as coordenadas com um fator de escala
def cvt(pt, scale):
    '''
    Function that scales point coordinates
    '''
    out = [pt[0]*scale[0], pt[1]*scale[1]]
    return out

#desenha o esqueleto em cima da imagem
def render_sk(left_display, img_scale, obj, color, BODY_BONES):
    for part in BODY_BONES:
        kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
        kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)
        # Check that the keypoints are inside the image
        if(kp_a[0] < left_display.shape[1] and kp_a[1] < left_display.shape[0] 
        and kp_b[0] < left_display.shape[1] and kp_b[1] < left_display.shape[0]
        and kp_a[0] > 0 and kp_a[1] > 0 and kp_b[0] > 0 and kp_b[1] > 0 ):
            cv2.line(left_display, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])), color, 1, cv2.LINE_AA)

    # Skeleton joints
    for kp in obj.keypoint_2d:
        cv_kp = cvt(kp, img_scale)
        if(cv_kp[0] < left_display.shape[1] and cv_kp[1] < left_display.shape[0]):
            cv2.circle(left_display, (int(cv_kp[0]), int(cv_kp[1])), 3, color, -1)

def detect_and_draw_hands(image):
    """
    Detecta e desenha as mãos na imagem usando MediaPipe
    """
    # Iniciar cronômetro total
    start_time_total = time.time()
    
    # Verificar se a imagem tem 4 canais (BGRA) e converter para 3 canais (BGR)
    start_time_conversion = time.time()
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    conversion_time = time.time() - start_time_conversion
    
    # Converter para RGB
    start_time_to_rgb = time.time()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    to_rgb_time = time.time() - start_time_to_rgb
    
    try:
        # Processar a imagem com MediaPipe
        start_time_detection = time.time()
        results = hands.process(image_rgb)
        detection_time = time.time() - start_time_detection
        
        # Desenhar os landmarks das mãos
        start_time_drawing = time.time()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar os keypoints/landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                if results.multi_handedness:
                    for idx, hand_handedness in enumerate(results.multi_handedness):
                        if idx < len(results.multi_hand_landmarks):
                            if hand_landmarks == results.multi_hand_landmarks[idx]:
                                handedness = hand_handedness.classification[0].label
                                # Inverter o rótulo para corresponder à perspectiva da câmera
                                if handedness == "Left":
                                    handedness = "Right"
                                else:
                                    handedness = "Left"
                                # Obter a posição do pulso para colocar o texto
                                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                                h, w, _ = image.shape
                                cx, cy = int(wrist.x * w), int(wrist.y * h)
                                cv2.putText(image, f"{handedness}", (cx-40, cy-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        drawing_time = time.time() - start_time_drawing
    except Exception as e:
        print(f"Erro ao processar mãos: {e}")
    
    # Calcular tempo total
    total_time = time.time() - start_time_total

    # Inicializar variáveis de estatísticas na primeira chamada
    if not hasattr(detect_and_draw_hands, "frame_count"):
        detect_and_draw_hands.frame_count = 0
        detect_and_draw_hands.times = {
            'total': [], 'conversion': [], 'to_rgb': [], 
            'detection': [], 'drawing': []
        }

    # Incrementar contador e armazenar tempos
    detect_and_draw_hands.frame_count += 1
    times = detect_and_draw_hands.times
    times['total'].append(total_time)
    times['conversion'].append(conversion_time)
    times['to_rgb'].append(to_rgb_time)
    times['detection'].append(detection_time)
    times['drawing'].append(drawing_time)

    # Imprimir estatísticas a cada 30 frames
    if detect_and_draw_hands.frame_count % 30 == 0:
        # Calcular médias
        avgs = {k: sum(v)/len(v) for k, v in times.items()}
        
        # Imprimir resultados
        print("\n--- Estatísticas de Tempo para Detecção de Mãos (ms) ---")
        print(f"Tempo total: {avgs['total']*1000:.2f} ms")
        detect_and_draw_hands.times = {k: [] for k in times}

    
    return image

def calculate_and_apply_mask(left_display, center_x, center_y, depth_z, mask_size, obj_id, current_time):

    # Ajustar o tamanho da máscara com base na profundidade, se disponível
    if depth_z is not None and depth_z > 0:
        # Ajustar o tamanho da máscara inversamente proporcional à distância
        reference_distance = 1.0  # 1 metro como referência
        reference_size = 120      # Tamanho de referência a 1 metro
        mask_size = int(reference_size * (reference_distance / depth_z))
        mask_size = max(10, min(200, mask_size))  # Limitar entre 10 e 200 pixels
    
    # Definir a região da máscara
    head_top = max(0, center_y - mask_size)
    head_bottom = min(left_display.shape[0], center_y + mask_size)
    head_left = max(0, center_x - mask_size)
    head_right = min(left_display.shape[1], center_x + mask_size)
    
    # Armazenar a posição da máscara e atualizar o timestamp
    render_2D.last_mask_positions[obj_id] = {
        'top': head_top,
        'bottom': head_bottom,
        'left': head_left,
        'right': head_right,
        'size': mask_size,
        'center_x': center_x,
        'center_y': center_y,
        'center_z': depth_z
    }
    render_2D.mask_timestamps[obj_id] = current_time
    
    # Extrair a região da cabeça
    head_region = left_display[head_top:head_bottom, head_left:head_right]
    
    # Aplicar máscara preta
    if head_region.size > 0:
        # Criar uma máscara preta do mesmo tamanho da região da cabeça
        gray_mask = np.zeros(head_region.shape, dtype=np.uint8)
        
        # Se a imagem for colorida, garantir que a máscara também tenha 3 canais
        if len(gray_mask.shape) == 2:  # Se for grayscale (1 canal)
            gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        elif len(gray_mask.shape) != 3:  # Se não for nem grayscale nem BGR
            gray_mask = np.zeros((head_region.shape[0], head_region.shape[1], 3), dtype=np.uint8)
        
        # Aplicar a máscara preta na região da cabeça
        left_display[head_top:head_bottom, head_left:head_right] = gray_mask
        return True
    
    return False

def apply_last_known_mask(left_display, obj_id, current_time, persistence_time):
    """
    Aplica a última máscara conhecida se estiver dentro do período de persistência.
    
    Args:
        left_display: Imagem onde a máscara será aplicada
        obj_id: ID do objeto/pessoa
        current_time: Tempo atual
        persistence_time: Tempo de persistência da máscara
        
    Returns:
        bool: True se a máscara foi aplicada com sucesso, False caso contrário
    """
    # Verificar se temos uma posição recente e se ainda estamos dentro do período de persistência
    if obj_id in render_2D.last_mask_positions and \
       current_time - render_2D.mask_timestamps.get(obj_id, 0) < persistence_time:
        
        # Recuperar a última posição conhecida da máscara
        mask_info = render_2D.last_mask_positions[obj_id]
        head_top = mask_info['top']
        head_bottom = mask_info['bottom']
        head_left = mask_info['left']
        head_right = mask_info['right']
        
        head_region = left_display[head_top:head_bottom, head_left:head_right]
        

        if head_region.size > 0:
            # Criar uma máscara preta do mesmo tamanho da região da cabeça
            gray_mask = np.zeros(head_region.shape, dtype=np.uint8)
            
            # Se a imagem for colorida, garantir que a máscara também tenha 3 canais
            if len(gray_mask.shape) == 2:  # Se for grayscale (1 canal)
                gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
            elif len(gray_mask.shape) != 3:  # Se não for nem grayscale nem BGR
                gray_mask = np.zeros((head_region.shape[0], head_region.shape[1], 3), dtype=np.uint8)
            
            left_display[head_top:head_bottom, head_left:head_right] = gray_mask
            return True
    
    return False

def render_2D(left_display, img_scale, objects, is_tracking_on, body_format):
    '''
    Parameters
        left_display (np.array): numpy array containing image data
        img_scale (list[float])
        objects (list[sl.ObjectData])
    '''

    if not hasattr(render_2D, 'last_mask_positions'):
        render_2D.last_mask_positions = {}  
        render_2D.mask_timestamps = {}      
        render_2D.mask_persistence = 0.5    # Tempo em segundos para manter a máscara (ajuste conforme necessário)
    
    # Lista para armazenar IDs de objetos ativos neste frame
    active_ids = []

    # Faz uma cópia da imagem original
    overlay = left_display.copy()
    original_image = left_display.copy()  # Nova cópia para detecção de mãos
    last_print_time = 0
    
    # Marcar o centro da câmera com um ponto vermelho
    camera_center_x = left_display.shape[1] // 2
    camera_center_y = left_display.shape[0] // 2
    cv2.circle(left_display, (camera_center_x, camera_center_y), 6, (0, 0, 255), -1)  # Vermelho, tamanho 6
    
    # Detectar mãos na imagem original (antes de qualquer modificação)
    hand_results = None
    if 'hands' in globals():
        if original_image is not None:
            try:
                # Converter para RGB (MediaPipe requer RGB)
                image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                # Processar a imagem com MediaPipe (apenas detecção, sem desenho)
                hand_results = hands.process(image_rgb)
            except Exception as e:
                print(f"Erro ao processar mãos: {e}")
    
    
    # Render skeleton joints and bones
    for obj in objects:
        if render_object(obj, is_tracking_on):
            if len(obj.keypoint_2d) > 0:
                obj_id = obj.id
                active_ids.append(obj_id)
                color = generate_color_id_u(obj.id)
                
                head_keypoints_indices = []
                if body_format == sl.BODY_FORMAT.BODY_18:
                    head_keypoints_indices = [0, 14, 15, 16, 17]  
                elif body_format == sl.BODY_FORMAT.BODY_34 or body_format == sl.BODY_FORMAT.BODY_38:
                    head_keypoints_indices = [0, 1, 2, 3, 4, 5]  
               

                valid_head_points_2d = []
                valid_head_points_3d = []  
                detected_keypoints = {
                    "Nariz": False,
                    "Olho Esquerdo": False,
                    "Olho Direito": False,
                    "Orelha Esquerda": False,
                    "Orelha Direita": False
                }


                for idx in head_keypoints_indices:
                    if idx < len(obj.keypoint_2d):
                        point_2d = obj.keypoint_2d[idx]
                        if (not np.isnan(point_2d[0]) and not np.isnan(point_2d[1]) and
                            0 <= point_2d[0] < left_display.shape[1] / img_scale[0] and
                            0 <= point_2d[1] < left_display.shape[0] / img_scale[1]):
                           
                            # Converter para a escala da imagem
                            scaled_point = cvt([point_2d[0], point_2d[1]], img_scale)
                           
                            if (0 <= scaled_point[0] < left_display.shape[1] and
                                0 <= scaled_point[1] < left_display.shape[0]):
                               
                                valid_head_points_2d.append(scaled_point)
                               
                                # Obter o ponto 3D correspondente se disponível
                                if hasattr(obj, 'keypoint') and len(obj.keypoint) > idx:
                                    point_3d = obj.keypoint[idx]
                                    if not np.isnan(point_3d[0]) and not np.isnan(point_3d[1]) and not np.isnan(point_3d[2]):
                                        valid_head_points_3d.append(point_3d)
                               
                                # Marcar qual keypoint foi detectado
                                if body_format == sl.BODY_FORMAT.BODY_18:
                                    if idx == 0: detected_keypoints["Nariz"] = True
                                    elif idx == 14: detected_keypoints["Olho Esquerdo"] = True
                                    elif idx == 15: detected_keypoints["Olho Direito"] = True
                                    elif idx == 16: detected_keypoints["Orelha Esquerda"] = True
                                    elif idx == 17: detected_keypoints["Orelha Direita"] = True
                                elif body_format == sl.BODY_FORMAT.BODY_34 or body_format == sl.BODY_FORMAT.BODY_38:
                                    if idx == 0: detected_keypoints["Nariz"] = True
                                    elif idx == 1: detected_keypoints["Olho Esquerdo"] = True
                                    elif idx == 2: detected_keypoints["Olho Direito"] = True
                                    elif idx == 3: detected_keypoints["Orelha Esquerda"] = True
                                    elif idx == 4: detected_keypoints["Orelha Direita"] = True


                mask_applied = False
                current_time = time.time()
                mask_applied = False

                # Se temos pelo menos 2 pontos válidos, criar uma região que englobe todos eles
                if len(valid_head_points_2d) >= 2:
                    # Converter para numpy array para facilitar operações
                    valid_head_points_2d = np.array(valid_head_points_2d)
                    
                    # Calcular o centroide 2D (média de todos os pontos válidos)
                    center_x = int(np.mean(valid_head_points_2d[:, 0]))
                    center_y = int(np.mean(valid_head_points_2d[:, 1]))
                    
                    # Calcular o centroide 3D se tivermos pontos 3D válidos
                    center_z = None
                    if len(valid_head_points_3d) >= 2:
                        valid_head_points_3d = np.array(valid_head_points_3d)
                        center_z = np.mean(valid_head_points_3d[:, 2])
                    
                    mask_applied = calculate_and_apply_mask(
                        left_display, 
                        center_x, 
                        center_y, 
                        abs(center_z) if center_z is not None else None,
                        120,  
                        obj_id, 
                        current_time
                    )

                #Se deteta o pescoço e não a cara, usar o pescoço para a máscara
                elif body_format == sl.BODY_FORMAT.BODY_18 and not mask_applied: 
                    neck_idx = 1  
                    if neck_idx < len(obj.keypoint_2d):
                        neck_point_2d = obj.keypoint_2d[neck_idx]
                        
                        if (not np.isnan(neck_point_2d[0]) and not np.isnan(neck_point_2d[1])):
                            print("Pescoço detectado!")
                            # Converter para a escala da imagem
                            neck_scaled = cvt([neck_point_2d[0], neck_point_2d[1]], img_scale)
                            
                            # Verificar se o ponto convertido está dentro da imagem
                            if (0 <= neck_scaled[0] < left_display.shape[1] and
                                0 <= neck_scaled[1] < left_display.shape[0]):
                                print(f"Pescoço dentro da imagem: ({neck_scaled[0]}, {neck_scaled[1]})")
                                
                                neck_z = None
                                if hasattr(obj, 'keypoint') and len(obj.keypoint) > neck_idx:
                                    neck_point_3d = obj.keypoint[neck_idx]
                                    if not np.isnan(neck_point_3d[2]):
                                        neck_z = abs(neck_point_3d[2])
                                        print(f"Profundidade do pescoço: {neck_z} metros")
                                

                                center_x = int(neck_scaled[0])
                                center_y = int(neck_scaled[1] - 50)  
                                print(f"Centro da máscara: ({center_x}, {center_y})")
                                
                                
                                mask_applied = calculate_and_apply_mask(
                                    left_display, 
                                    center_x, 
                                    center_y, 
                                    neck_z,
                                    120,  
                                    obj_id, 
                                    current_time
                                )

                # Se não conseguimos aplicar uma máscara neste frame, mas temos uma posição recente
                if not mask_applied:
                    mask_applied = apply_last_known_mask(
                        left_display, 
                        obj_id, 
                        current_time, 
                        render_2D.mask_persistence
                    )

                
                #Algoritmo para desenhar centroide do esqueleto
                if body_format == sl.BODY_FORMAT.BODY_18:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_18_BONES)
                    
                    neck_idx = 1
                    right_hip_idx = 8
                    left_hip_idx = 11
                    
                    if (neck_idx < len(obj.keypoint_2d) and
                        right_hip_idx < len(obj.keypoint_2d) and
                        left_hip_idx < len(obj.keypoint_2d)):
                        
                        neck = obj.keypoint_2d[neck_idx]
                        right_hip = obj.keypoint_2d[right_hip_idx]
                        left_hip = obj.keypoint_2d[left_hip_idx]
                        
                        if (not np.isnan(neck[0]) and not np.isnan(neck[1]) and
                            not np.isnan(right_hip[0]) and not np.isnan(right_hip[1]) and
                            not np.isnan(left_hip[0]) and not np.isnan(left_hip[1])):

                            pelvis_x = (right_hip[0] + left_hip[0]) / 2
                            pelvis_y = (right_hip[1] + left_hip[1]) / 2
                            
                            # Calcular o centro do corpo
                            proportion = 0.6
                            body_center_x = neck[0] + proportion * (pelvis_x - neck[0])
                            body_center_y = neck[1] + proportion * (pelvis_y - neck[1])
                            
                            body_center_point = cvt([body_center_x, body_center_y], img_scale)
                            
                            # Desenhar um círculo para o centro do corpo
                            if(body_center_point[0] < left_display.shape[1] and body_center_point[1] < left_display.shape[0]):
                                body_center_x_px = int(body_center_point[0])
                                body_center_y_px = int(body_center_point[1])
                                
                                cv2.circle(left_display, (body_center_x_px, body_center_y_px),
                                          6, (0, 255, 255), -1)  
                                text_position = (int(body_center_point[0] + 10), int(body_center_point[1]))
                                cv2.putText(left_display, f"ID: {obj_id}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, (0, 255, 255), 1, cv2.LINE_AA) 
                                
                                
                                if hasattr(obj, 'keypoint') and len(obj.keypoint) > 0:
                                    # Obter as coordenadas 3D dos pontos-chave
                                    neck_3d = obj.keypoint[neck_idx]
                                    right_hip_3d = obj.keypoint[right_hip_idx]
                                    left_hip_3d = obj.keypoint[left_hip_idx]
                                    
                                    
                                    if (not np.isnan(neck_3d[0]) and not np.isnan(right_hip_3d[0]) and not np.isnan(left_hip_3d[0])):
                                        # Calcular o centro do corpo em 3D
                                        pelvis_3d_x = (right_hip_3d[0] + left_hip_3d[0]) / 2
                                        pelvis_3d_y = (right_hip_3d[1] + left_hip_3d[1]) / 2
                                        pelvis_3d_z = (right_hip_3d[2] + left_hip_3d[2]) / 2  
                                        
                                        body_center_3d_x = neck_3d[0] + proportion * (pelvis_3d_x - neck_3d[0])
                                        body_center_3d_y = neck_3d[1] + proportion * (pelvis_3d_y - neck_3d[1])
                                        body_center_3d_z = neck_3d[2] + proportion * (pelvis_3d_z - neck_3d[2])  
                                        
                                       
                                        displacement_x_cm = body_center_3d_x * 100
                                        displacement_y_cm = body_center_3d_y * 100
                                        displacement_z_cm = body_center_3d_z * 100
                                        
                                        # Calcular distância 3D completa
                                        distance_3d_cm = np.sqrt(displacement_x_cm**2 + displacement_y_cm**2 + displacement_z_cm**2)
                                        
                                        # Desenhar linha entre centro da câmera e centro do corpo
                                        cv2.line(left_display, (camera_center_x, camera_center_y),
                                                (body_center_x_px, body_center_y_px),
                                                (0, 165, 255), 2)  
                                        
    
                                        current_time = time.time()
                                        last_print_time = getattr(render_2D, 'last_print_time', 0)
                                        if current_time - last_print_time >= 2.0:
                                            print(f"Deslocamento 3D: X={displacement_x_cm:.2f} cm, Y={displacement_y_cm:.2f} cm, Z={displacement_z_cm:.2f} cm")
                                            print(f"Distância 3D entre corpo e centro da câmara: {distance_3d_cm:.2f} cm")
                                            print(f"------------------------------------------------------")
                                            render_2D.last_print_time = current_time
                
                elif body_format == sl.BODY_FORMAT.BODY_34:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_34_BONES)
                elif body_format == sl.BODY_FORMAT.BODY_38:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_38_BONES)

    # Limpar máscaras de objetos que não estão mais ativos após um período
    current_time = time.time()
    ids_to_remove = []
    for obj_id in render_2D.mask_timestamps:
        if obj_id not in active_ids and current_time - render_2D.mask_timestamps[obj_id] > render_2D.mask_persistence * 2:
            ids_to_remove.append(obj_id)
    
    for obj_id in ids_to_remove:
        if obj_id in render_2D.last_mask_positions:
            del render_2D.last_mask_positions[obj_id]
        if obj_id in render_2D.mask_timestamps:
            del render_2D.mask_timestamps[obj_id]

    if hand_results and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Desenhar os keypoints/landmarks
            mp_drawing.draw_landmarks(
                left_display,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Adicionar texto indicando "Mão Esquerda" ou "Mão Direita"
            if hand_results.multi_handedness:
                for idx, hand_handedness in enumerate(hand_results.multi_handedness):
                    if idx < len(hand_results.multi_hand_landmarks):
                        if hand_landmarks == hand_results.multi_hand_landmarks[idx]:
                            handedness = hand_handedness.classification[0].label
                            # Inverter o rótulo para corresponder à perspectiva da câmera
                            if handedness == "Left":
                                handedness = "Right"
                            else:
                                handedness = "Left"
                            # Obter a posição do pulso para colocar o texto
                            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                            h, w, _ = left_display.shape
                            cx, cy = int(wrist.x * w), int(wrist.y * h)
                            cv2.putText(left_display, f"{handedness}", (cx-40, cy-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Combinar imagens
    if left_display is not None and overlay is not None:
        cv2.addWeighted(left_display, 1, overlay, 0, 0.0, left_display)
    else:
        print("Erro: left_display ou overlay é None.")
    
    return left_display

def cleanup():
    """Liberta os recursos do MediaPipe"""
    if 'hands' in globals():
        hands.close()

