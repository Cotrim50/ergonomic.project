# @jcotrim
"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window with anonymzed human faces.
"""
''''''
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
import math

#hands
import mediapipe as mp
import time



def parse_args(init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print(f"[Sample] Using SVO File input: {opt.input_svo_file}")
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ":" in ip_str and ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            host, port = ip_str.split(':')
            init.set_from_stream(host, int(port))
        elif ip_str.replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
        else:
            print("Invalid IP format. Using live stream")
        print(f"[Sample] Using Stream input, IP: {ip_str}")
    
    # Configurar resolução
    resolutions = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA
    }
    
    for res_name, res_value in resolutions.items():
        if res_name in opt.resolution:
            init.camera_resolution = res_value
            print(f"[Sample] Using Camera in resolution {res_name}")
            return
    
    print("[Sample] Using default resolution")


# Detector de mãos do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=6,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,  # 0=Lite, 1=Full
)



def main():
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = False            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    print("\n=== Informações da Câmera ===")
    print(f"Resolução: {camera_info.camera_configuration.resolution.width} x {camera_info.camera_configuration.resolution.height}")
    print(f"FPS: {camera_info.camera_configuration.fps}")

    calibration_params = camera_info.camera_configuration.calibration_parameters
    fov_h = calibration_params.left_cam.h_fov  # FOV horizontal em graus
    fov_v = calibration_params.left_cam.v_fov  # FOV vertical em graus
    print(f"Campo de Visão (FOV): Horizontal = {fov_h:.2f}°, Vertical = {fov_v:.2f}°")

    # Calcular a largura da cena a 1 metro de distância
    distance_m = 1.0  # 1 metro de distância
    scene_width_at_1m = 2 * math.tan(math.radians(fov_h/2)) * distance_m
    print(f"Largura da cena a 1 metro: {scene_width_at_1m:.2f} metros ({scene_width_at_1m*100:.2f} cm)")
    print("=============================\n")

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10 

    # parametros analise fadiga
    # Thresholds
    PITCH_LIMIT = 20  # graus
    ROLL_LIMIT = 10   # graus
    WINDOW_DURATION = 2  # segundos para média

    pitch_history = []  
    roll_history = []   

    left_arm_history = []  
    right_arm_history = []

    last_print_time = 0
    PRINT_INTERVAL = 2   # segundos

    while viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            start_time_body_detection = time.time()
            zed.retrieve_bodies(bodies, body_runtime_param)
            current_time = time.time()
            if bodies.body_list: 
                for i, body in enumerate(bodies.body_list):
                    if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                        

                        
                        if current_time - last_print_time >= PRINT_INTERVAL:
                            last_print_time = current_time  # atualiza timestamp

                            # Iterar apenas pelos keypoints 0, 1 e 2
                            if hasattr(body, "keypoint") and hasattr(body, "local_orientation_per_joint"):
                                for idx, (kp, quat) in enumerate(zip(body.keypoint[:3], body.local_orientation_per_joint[:3])):
                                    #print(f"Joint {idx}: position={kp}, local_orientation={quat}")
                                    if hasattr(body, "keypoint_confidence"):
                                        conf = body.keypoint_confidence[idx]
                                        #print(f"Joint {idx} confidence: {conf}")

                        # Má postura - Inclinação Tronco
                        if hasattr(body, "local_orientation_per_joint") and len(body.local_orientation_per_joint) > 1:
                            naval_spine_quat = body.local_orientation_per_joint[2]  # joint 1 = Naval_Spine

                            x, y, z, w = naval_spine_quat

                            # Converter para Euler
                            roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
                            pitch = np.arcsin(2*(w*y - z*x))
                            roll_deg = np.degrees(roll)
                            pitch_deg = np.degrees(pitch)

                            pitch_history.append((current_time, pitch_deg))
                            roll_history.append((current_time, roll_deg))

                            pitch_history = [(t, val) for t, val in pitch_history if current_time - t <= WINDOW_DURATION]
                            roll_history = [(t, val) for t, val in roll_history if current_time - t <= WINDOW_DURATION]

                            pitch_avg = np.mean([val for t, val in pitch_history])
                            roll_avg = np.mean([val for t, val in roll_history])

                            # Print valores atuais e média
                            print(f"Roll: {roll_deg:.2f}° (média 2s: {roll_avg:.2f}°), "
                                f"Pitch: {pitch_deg:.2f}° (média 2s: {pitch_avg:.2f}°)")


                            # Verificar média com thresholds
                            if abs(pitch_avg) > PITCH_LIMIT:
                                print(f"Má postura detectada no pitch! Média últimos 2s: {pitch_avg:.2f}°")
                            if abs(roll_avg) > ROLL_LIMIT:
                                print(f"Má postura detectada no roll! Média últimos 2s: {roll_avg:.2f}°")

                        else:
                            print("Local orientation do Naval_Spine não disponível ainda")   

                        if hasattr(body, "local_orientation_per_joint") and len(body.local_orientation_per_joint) > 14:
                            right_wrist_quat = body.local_orientation_per_joint[14]
                            left_wrist_quat = body.local_orientation_per_joint[7]
                            right_elbow_quat = body.local_orientation_per_joint[13]
                            left_elbow_quat = body.local_orientation_per_joint[6]

                            def quat_to_pitch(quat):
                                x,y,z,w = quat
                                pitch = np.arcsin(2*(w*y - z*x))
                                return np.degrees(pitch)

                            left_upper_pitch = quat_to_pitch(left_elbow_quat)
                            left_lower_pitch = quat_to_pitch(left_wrist_quat)                            
                            right_upper_pitch = quat_to_pitch(right_elbow_quat)
                            right_lower_pitch = quat_to_pitch(right_wrist_quat)
                            print(f"BRAÇOS {left_upper_pitch},{right_upper_pitch}, {left_lower_pitch} , {right_lower_pitch}")
                            current_time = time.time()

                            left_above = 1 if left_upper_pitch > 0 and left_lower_pitch > 0 else 0
                            right_above = 1 if right_upper_pitch > 0 and right_lower_pitch > 0 else 0


                            left_arm_history.append((current_time, left_above))
                            right_arm_history.append((current_time, right_above))

                            WINDOW_DURATION = 2.0
                            left_arm_history = [(t, val) for t, val in left_arm_history if current_time - t <= WINDOW_DURATION]
                            right_arm_history = [(t, val) for t, val in right_arm_history if current_time - t <= WINDOW_DURATION]


                            left_avg = np.mean([val for t, val in left_arm_history])
                            right_avg = np.mean([val for t, val in right_arm_history])

                            if left_avg > 0.5:
                                print(f"Má postura braço esquerdo! Média últimos 2s: {left_avg:.2f}")
                            if right_avg > 0.5:
                                print(f"Má postura braço direito! Média últimos 2s: {right_avg:.2f}")





            body_detection_time = time.time() - start_time_body_detection
            # Armazenar e calcular estatísticas
            if not hasattr(main, "body_detection_times"):
                main.body_detection_times = []
                main.body_frame_count = 0

            main.body_detection_times.append(body_detection_time)
            main.body_frame_count += 1

            if main.body_frame_count % 30 == 0:
                avg_body_detection = sum(main.body_detection_times) / len(main.body_detection_times)
                print("\n--- Estatísticas de Tempo para Detecção de Corpo (ms) ---")
                print(f"Tempo de detecção do corpo: {avg_body_detection*1000:.2f} ms")
                #print(f"FPS estimado para detecção de corpo: {1/avg_body_detection:.2f}")
                main.body_detection_times = []

            # Update GL view
            viewer.update_view(image, bodies) 
            # Update OCV view
            image_left_ocv = image.get_data()
            
            # Converter para BGR se tiver 4 canais
            if len(image_left_ocv.shape) == 3 and image_left_ocv.shape[2] == 4:
                image_left_ocv = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGRA2BGR)
                
            cv_viewer.hands = hands
            cv_viewer.mp_hands = mp_hands
            cv_viewer.mp_drawing = mp_drawing
            cv_viewer.mp_drawing_styles = mp_drawing_styles
            cv_viewer.render_2D(image_left_ocv, image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            key = cv2.waitKey(key_wait)
            if key == 113: # for 'q' key
                print("Exiting...")
                break
            if key == 109: # for 'm' key
                if (key_wait>0):
                    print("Pause")
                    key_wait = 0 
                else : 
                    print("Restart")
                    key_wait = 10 
    cv_viewer.cleanup()
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    hands.close()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 

