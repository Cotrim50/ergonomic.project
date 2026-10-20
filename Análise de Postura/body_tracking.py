########################################################################
#
# Algorithm developed by José Cotrim, October 2025
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import viewer.ogl_viewer.viewer as gl
import viewer.cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
import time

from utils.pdf_report import gerar_pdf_do_json
from utils.json_report import RelatorioJSON
from detectors.sitting_detection import detectar_sentado
from detectors.posture_check import (
    verificar_postura_tronco,
    calcular_flexao_lateral_tronco,
    calcular_rotacao_axial_tronco, #not used so far, still need improvements
    avaliar_posicao_bracos,
    verificar_inclinacao_vertical_cabeca)


def parse_args(init, opt):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")



def main(opt):
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    parse_args(init_params, opt)

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
    body_param.enable_body_fitting = True            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE 
    body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use -> sl.BODY_FORMAT.BODY_18 or sl.BODY_FORMAT.BODY_34 or sl.BODY_FORMAT.BODY_38

    # Enable Object Detection module
    zed_error = zed.enable_body_tracking(body_param)
    if zed_error != sl.ERROR_CODE.SUCCESS:
        print("enable_body_tracking", zed_error, "\nExit program.")
        zed.close()
        exit(-1)


    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()
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
    
    last_print_time = 0
    PRINT_INTERVAL = 2.0
    sitting_states = {}
    bad_posture_states = {}

    relatorio = RelatorioJSON()
    start_time = time.time()
    frame_counter = 0

    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (podes usar outros como 'MJPG' ou 'MP4V')
    fps = 5  # Define a taxa de frames, ajusta conforme o teu caso
    width = display_resolution.width  # largura da imagem
    height = display_resolution.height  # altura da imagem
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

    #Main Loop
    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.retrieve_bodies(bodies, body_runtime_param)

            current_time = time.time()

            #if current_time - last_print_time >= PRINT_INTERVAL: last_print_time += PRINT_INTERVAL if bodies.body_list:
            if bodies.body_list and (current_time - last_print_time >= PRINT_INTERVAL):
                last_print_time = current_time

                for i, body in enumerate(bodies.body_list):
                    if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                        print(f"\n--- Corpo {i} (ID: {body.id}) ---")
                        sentado = detectar_sentado(body)
                        estado = "sentado" if sentado else "de_pe"
                        sitting_states[body.id] = sentado

                        # Regista no relatório
                        relatorio.adicionar_estado(body.id, int(time.time() - start_time), estado)
                        
                        # Avaliar postura do tronco (boolean + ângulo)
                        ma_postura_tronco, angulo_tronco = verificar_postura_tronco(body)

                        # Avaliar postura dos braços (boolean + alturas)
                        ma_postura_bracos, ombro_dir, cotovelo_dir, ombro_esq, cotovelo_esq = avaliar_posicao_bracos(body)

                        # Guarda os booleanos para má postura geral (por exemplo se algum dos dois estiver mau)
                        ma_postura = ma_postura_tronco or ma_postura_bracos
                        bad_posture_states[body.id] = ma_postura

                        relatorio.adicionar_angulo_tronco(body.id, int(current_time - start_time), angulo_tronco)
                        relatorio.adicionar_posicao_bracos(body.id, int(current_time - start_time), ombro_dir, cotovelo_dir, ombro_esq, cotovelo_esq)

                        #else: # Sem deteções — ainda assim guarda no relatório relatorio.adicionar_estado("sem_id", int(current_time - start_time), "sem_deteccao")

            # Update GL view
            viewer.update_view(image, bodies) 
            # Update OCV view
            image_left_ocv = image.get_data()

            # Se tiver canal alfa, converte para BGR
            if image_left_ocv.shape[2] == 4:
                image_left_ocv = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGRA2BGR)

            
            # # Se a imagem tiver canal alfa, remove
            # if image_left_ocv.shape[2] == 4:
            #     image_left_ocv = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGRA2BGR)
            # pixel_size = 15  # ajuste do tamanho do pixel para pixelizar

            # for body in bodies.body_list:
            #     if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
            #         # Verifica se o objeto tem a máscara
            #         if hasattr(body, "mask") and body.mask is not None:
            #             mask_mat = body.mask  # objeto pyzed.sl.Mat
            #             mask_np = mask_mat.get_data()  # converte para numpy array

            #             # Se a máscara não tem mesma resolução da imagem, redimensiona
            #             if mask_np.shape[:2] != image_left_ocv.shape[:2]:
            #                 mask_np = cv2.resize(mask_np, (image_left_ocv.shape[1], image_left_ocv.shape[0]), interpolation=cv2.INTER_NEAREST)

            #             # Cria versão pixelizada do frame
            #             height, width = image_left_ocv.shape[:2]
            #             temp = cv2.resize(image_left_ocv, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
            #             pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

            #             # Converte máscara para 3 canais
            #             mask_3ch = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)

            #             # Aplica pixelização só na região da máscara
            #             image_left_ocv = np.where(mask_3ch == 255, pixelated, image_left_ocv)

            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format, sitting_states, bad_posture_states)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            out.write(image_left_ocv)
            #frame_counter += 1
            

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
    out.release()
    duracao_total = int(time.time() - start_time)
    #duracao_total = int(frame_counter / fps)
    relatorio.set_duracao(duracao_total)
    relatorio.calcular_estatisticas()
    json_path = relatorio.guardar_json()
    print(f"✅ Relatório JSON guardado em: {json_path}")
    pdf_path = gerar_pdf_do_json(json_path)
    print(f"📄 PDF gerado em: {pdf_path}")
    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
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
    main(opt) 