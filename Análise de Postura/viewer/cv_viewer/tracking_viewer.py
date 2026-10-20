import cv2
import numpy as np

from viewer.cv_viewer.utils import *
import pyzed.sl as sl

#----------------------------------------------------------------------
#       2D VIEW
#----------------------------------------------------------------------
def cvt(pt, scale):
    '''
    Function that scales point coordinates
    '''
    out = [pt[0]*scale[0], pt[1]*scale[1]]
    return out

def render_sk(left_display, img_scale, obj, color, BODY_BONES, sitting_states=None, bad_posture_states=None):
    # Draw skeleton bones
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

    if hasattr(obj, "bounding_box_2d") and obj.bounding_box_2d is not None:
        # Converte e escala os pontos da bounding box
        bbox_pts = np.array([cvt(pt, img_scale) for pt in obj.bounding_box_2d], dtype=np.int32)
        
        if bad_posture_states and bad_posture_states.get(obj.id, False):
            box_color = (0, 0, 255)  # vermelho
        else:
            box_color = (0, 255, 0)  # verde

        cv2.polylines(left_display, [bbox_pts], isClosed=True, color=box_color, thickness=2)

        # Posição para colocar o texto (canto superior direito da bbox)
        top_right = np.array([
            np.max(np.array(obj.bounding_box_2d)[:, 0]),  # maior X (direita)
            np.min(np.array(obj.bounding_box_2d)[:, 1])   # menor Y (topo)
        ])
        text_pos = (int(top_right[0]*img_scale[0]), int(top_right[1]*img_scale[1]) - 10)
    else:
        # Se não tiver bbox, posiciona no pescoço (keypoint 1)
        neck = obj.keypoint_2d[1]
        text_pos = (int(neck[0]*img_scale[0]), int(neck[1]*img_scale[1]) - 10)

    # Estado da ação (Idle ou Moving)
    action_text = "Unknown"
    if hasattr(obj, "action_state"):
        if obj.action_state == sl.OBJECT_ACTION_STATE.IDLE:
            action_text = "Idle"
        elif obj.action_state == sl.OBJECT_ACTION_STATE.MOVING:
            action_text = "Moving"

    sitting_text = "Sentado" if sitting_states and sitting_states.get(obj.id, False) else "De pe"

    # Desenha textos na imagem
    cv2.putText(left_display, action_text, (text_pos[0], text_pos[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(left_display, sitting_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(left_display, f"ID: {obj.id}", (text_pos[0], text_pos[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)



def render_2D(left_display, img_scale, objects, is_tracking_on, body_format, sitting_states=None, bad_posture_states=None):
    '''
    Parameters
        left_display (np.array): numpy array containing image data
        img_scale (list[float])
        objects (list[sl.ObjectData]) 
    '''
    overlay = left_display.copy()

    # Render skeleton joints and bones
    for obj in objects:
        if render_object(obj, is_tracking_on):
            if len(obj.keypoint_2d) > 0:
                color = generate_color_id_u(obj.id)
                if body_format == sl.BODY_FORMAT.BODY_18:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_18_BONES, sitting_states,  bad_posture_states)
                elif body_format == sl.BODY_FORMAT.BODY_34:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_34_BONES, sitting_states,  bad_posture_states)
                elif body_format == sl.BODY_FORMAT.BODY_38:
                    render_sk(left_display, img_scale, obj, color, sl.BODY_38_BONES, sitting_states,  bad_posture_states) 

    cv2.addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display)