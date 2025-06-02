import cv2
import mediapipe as mp
import numpy as np
import math

# Inicializa MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Tamanho da janela
WIDTH, HEIGHT = 1280, 720
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Cores disponíveis com caixas clicáveis
color_palette = [
    {'color': (255, 255, 255), 'name': 'Branco', 'box': {'x1': 10, 'y1': 120, 'x2': 60, 'y2': 170}},
    {'color': (0, 0, 255), 'name': 'Vermelho', 'box': {'x1': 10, 'y1': 180, 'x2': 60, 'y2': 230}},
    {'color': (255, 0, 0), 'name': 'Azul', 'box': {'x1': 10, 'y1': 240, 'x2': 60, 'y2': 290}},
    {'color': (0, 255, 0), 'name': 'Verde', 'box': {'x1': 10, 'y1': 300, 'x2': 60, 'y2': 350}},
    {'color': (0, 255, 255), 'name': 'Amarelo', 'box': {'x1': 10, 'y1': 360, 'x2': 60, 'y2': 410}},
    {'color': (255, 0, 255), 'name': 'Magenta', 'box': {'x1': 10, 'y1': 420, 'x2': 60, 'y2': 470}},
    {'color': (0, 0, 0), 'name': 'Preto', 'box': {'x1': 10, 'y1': 480, 'x2': 60, 'y2': 530}}
]
current_color = (255, 0, 255)  # Cor padrão (magenta)

# Botão limpar
clear_button = {'x1': WIDTH-120, 'y1': 10, 'x2': WIDTH-10, 'y2': 60}

def count_fingers(landmarks):
    """Conta o número de dedos levantados"""
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]
    
    fingers_up = 0
    
    # Polegar (diferente dos outros dedos)
    if landmarks[4].x > landmarks[3].x:
        fingers_up += 1
    
    # Outros dedos
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            fingers_up += 1
    
    return fingers_up

def get_brush_thickness(landmarks):
    """Calcula espessura do pincel baseada na distância da mão à câmera"""
    # Usa a coordenada Z do pulso (landmark 0) para determinar profundidade
    wrist_z = landmarks[0].z
    # Converte para espessura (valores mais negativos = mais perto = mais grosso)
    thickness = max(5, min(25, int(15 + wrist_z * 100)))
    return thickness

def check_color_selection(x, y, pinch_detected):
    """Verifica se o usuário clicou em alguma cor"""
    global current_color
    
    for color_info in color_palette:
        box = color_info['box']
        if (box['x1'] < x < box['x2'] and box['y1'] < y < box['y2']):
            if pinch_detected:
                current_color = color_info['color']
                return True, color_info['name']
            else:
                return False, color_info['name']  # Apenas hovering
    return False, None

# Inicia webcam
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Inicializa MediaPipe Hands
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    drawing = False
    prev_x, prev_y = 0, 0
    brush_thickness = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Desenha botão limpar
        cv2.rectangle(frame, (clear_button['x1'], clear_button['y1']), 
                     (clear_button['x2'], clear_button['y2']), (0, 0, 255), -1)
        cv2.putText(frame, 'LIMPAR', (clear_button['x1']+5, clear_button['y1']+35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Desenha paleta de cores
        for color_info in color_palette:
            box = color_info['box']
            # Borda branca se for a cor atual
            border_color = (255, 255, 255) if color_info['color'] == current_color else (100, 100, 100)
            cv2.rectangle(frame, (box['x1']-2, box['y1']-2), (box['x2']+2, box['y2']+2), border_color, 2)
            cv2.rectangle(frame, (box['x1'], box['y1']), (box['x2'], box['y2']), color_info['color'], -1)

        # Mostra cor atual e espessura
        cv2.rectangle(frame, (10, 10), (100, 60), current_color, -1)
        cv2.putText(frame, f'Espessura: {brush_thickness}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            # Calcula espessura do pincel
            brush_thickness = get_brush_thickness(lm)

            # Coordenadas do dedo indicador e polegar
            x1 = int(lm[8].x * WIDTH)   # Indicador
            y1 = int(lm[8].y * HEIGHT)
            x2 = int(lm[4].x * WIDTH)   # Polegar
            y2 = int(lm[4].y * HEIGHT)

            # Distância entre polegar e indicador para detectar pinça
            dist = math.hypot(x2 - x1, y2 - y1)
            pinch_detected = dist < 40

            # Verifica seleção de cor
            color_clicked, color_name = check_color_selection(x1, y1, pinch_detected)
            
            if color_name:
                # Mostra nome da cor quando hovering ou clicando
                cv2.putText(frame, color_name, (70, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if color_clicked:
                    cv2.circle(frame, (x1, y1), 20, (0, 255, 0), 3)  # Feedback visual de seleção
                else:
                    cv2.circle(frame, (x1, y1), 15, (255, 255, 0), 2)  # Hovering
            # Verifica clique no botão limpar
            elif (clear_button['x1'] < x1 < clear_button['x2'] and 
                  clear_button['y1'] < y1 < clear_button['y2']):
                if pinch_detected:
                    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), -1)  # Feedback visual
                else:
                    cv2.circle(frame, (x1, y1), 10, (255, 255, 0), -1)  # Sobre o botão
            else:
                # Modo desenho normal
                if pinch_detected:
                    # Desenhar quando os dedos estiverem próximos
                    drawing = True
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x1, y1
                    cv2.line(canvas, (prev_x, prev_y), (x1, y1), current_color, brush_thickness)
                    prev_x, prev_y = x1, y1
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), -1)  # Verde quando desenhando
                else:
                    drawing = False
                    prev_x, prev_y = 0, 0
                    cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)  # Círculo normal

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Combina a pintura com a imagem da webcam
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, canvas)

        cv2.imshow('Pintura com a Mão 🖐️', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
