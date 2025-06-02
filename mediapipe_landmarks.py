import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Função para detectar o gesto
def detectar_gesto(landmarks):
    dedos_abertos = 0

    # Polegar: eixo X (compara dedo com ponto 2 e 4)
    if landmarks[4].x < landmarks[3].x:
        dedos_abertos += 1

    # Indicador, médio, anelar e mínimo: eixo Y
    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            dedos_abertos += 1

    if dedos_abertos == 5:
        return "Mao aberta"
    elif dedos_abertos == 0:
        return "Punho fechado"
    elif dedos_abertos == 1:
        return "1 dedo aberto"
    else:
        return f"{dedos_abertos} dedos abertos"

# Captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesto = detectar_gesto(hand_landmarks.landmark)
            cv2.putText(frame, gesto, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Reconhecimento de Gestos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
