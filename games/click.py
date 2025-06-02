import cv2
import mediapipe as mp
import pyautogui
import math
import random
import pygame
import time

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Inicializa pygame para sons
pygame.mixer.init()
# Cria um som simples de "pop" (você pode substituir por um arquivo .wav)
try:
    # Se você tiver um arquivo de som, descomente a linha abaixo:
    # pop_sound = pygame.mixer.Sound("pop.wav")
    pop_sound = None
except:
    pop_sound = None

# Webcam
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# Configurações do jogo
balloon_radius = 40
num_balloons = 5
balloons = []
score = 0
clicked = False

# Timer
game_duration = 60  # 60 segundos
start_time = time.time()

# Inicializa balões
def create_balloon():
    return {
        'x': random.randint(balloon_radius, 600 - balloon_radius),
        'y': random.randint(balloon_radius, 400 - balloon_radius),
        'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    }

# Cria balões iniciais
for _ in range(num_balloons):
    balloons.append(create_balloon())

def euclidean_dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def play_pop_sound():
    if pop_sound:
        pop_sound.play()
    else:
        # Som simples usando beep (alternativa)
        print('\a', end='')  # Beep do sistema

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Calcula tempo restante
    elapsed_time = time.time() - start_time
    remaining_time = max(0, game_duration - elapsed_time)
    
    # Verifica se o jogo acabou
    if remaining_time <= 0:
        cv2.putText(frame, f"GAME OVER! Pontuacao Final: {score}", 
                   (frame_w//2 - 200, frame_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Mouse com Mão - Estoura Balão", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    index_x, index_y = 0, 0
    pinch = False

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lm_list = hand.landmark
            index = lm_list[8]
            thumb = lm_list[4]

            index_x = int(index.x * frame_w)
            index_y = int(index.y * frame_h)
            thumb_x = int(thumb.x * frame_w)
            thumb_y = int(thumb.y * frame_h)

            dist = euclidean_dist(index_x, index_y, thumb_x, thumb_y)

            # Detecta gesto de pinça
            if dist < 30:
                pinch = True
                cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
            else:
                cv2.circle(frame, (index_x, index_y), 15, (255, 0, 0), cv2.FILLED)

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Desenha todos os balões
    for balloon in balloons:
        cv2.circle(frame, (balloon['x'], balloon['y']), balloon_radius, balloon['color'], -1)

    # Exibe informações do jogo
    cv2.putText(frame, f"Pontos: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, f"Tempo: {int(remaining_time)}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, f"Baloes: {len(balloons)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Verifica clique em qualquer balão
    if pinch and not clicked:
        for i, balloon in enumerate(balloons):
            if euclidean_dist(index_x, index_y, balloon['x'], balloon['y']) < balloon_radius:
                score += 1
                play_pop_sound()
                balloons[i] = create_balloon()  # Substitui o balão estourado por um novo
                clicked = True
                break
    
    if not pinch:
        clicked = False

    cv2.imshow("Mouse com Mão - Estoura Balão", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
