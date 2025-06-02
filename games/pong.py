import cv2
import mediapipe as mp
import random
import time
import pygame

# Inicializando MediaPipe e captura
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Parâmetros do jogo
width, height = 640, 480
raquete_altura = 100
raquete_largura = 10
bola_tamanho = 20

# Posição inicial
raquete_y = height // 2
bola_x = width // 2
bola_y = height // 2
vel_x = 5 * random.choice([-1, 1])
vel_y = 5 * random.choice([-1, 1])
score = 0
start_time = time.time()
base_speed = 5

# Inicializa pygame mixer para som
pygame.mixer.init()
# Cria um som simples (beep) programaticamente
pygame.mixer.set_num_channels(1)

# Função para criar som de beep
def create_beep_sound():
    import numpy as np
    sample_rate = 22050
    duration = 0.1
    frequency = 800
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2))
    for i in range(frames):
        wave = 4096 * np.sin(2 * np.pi * frequency * i / sample_rate)
        arr[i] = [wave, wave]
    sound = pygame.sndarray.make_sound(arr.astype(np.int16))
    return sound

beep_sound = create_beep_sound()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (width, height))

    # Detecta a mão
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            y = int(hand_landmark.landmark[0].y * height)
            raquete_y = y - raquete_altura // 2
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # Calcula velocidade baseada no tempo
    elapsed_time = time.time() - start_time
    speed_multiplier = 1 + (elapsed_time // 10) * 0.2  # Aumenta 20% a cada 10 segundos
    current_speed = int(base_speed * speed_multiplier)

    # Atualiza posição da bola
    bola_x += int(vel_x * speed_multiplier)
    bola_y += int(vel_y * speed_multiplier)

    # Colisão com as paredes superior e inferior
    if bola_y <= 0 or bola_y >= height - bola_tamanho:
        vel_y *= -1

    # Colisão com a raquete
    if bola_x <= raquete_largura + 10:
        if raquete_y < bola_y < raquete_y + raquete_altura:
            vel_x *= -1
            score += 1
            beep_sound.play()  # Toca som ao rebater
        else:
            score = 0
            bola_x, bola_y = width // 2, height // 2
            vel_x = base_speed * random.choice([-1, 1])
            vel_y = base_speed * random.choice([-1, 1])
            start_time = time.time()  # Reset timer

    # Colisão com parede da direita
    if bola_x >= width - bola_tamanho:
        vel_x *= -1

    # Desenho do jogo
    img = cv2.rectangle(img, (10, raquete_y), (10 + raquete_largura, raquete_y + raquete_altura), (255, 255, 255), -1)
    img = cv2.circle(img, (bola_x, bola_y), bola_tamanho, (255, 255, 255), -1)
    cv2.putText(img, f"Pontos: {score}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Velocidade: {current_speed}", (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Pong com a Mão ✋", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
