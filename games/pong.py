import cv2
import mediapipe as mp
import random
import time
import pygame

# Inicializando MediaPipe e captura
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Detectar até 2 mãos
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Parâmetros do jogo
width, height = 640, 480
raquete_altura = 150  # Aumentado de 100 para 120
raquete_largura = 10
bola_tamanho = 20

# Posição inicial das raquetes
raquete1_y = height // 2  # Player 1 (esquerda)
raquete2_y = height // 2  # Player 2 (direita)
bola_x = width // 2
bola_y = height // 2
base_speed = 10
vel_x = base_speed * random.choice([-1, 1])  # Usar base_speed em vez de 5
vel_y = base_speed * random.choice([-1, 1])  # Usar base_speed em vez de 5
score_player1 = 0
score_player2 = 0
start_time = time.time()

# Inicializa pygame mixer para som
pygame.mixer.init()
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

    # Detecta as mãos
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            # Pega a posição x da mão para determinar lado
            x = int(hand_landmark.landmark[0].x * width)
            y = int(hand_landmark.landmark[0].y * height)
            
            # Se a mão está no lado esquerdo da tela, controla raquete esquerda
            if x < width // 2:
                raquete1_y = y - raquete_altura // 2
            # Se a mão está no lado direito da tela, controla raquete direita
            else:
                raquete2_y = y - raquete_altura // 2
                
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # Limita movimento das raquetes dentro da tela
    raquete1_y = max(0, min(height - raquete_altura, raquete1_y))
    raquete2_y = max(0, min(height - raquete_altura, raquete2_y))

    # Calcula velocidade baseada no tempo
    elapsed_time = time.time() - start_time
    speed_multiplier = 1 + (elapsed_time // 3) * 0.1  # Aumenta 10% a cada 5 segundos (mais suave)
    current_speed = int(base_speed * speed_multiplier)

    # Atualiza posição da bola
    bola_x += int(vel_x * speed_multiplier)
    bola_y += int(vel_y * speed_multiplier)

    # Colisão com as paredes superior e inferior
    if bola_y <= 0 or bola_y >= height - bola_tamanho:
        vel_y *= -1

    # Colisão com a raquete do Player 1 (esquerda)
    if bola_x <= raquete_largura + 10 and vel_x < 0:
        if raquete1_y < bola_y < raquete1_y + raquete_altura:
            vel_x *= -1
            beep_sound.play()
        else:
            # Player 2 marca ponto
            score_player2 += 1
            bola_x, bola_y = width // 2, height // 2
            vel_x = base_speed * random.choice([-1, 1])
            vel_y = base_speed * random.choice([-1, 1])
            start_time = time.time()  # Reset timer

    # Colisão com a raquete do Player 2 (direita)
    if bola_x >= width - raquete_largura - 10 - bola_tamanho and vel_x > 0:
        if raquete2_y < bola_y < raquete2_y + raquete_altura:
            vel_x *= -1
            beep_sound.play()
        else:
            # Player 1 marca ponto
            score_player1 += 1
            bola_x, bola_y = width // 2, height // 2
            vel_x = base_speed * random.choice([-1, 1])
            vel_y = base_speed * random.choice([-1, 1])
            start_time = time.time()  # Reset timer

    # Desenho do jogo
    # Raquete Player 1 (esquerda)
    img = cv2.rectangle(img, (10, raquete1_y), (10 + raquete_largura, raquete1_y + raquete_altura), (0, 255, 0), -1)
    # Raquete Player 2 (direita)
    img = cv2.rectangle(img, (width - 20, raquete2_y), (width - 10, raquete2_y + raquete_altura), (0, 0, 255), -1)
    # Bola
    img = cv2.circle(img, (bola_x, bola_y), bola_tamanho, (255, 255, 255), -1)
    
    # Placar
    cv2.putText(img, f"P1: {score_player1}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"P2: {score_player2}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"Velocidade: {current_speed}", (width // 2 - 80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Linha central
    cv2.line(img, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

    cv2.imshow("Pong 2 Players ✋✋", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
