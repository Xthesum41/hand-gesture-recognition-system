import cv2
import numpy as np
import math
import time
from collections import deque

class HandGestureRecognizer:
    def __init__(self):
        # Kernels básicos
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # ROI configuration
        self.roi = {
            'top': 50, 'bottom': 450,
            'left': 150, 'right': 550
        }
        
        # Background model
        self.bg_model = None
        self.bg_accumulator = None
        self.bg_captured = False
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.last_fps_update = 0
    
    def _get_roi(self, frame):
        """Extract ROI efficiently"""
        return frame[self.roi['top']:self.roi['bottom'], 
                    self.roi['left']:self.roi['right']]
    
    def capture_background(self, frame):
        """Captura simples do background"""
        roi = self._get_roi(frame)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.bg_model = cv2.GaussianBlur(gray_roi, (15, 15), 0).astype(np.float32)
        self.bg_captured = True
        return True
    
    def segment_hand(self, frame):
        """Segmentação otimizada"""
        if self.bg_model is None:
            return None
            
        roi = self._get_roi(frame)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (19, 19), 0)
        
        # Background subtraction
        diff = cv2.absdiff(self.bg_model.astype(np.uint8), gray_roi)
        _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel_large, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_small, iterations=1)
        
        return binary
    
    def detect_skin(self, frame):
        """Detecção de pele otimizada"""
        roi = self._get_roi(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Range para pele
        lower_hsv = np.array([0, 30, 60])
        upper_hsv = np.array([20, 150, 255])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Filtros
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_large, iterations=2)
        
        return mask
    
    def count_fingers(self, contour):
        """Contador de dedos com detecção melhorada para 1 dedo"""
        try:
            area = cv2.contourArea(contour)
            if area < 2000:
                return 0, []
            
            # Calcular centro
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return 0, []
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            # DETECÇÃO APRIMORADA PARA 1 DEDO - Múltiplas validações
            
            # Método 1: Análise de forma alongada
            if area < 18000 and aspect_ratio > 1.0:  # Critérios mais amplos
                topmost = tuple(contour[contour[:,:,1].argmin()][0])
                
                # Verificar se o topo está suficientemente acima do centro
                height_from_center = cy - topmost[1]
                min_height_threshold = max(20, h * 0.12)  # Threshold adaptativo mais baixo
                
                if height_from_center > min_height_threshold:
                    # Análise da largura ao longo da altura
                    finger_valid = self._validate_finger_shape(contour, topmost, cx, cy, w, h)
                    
                    if finger_valid:
                        return 1, [topmost]
            
            # Método 2: Análise por aproximação de contorno
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Se tem poucos vértices e é alongado, pode ser 1 dedo
            if len(approx) <= 8 and aspect_ratio > 1.1 and area < 16000:
                # Encontrar o ponto mais alto
                topmost = tuple(contour[contour[:,:,1].argmin()][0])
                
                # Verificar se é estreito na parte superior
                upper_third_y = y + h // 3
                if topmost[1] < upper_third_y:
                    width_at_top = self._get_width_at_height(contour, topmost[1])
                    width_ratio = width_at_top / w if w > 0 else 1
                    
                    # Se é estreito no topo comparado à base
                    if width_ratio < 0.7:
                        return 1, [topmost]
            
            # Método 3: Análise de solidez e convexidade
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if area < 20000 and solidity > 0.75 and aspect_ratio > 1.05:
                topmost = tuple(contour[contour[:,:,1].argmin()][0])
                
                # Verificar se o ponto mais alto está numa posição razoável
                relative_top_position = (cy - topmost[1]) / h if h > 0 else 0
                
                if relative_top_position > 0.15:  # Pelo menos 15% da altura acima do centro
                    # Análise adicional: verificar se não é muito irregular
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Um dedo deve ter baixa circularidade (não ser circular)
                    if circularity < 0.6:
                        return 1, [topmost]
            
            # Método 4: Detecção por análise de esqueleto (simplificada)
            if area > 3000 and area < 25000 and aspect_ratio > 0.8:
                topmost = tuple(contour[contour[:,:,1].argmin()][0])
                
                # Verificar densidade de pontos na parte superior vs inferior
                upper_half_count = np.sum(contour[:, :, 1] < cy)
                lower_half_count = np.sum(contour[:, :, 1] >= cy)
                
                # Um dedo deve ter menos pontos na parte superior (mais estreito)
                if upper_half_count > 0 and lower_half_count > 0:
                    upper_lower_ratio = upper_half_count / lower_half_count
                    
                    if 0.3 < upper_lower_ratio < 0.9 and topmost[1] < cy - 10:
                        return 1, [topmost]
            
            # Convex hull para múltiplos dedos (código existente)
            hull = cv2.convexHull(contour, returnPoints=False)
            fingertips = []
            
            if len(hull) >= 4:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        
                        # Calcular ângulo
                        try:
                            a = np.linalg.norm(np.array(end) - np.array(start))
                            b = np.linalg.norm(np.array(far) - np.array(start))
                            c = np.linalg.norm(np.array(end) - np.array(far))
                            
                            if b > 0 and c > 0:
                                cos_angle = max(-1, min(1, (b**2 + c**2 - a**2) / (2 * b * c)))
                                angle = math.acos(cos_angle) * 180 / math.pi
                            else:
                                continue
                        except:
                            continue
                        
                        # Critérios
                        defect_depth = d / 256.0
                        if defect_depth > 15 and angle < 80:
                            if start[1] < cy - 10:
                                fingertips.append(start)
                            if end[1] < cy - 10:
                                fingertips.append(end)
            
            # Remover duplicatas
            unique_fingertips = []
            for tip in fingertips:
                is_unique = True
                for existing in unique_fingertips:
                    if np.linalg.norm(np.array(tip) - np.array(existing)) < 35:
                        is_unique = False
                        break
                if is_unique:
                    unique_fingertips.append(tip)
            
            finger_count = len(unique_fingertips)
            
            # FALLBACK PARA 1 DEDO - Se não detectou nada mas parece ser 1 dedo
            if finger_count == 0 and area > 3000 and area < 15000 and aspect_ratio > 1.15:
                topmost = tuple(contour[contour[:,:,1].argmin()][0])
                if topmost[1] < cy - 5:  # Threshold bem baixo
                    return 1, [topmost]
            
            # Limites baseados na área (ajustados)
            if area < 12000:  # Aumentado de 10000
                finger_count = min(finger_count, 2)
            elif area < 20000:
                finger_count = min(finger_count, 3)
            
            return min(finger_count, 5), unique_fingertips[:5]
            
        except:
            return 0, []
    
    def _validate_finger_shape(self, contour, topmost, cx, cy, w, h):
        """Validar se a forma se parece com um dedo"""
        try:
            # Analisar largura em diferentes alturas
            top_y = topmost[1]
            mid_y = cy
            
            # Largura no topo (1/4 superior)
            top_width = self._get_width_at_height(contour, top_y + h // 4)
            
            # Largura no meio
            mid_width = self._get_width_at_height(contour, mid_y)
            
            # Largura na base (3/4 para baixo)
            base_width = self._get_width_at_height(contour, top_y + 3 * h // 4)
            
            # Um dedo deve ser mais estreito no topo
            if top_width > 0 and mid_width > 0:
                top_mid_ratio = top_width / mid_width
                if top_mid_ratio > 1.2:  # Se o topo é mais largo que o meio, não é dedo
                    return False
            
            # Verificar se não é muito largo em geral
            avg_width = (top_width + mid_width + base_width) / 3
            width_height_ratio = avg_width / h if h > 0 else 1
            
            # Um dedo não deve ser muito largo em relação à altura
            if width_height_ratio > 0.4:
                return False
            
            return True
            
        except:
            return True  # Em caso de erro, assumir válido
    
    def _get_width_at_height(self, contour, target_y):
        """Calcular largura do contorno numa altura específica"""
        try:
            # Encontrar pontos próximos à altura alvo
            points_at_height = []
            tolerance = 3
            
            for point in contour:
                if abs(point[0][1] - target_y) <= tolerance:
                    points_at_height.append(point[0][0])
            
            if len(points_at_height) < 2:
                return 0
            
            return max(points_at_height) - min(points_at_height)
            
        except:
            return 0
    
    def update_fps(self):
        """Atualizar FPS"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        self.frame_times.append(frame_time)
        
        if current_time - self.last_fps_update >= 1.0:
            if len(self.frame_times) > 5:
                avg_frame_time = np.mean(list(self.frame_times)[-10:])
                self.fps = 1.0 / max(0.001, avg_frame_time)
            self.last_fps_update = current_time
    
    def draw_text_with_bg(self, frame, text, position, font_scale=0.6, 
                         color=(255, 255, 255), bg_color=(40, 40, 40)):
        """Desenhar texto com fundo"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
        
        x, y = position
        padding = 5
        cv2.rectangle(frame, (x - padding, y - text_size[1] - padding), 
                     (x + text_size[0] + padding, y + padding), bg_color, -1)
        cv2.putText(frame, text, position, font, font_scale, color, 1)
    
    def draw_fingertips(self, frame, fingertips):
        """Desenhar fingertips"""
        for i, tip in enumerate(fingertips):
            tip_point = (int(tip[0]), int(tip[1])) if isinstance(tip, tuple) else tuple(tip.astype(int))
            
            cv2.circle(frame, tip_point, 7, (0, 0, 200), -1)
            cv2.circle(frame, tip_point, 7, (255, 255, 255), 1)
            cv2.putText(frame, str(i + 1), (tip_point[0] - 5, tip_point[1] + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_ui(self, frame, contour=None, fingertips=None, bg_captured=True):
        """Interface simplificada"""
        # ROI
        roi_color = (0, 180, 0) if bg_captured else (0, 0, 180)
        cv2.rectangle(frame, (self.roi['left'], self.roi['top']), 
                     (self.roi['right'], self.roi['bottom']), roi_color, 2)
        
        # Contorno e fingertips
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 200, 0), 1)
        
        if fingertips is not None and len(fingertips) > 0:
            self.draw_fingertips(frame, fingertips)
        
        # Controles
        self.draw_text_with_bg(frame, "b=BG | s=Skin | m=Mask | r=Reset | q=Quit", 
                              (15, 55), 0.4, (200, 200, 200))
        
        # FPS
        self.draw_text_with_bg(frame, f"FPS: {self.fps:.1f}", 
                              (15, frame.shape[0] - 25), 0.5)
    
    def run(self):
        """Loop principal otimizado"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Erro: Câmera não encontrada")
            return
        
        print("=== SISTEMA OTIMIZADO ===")
        print("b = Background | s = Skin mode | m = Mask | r = Reset | q = Quit")
        
        use_skin_mode = False
        show_mask = False
        last_detection = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.update_fps()
            
            # Processamento principal
            if self.bg_captured:
                hand_mask = self.detect_skin(frame) if use_skin_mode else self.segment_hand(frame)
                
                if hand_mask is not None:
                    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        if cv2.contourArea(largest_contour) > 1500:
                            # Ajustar coordenadas
                            largest_contour[:, :, 0] += self.roi['left']
                            largest_contour[:, :, 1] += self.roi['top']
                            
                            finger_count, fingertips = self.count_fingers(largest_contour)
                            last_detection = (largest_contour, fingertips)
                        else:
                            last_detection = None
                    else:
                        last_detection = None
                    
                    if show_mask:
                        cv2.imshow('Hand Mask', hand_mask)
            
            # Desenhar UI
            contour, fingertips = last_detection if last_detection else (None, None)
            self.draw_ui(frame, contour, fingertips, self.bg_captured)
            
            cv2.imshow('Hand Gesture - Optimized', frame)
            
            # Input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b') and not self.bg_captured:
                self.capture_background(frame)
                print("Background capturado!")
            elif key == ord('r'):
                self.__init__()
                last_detection = None
                print("Sistema resetado")
            elif key == ord('s'):
                use_skin_mode = not use_skin_mode
                print(f"Modo: {'Pele' if use_skin_mode else 'Background'}")
            elif key == ord('m'):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow('Hand Mask')
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = HandGestureRecognizer()
    recognizer.run()