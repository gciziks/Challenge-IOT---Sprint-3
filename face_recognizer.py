import cv2
import numpy as np
import os
import time
import pickle

class FaceRecognizer:
    def __init__(self):
        # Parâmetros ajustáveis do sistema
        self.face_detector = None
        self.face_recognizer = None
        self.recognizer_trained = False
        self.dataset_path = "facial_dataset"
        self.recognizer_file = "face_recognizer.yml"
        self.labels_file = "labels.pickle"
        
        # Parâmetros de detecção
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
        self.confidence_threshold = 70
        
        # Histórico de reconhecimentos para filtragem
        self.recognition_history = []
        self.history_size = 5
        
        # Inicializar detectores
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        try:
            # Carregar classificadores Haar Cascade
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Inicializar reconhecedor LBPH
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=2, neighbors=16, grid_x=8, grid_y=8
            )
            
            # Tentar carregar modelo treinado se existir
            if os.path.exists(self.recognizer_file):
                self.face_recognizer.read(self.recognizer_file)
                self.recognizer_trained = True
                print("Modelo de reconhecimento carregado com sucesso!")
                
        except Exception as e:
            print(f"Erro ao inicializar detectores: {e}")
            raise
    
    def adjust_detection_parameters(self, scale_factor=None, min_neighbors=None, 
                                  min_size=None, confidence=None):
        if scale_factor is not None:
            self.scale_factor = max(1.05, min(scale_factor, 1.5))
        if min_neighbors is not None:
            self.min_neighbors = max(1, min(min_neighbors, 10))
        if min_size is not None:
            self.min_size = (max(20, min_size[0]), max(20, min_size[1]))
        if confidence is not None:
            self.confidence_threshold = max(40, min(confidence, 90))
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar faces com parâmetros ajustáveis
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces, gray
    
    def recognize_face(self, face_roi):
        if not self.recognizer_trained:
            return "Sistema não treinado", 100, -1
        
        try:
            # Pré-processamento da imagem
            face_resized = cv2.resize(face_roi, (200, 200))
            
            face_resized = cv2.equalizeHist(face_resized)
            
            # Predição
            label, confidence = self.face_recognizer.predict(face_resized)
            # Carregar labels
            if os.path.exists(self.labels_file):
                with open(self.labels_file, 'rb') as f:
                    labels_dict = pickle.load(f)
                name = labels_dict.get(label, "Desconhecido")
            else:
                name = "Desconhecido"
                confidence = 100
            
            return name, confidence, label
            
        except Exception as e:
            print(f"Erro no reconhecimento: {e}")
            return "Erro", 100, -1
    
    def update_recognition_history(self, name, confidence, label):
        self.recognition_history.append({
            'name': name,
            'confidence': confidence,
            'label': label,
            'timestamp': time.time()
        })
        
        # Manter apenas os últimos reconhecimentos
        if len(self.recognition_history) > self.history_size:
            self.recognition_history.pop(0)
    
    def get_filtered_recognition(self):
        if not self.recognition_history:
            return "Desconhecido", 100, -1
        
        # Contar ocorrências de cada nome
        name_counts = {}
        for rec in self.recognition_history:
            if rec['name'] != "Desconhecido":
                name_counts[rec['name']] = name_counts.get(rec['name'], 0) + 1
        
        if not name_counts:
            return "Desconhecido", 100, -1
        
        # Retornar nome mais frequente
        most_common = max(name_counts.items(), key=lambda x: x[1])
        return most_common[0], 50, -1
    
    def collect_training_data(self, user_name, num_samples=30):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro: Não foi possível acessar a câmera")
            return False
        
        user_path = os.path.join(self.dataset_path, user_name)
        if not os.path.exists(user_path):
            os.makedirs(user_path)
        
        print(f"Coletando {num_samples} amostras para {user_name}...")
        print("Posicione-se frente à câmera e aguarde a coleta")
        print("Mova levemente a cabeça para diferentes ângulos durante a coleta")
        
        count = 0
        last_capture_time = 0
        capture_interval = 0.3  # 300ms entre capturas
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            current_time = time.time()
            faces, gray = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                if current_time - last_capture_time >= capture_interval:
                    # Garantir tamanho mínimo do rosto
                    if w > 60 and h > 60:
                        # Salvar imagem do rosto
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Aplicar pré-processamento add
                        face_roi = cv2.equalizeHist(face_roi)
                        
                        face_filename = os.path.join(user_path, f"{count:03d}.jpg")
                        cv2.imwrite(face_filename, face_roi)
                        
                        count += 1
                        last_capture_time = current_time
                
                # Desenhar retângulo e contador
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Amostra: {count}/{num_samples}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Mova a cabeça lentamente", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Coleta de Dados - Pressione ESC para cancelar', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count >= 10:
            success = self.train_recognizer()
            if success:
                print(f"Coleta concluída! {count} amostras coletadas.")
            return success
        else:
            print("Coleta cancelada ou número insuficiente de amostras (mínimo: 10)")
            return False
    
    def train_recognizer(self):
        faces = []
        labels = []
        label_ids = {}
        current_id = 0
        
        if not os.path.exists(self.dataset_path):
            print("Diretório de dataset não encontrado!")
            return False
        
        print("Iniciando treinamento do reconhecedor...")
        
        # Percorrer dataset
        for root, dirs, files in os.walk(self.dataset_path):
            for file in sorted(files):
                if file.endswith(('jpg', 'jpeg', 'png')):
                    path = os.path.join(root, file)
                    label = os.path.basename(root)
                    
                    if label not in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    
                    # Ler e pré-processar imagem
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Redimensionar e equalizar
                        img = cv2.resize(img, (200, 200))
                        img = cv2.equalizeHist(img)
                        faces.append(img)
                        labels.append(label_ids[label])
        
        if len(faces) == 0:
            print("Nenhuma imagem encontrada para treinamento!")
            return False
        
        print(f"Treinando com {len(faces)} imagens de {len(label_ids)} pessoas...")
        
        # Treinar reconhecedor
        try:
            start_time = time.time()
            self.face_recognizer.train(faces, np.array(labels))
            training_time = time.time() - start_time
            
            self.face_recognizer.save(self.recognizer_file)
            
            # Salvar mapeamento de labels
            with open(self.labels_file, 'wb') as f:
                pickle.dump({v: k for k, v in label_ids.items()}, f)
            
            self.recognizer_trained = True
            print(f"Treinamento concluído em {training_time:.1f} segundos!")
            print(f"Total: {len(faces)} imagens, {len(label_ids)} pessoas")
            return True
            
        except Exception as e:
            print(f"Erro no treinamento: {e}")
            return False
    
    def display_parameters(self, frame):
        params_text = [
            f"Scale: {self.scale_factor:.2f}",
            f"Neighbors: {self.min_neighbors}",
            f"Min Size: {self.min_size[0]}x{self.min_size[1]}",
            f"Conf Threshold: {self.confidence_threshold}%",
            f"Status: {'Treinado' if self.recognizer_trained else 'Não Treinado'}"
        ]
        
        for i, text in enumerate(params_text):
            cv2.putText(frame, text, (10, 60 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def display_help(self, frame):
        help_text = [
            "Comandos: 1-4:Scale 5-8:Neighbors 9:Reset",
            "T:Treinar R:Retreinar H:Ajuda ESC:Sair"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (10, frame.shape[0] - 60 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def run_recognition(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro: Não foi possível acessar a câmera")
            return
        
        print("\n=== Sistema Avançado de Reconhecimento Facial ===")
        print("Comandos:")
        print("1-4: Ajustar Scale Factor (1.05-1.5)")
        print("5-8: Ajustar Min Neighbors (1-10)")
        print("9: Resetar parâmetros para padrão")
        print("T: Treinar novo usuário")
        print("R: Retreinar reconhecedor")
        print("H: Mostrar/ocultar ajuda")
        print("ESC: Sair")
        
        show_help = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Detectar faces
            faces, gray = self.detect_faces(frame)
            
            # Processar cada face detectada
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Reconhecer face
                name, confidence, label = self.recognize_face(face_roi)
                
                # Atualizar histórico
                self.update_recognition_history(name, confidence, label)
                
                # Obter reconhecimento filtrado
                filtered_name, filtered_conf, _ = self.get_filtered_recognition()
                
              
                if confidence < self.confidence_threshold:
                    color = (0, 255, 0)  # Verde - reconhecido
                    status = f"{name} ({confidence:.1f})"
                else:
                    color = (0, 0, 255)  # Vermelho - desconhecido
                    status = f"Desconhecido ({confidence:.1f})"
                
                # Usar resultado filtrado se disponível
                if filtered_name != "Desconhecido":
                    status = f"{filtered_name} (filtrado)"
                    color = (0, 255, 255)  # Amarelo para filtrado
                
                # Desenhar retângulo e infos
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, status, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Desenhar landmarks
                cv2.circle(frame, (x + w//4, y + h//3), 4, color, -1)
                cv2.circle(frame, (x + 3*w//4, y + h//3), 4, color, -1)
                cv2.ellipse(frame, (x + w//2, y + 2*h//3), 
                           (w//6, h//12), 0, 0, 180, color, 2)
            
            # Exibir informações do sistema
            cv2.putText(frame, "Sistema de Reconhecimento Facial - OpenCV", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Exibir parâmetros atuais
            self.display_parameters(frame)
            
            # Exibir ajuda
            if show_help:
                self.display_help(frame)
            
            # Exibir frame
            cv2.imshow('Reconhecimento Facial Avançado', frame)
            
            # Processar comandos do teclado
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('1'):
                self.adjust_detection_parameters(scale_factor=1.05)
            elif key == ord('2'):
                self.adjust_detection_parameters(scale_factor=1.2)
            elif key == ord('3'):
                self.adjust_detection_parameters(scale_factor=1.3)
            elif key == ord('4'):
                self.adjust_detection_parameters(scale_factor=1.5)
            elif key == ord('5'):
                self.adjust_detection_parameters(min_neighbors=1)
            elif key == ord('6'):
                self.adjust_detection_parameters(min_neighbors=3)
            elif key == ord('7'):
                self.adjust_detection_parameters(min_neighbors=7)
            elif key == ord('8'):
                self.adjust_detection_parameters(min_neighbors=10)
            elif key == ord('9'):
                # Reset para parâmetros padrão
                self.adjust_detection_parameters(scale_factor=1.1, min_neighbors=5, 
                                               min_size=(30, 30), confidence=70)
            elif key == ord('t'):
                cap.release()
                cv2.destroyAllWindows()
                user_name = input("Digite o nome do usuário: ")
                success = self.collect_training_data(user_name)
                if success:
                    print("Treinamento concluído com sucesso!")
                cap = cv2.VideoCapture(0)
            elif key == ord('r'):
                success = self.train_recognizer()
                if success:
                    print("Retreinamento concluído!")
            elif key == ord('h'):
                show_help = not show_help
        
        cap.release()
        cv2.destroyAllWindows()
