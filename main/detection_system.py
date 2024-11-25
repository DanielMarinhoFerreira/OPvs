import torch
from ultralytics import YOLO
import cv2
import numpy as np

class DetectionSystem:
    
    def __init__(self, train_model=False):
        
        if train_model:
            self.model = YOLO("yolo11n.yaml")  # build a new model from YAM
        
            # Carrega o modelo YOLO pré-treinado
            self.model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
        
            self.model =  YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
        
            # Train the model
            results = self.model.train(data="coco8.yaml", epochs=100, imgsz=640)
            
        else:
            # Usa modelo pré-treinado
            self.model = YOLO('yolo11n.pt')
        
        # Dicionário de cores para diferentes classes
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
        
        # Classes do COCO dataset
        self.classes = ['pessoa', 'bicicleta', 'carro', 'moto', 'avião', 'ônibus', 'trem', 'caminhão', 
                       'barco', 'semáforo', 'hidrante', 'placa', 'parquímetro', 'banco', 'pássaro', 
                       'gato', 'cachorro', 'cavalo', 'ovelha', 'vaca', 'elefante', 'urso', 'zebra', 
                       'girafa', 'mochila', 'guarda-chuva', 'bolsa', 'gravata', 'mala', 'frisbee', 
                       'esquis', 'snowboard', 'bola', 'pipa', 'taco baseball', 'luva baseball', 
                       'skate', 'prancha surf', 'raquete tênis', 'garrafa', 'taça', 'copo', 'garfo', 
                       'faca', 'colher', 'tigela', 'banana', 'maçã', 'sanduíche', 'laranja', 'brócolis', 
                       'cenoura', 'cachorro-quente', 'pizza', 'rosquinha', 'bolo', 'cadeira', 'sofá', 
                       'vaso de planta', 'cama', 'mesa jantar', 'vaso sanitário', 'tv', 'laptop', 
                       'mouse', 'controle', 'teclado', 'celular', 'microondas', 'forno', 'torradeira', 
                       'pia', 'geladeira', 'livro', 'relógio', 'vaso', 'tesoura', 'ursinho', 
                       'secador', 'escova dentes']
    
    def draw_detection(self, image, box, cls_id, conf):
        x1, y1, x2, y2 = box
        color = self.colors[int(cls_id)].tolist()
        
        # Desenha o retângulo
        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )
        
        # Adiciona texto com a classe e confiança
        label = f'{self.classes[int(cls_id)]}: {conf:.2f}'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), color, -1)
        cv2.putText(
            image,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
    def detect_from_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Não foi possível carregar a imagem")
            
        results = self.model(image)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                self.draw_detection(
                    image,
                    box.xyxy[0],
                    box.cls[0],
                    box.conf[0]
                )
        
        return image
    
    def detect_from_webcam(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    self.draw_detection(
                        frame,
                        box.xyxy[0],
                        box.cls[0],
                        box.conf[0]
                    )
            
            # Mostra o resultado
            cv2.imshow('Detecção Multi-Objetos', frame)
            
            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = DetectionSystem()
    
    # Para usar a webcam
    detector.detect_from_webcam()

if __name__ == "__main__":
    main()