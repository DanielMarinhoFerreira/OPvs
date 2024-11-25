import random
import torch
from ultralytics import YOLO
import cv2
import numpy as np

class DetectionSystem:

    def __init__(self, train_model=False):
        if train_model:
            self.model = YOLO("yolo11n.yaml").load("yolo11n.pt")
        else:
            self.model = YOLO('runs/detect/train/weights/best.pt')

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

    @staticmethod
    def evaluate_hyperparameters(params):
        learning_rate, epochs, img_size = params
        model = YOLO("yolo11n.yaml").load("yolo11n.pt")

        results = model.train(
            data="coco8.yaml",
            epochs=int(epochs),
            imgsz=int(img_size),
            lr0=learning_rate
        )

        return results.metrics['loss']

    @staticmethod
    def genetic_algorithm(pop_size, generations):
        population = [
            (random.uniform(0.0001, 0.01),  # Learning rate
             random.randint(10, 100),      # Epochs
             random.randint(320, 640))    # Image size
            for _ in range(pop_size)
        ]

        for gen in range(generations):
            print(f"Generation {gen}")
            fitness_scores = [DetectionSystem.evaluate_hyperparameters(ind) for ind in population]

            selected = [x for _, x in sorted(zip(fitness_scores, population))][:pop_size // 2]

            new_population = selected[:]
            while len(new_population) < pop_size:
                parent1, parent2 = random.sample(selected, 2)
                child = (
                    (parent1[0] + parent2[0]) / 2,  # Average learning rates
                    random.choice([parent1[1], parent2[1]]),  # Random epochs
                    random.choice([parent1[2], parent2[2]])   # Random image size
                )
                if random.random() < 0.1:
                    child = (
                        child[0] * random.uniform(0.8, 1.2),
                        child[1] + random.randint(-5, 5),
                        child[2] + random.randint(-32, 32)
                    )
                new_population.append(child)

            population = new_population

        best_individual = min(population, key=DetectionSystem.evaluate_hyperparameters)
        print(f"Best hyperparameters: {best_individual}")

if __name__ == "__main__":
    # Executar a otimização genética
    # DetectionSystem.genetic_algorithm(pop_size=10, generations=5)
    
    detector = DetectionSystem()
    detector.detect_from_webcam()

