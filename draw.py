import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np

class DrawApp:
    def __init__(self, root, model_predict):
        self.root = root
        self.root.title("Digit Drawing Canvas")

        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        # Bind mouse drag to draw
        self.canvas.bind("<B1-Motion>", self.draw)

        # Button frame
        button_frame = tk.Frame(root)
        button_frame.pack()

        # Predict button
        self.predict_btn = tk.Button(button_frame, text="identify", command=self.identify)
        self.predict_btn.pack(side=tk.LEFT)

        # Clear button
        self.clear_btn = tk.Button(button_frame, text="Clear", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT)

        # Label to show prediction
        self.result_label = tk.Label(root, text="Draw a digit and click identify", font=("Helvetica", 16))
        self.result_label.pack()

        # Label to show confidence percentage
        self.confidence_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.confidence_label.pack()

        # For drawing on PIL Image (to capture canvas drawing)
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), 'white')
        self.draw1 = ImageDraw.Draw(self.image1)

        self.model_predict = model_predict  # your model's predict function

    def draw(self, event):
        radius = 6
        x = max(radius, min(event.x, self.canvas_width - radius))
        y = max(radius, min(event.y, self.canvas_height - radius))

        x1, y1 = (x - radius), (y - radius)
        x2, y2 = (x + radius), (y + radius)

        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw1.ellipse([x1, y1, x2, y2], fill='black')

    def clear(self):
        self.canvas.delete("all")
        self.draw1.rectangle([0, 0, self.canvas_width, self.canvas_height], fill='white')
        self.result_label.config(text="Draw a digit and click identify")
        self.confidence_label.config(text="")

    def identify(self):
        img = self.image1.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(784, 1)

        pred, confidence = self.model_predict(img)

        if pred == "Not sure":
            self.result_label.config(text="identified: Not sure")
            
        else:
            self.result_label.config(text=f"identified: {pred}")
            self.confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")

def softmax(z):
    zexp = np.exp(z - np.max(z))
    return zexp / np.sum(zexp)

def relu(x):
    return np.maximum(0, x)

def image_recog(img, threshold=0.7):
    A = img.reshape(-1,1)
    layer_dims = [784, 128, 100, 80, 32 , 10]
    Length = len(layer_dims)
    weights = {}
    biases = {}

    for l in range(1, Length):
        weights['w' + str(l)] = np.load(f'weights07_w{l}.npy')
        biases['b' + str(l)] = np.load(f'biases02_b{l}.npy')

    for l in range(1, Length):
        w = weights['w' + str(l)]
        b = biases['b' + str(l)]
        z = np.dot(w, A) + b
        if l == Length - 1:
            A = softmax(z)
        else:
            A = relu(z)
    
    pred = np.argmax(A)
    confidence = np.max(A)

    if confidence < threshold:
        return "Not sure",confidence
    else:
        return str(pred), confidence

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root, model_predict=image_recog)
    root.mainloop()
