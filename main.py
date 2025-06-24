
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, binary_erosion

class ImageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Aplikasi Pengolahan Citra Digital")

        self.original_image = None
        self.processed_image = None

        tk.Label(master, text="Aplikasi Pengolahan Citra Digital", font=("Arial", 16, "bold")).pack()

        frame = tk.Frame(master)
        frame.pack()

        # Add labels for "Gambar Asli" and "Gambar Proses"
        tk.Label(frame, text="Gambar Asli").grid(row=0, column=0)
        tk.Label(frame, text="Gambar Proses").grid(row=0, column=1)

        self.image_label = tk.Label(frame)
        self.image_label.grid(row=1, column=0)
        self.result_label = tk.Label(frame)
        self.result_label.grid(row=1, column=1)

        control_frame = tk.Frame(master)
        control_frame.pack()

        tk.Button(control_frame, text="Input Gambar", command=self.load_image).grid(row=0, column=0)
        tk.Button(control_frame, text="Grayscale", command=self.to_grayscale).grid(row=0, column=1)
        tk.Button(control_frame, text="Biner", command=self.to_binary).grid(row=0, column=2)

        tk.Button(control_frame, text="Penjumlahan +", command=self.addition).grid(row=1, column=0)
        tk.Button(control_frame, text="OR", command=self.logical_or).grid(row=1, column=1)
        tk.Button(control_frame, text="Histogram Gabungan", command=self.show_histogram).grid(row=1, column=2)

        tk.Button(control_frame, text="Konvolusi Blur", command=self.blur_image).grid(row=2, column=0)
        tk.Button(control_frame, text="Erosi (SE1)", command=lambda: self.erosion('rect')).grid(row=2, column=1)
        tk.Button(control_frame, text="Erosi (SE2)", command=lambda: self.erosion('ellipse')).grid(row=2, column=2)

        tk.Button(control_frame, text="Simpan Gambar", command=self.save_image).grid(row=3, column=1)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.original_image = Image.open(path).convert("RGB")
            self.display_image(self.original_image, self.image_label)

    def display_image(self, img, label):
        img_resized = img.resize((256, 256))
        imgtk = ImageTk.PhotoImage(img_resized)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def to_grayscale(self):
        if self.original_image:
            gray = self.original_image.convert("L")
            self.processed_image = gray
            self.display_image(gray, self.result_label)

    def to_binary(self):
        if self.original_image:
            gray = self.original_image.convert("L")
            binary = gray.point(lambda p: 255 if p > 127 else 0)
            self.processed_image = binary
            self.display_image(binary, self.result_label)

    def addition(self):
        if self.original_image:
            arr = np.array(self.original_image)
            result = np.clip(arr + 50, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(result)
            self.display_image(self.processed_image, self.result_label)

    def logical_or(self):
        if self.original_image:
            arr = np.array(self.original_image)
            result = np.bitwise_or(arr, 128)
            self.processed_image = Image.fromarray(result)
            self.display_image(self.processed_image, self.result_label)

    def show_histogram(self):
        if self.original_image:
            arr = np.array(self.original_image)
            colors = ('r', 'g', 'b')
            for i, col in enumerate(colors):
                plt.hist(arr[:, :, i].ravel(), bins=256, color=col, alpha=0.5)
            plt.title("Histogram Gabungan RGB")
            plt.show()

    def blur_image(self):
        if self.original_image:
            kernel = np.ones((5, 5)) / 25
            arr = np.array(self.original_image.convert("L"))
            result = convolve(arr, kernel)
            result = np.clip(result, 0, 255).astype(np.uint8)
            self.processed_image = Image.fromarray(result)
            self.display_image(self.processed_image, self.result_label)

    def erosion(self, shape):
        if self.original_image:
            binary = np.array(self.original_image.convert("L").point(lambda p: p > 127 and 1))
            if shape == 'rect':
                se = np.ones((3, 3))
            else:  # ellipse
                se = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
            result = binary_erosion(binary, structure=se).astype(np.uint8) * 255
            self.processed_image = Image.fromarray(result)
            self.display_image(self.processed_image, self.result_label)

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                self.processed_image.save(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
