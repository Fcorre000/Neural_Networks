# Correa (last-name, first-name)
# 100x_xxx_xxx
# 2026_03_30
# Assignment_04

import tkinter as tk
import numpy as np
import torch
from PIL import Image, ImageTk
from vae_model import VAE

LATENT_DIM = 6
MODEL_PATH = "vae_model.pth"
IMAGE_DISPLAY_SIZE = 250
SLIDER_RANGE = 3.0


class VAEApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VAE Face Generator")

        # Load model
        self.model = VAE(latent_dim=LATENT_DIM)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        self.model.eval()

        self._build_gui()
        self._update_image(None)

    def _build_gui(self):
        # Left frame: sliders
        left_frame = tk.Frame(self.root, padx=20, pady=20)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

        tk.Label(left_frame, text="Latent Variables", font=("Arial", 14, "italic")).pack(
            anchor=tk.W, pady=(0, 10)
        )

        self.sliders = []
        for i in range(LATENT_DIM):
            slider = tk.Scale(
                left_frame,
                from_=-SLIDER_RANGE,
                to=SLIDER_RANGE,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                length=200,
                showvalue=False,
                command=self._update_image,
            )
            slider.set(0.0)
            slider.pack(pady=5)
            self.sliders.append(slider)

        # Right frame: image display
        right_frame = tk.Frame(self.root, padx=20, pady=20)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(right_frame, borderwidth=2, relief="solid")
        self.image_label.pack(expand=True)

    def _update_image(self, _event):
        z_values = [slider.get() for slider in self.sliders]
        z = torch.tensor([z_values], dtype=torch.float32)

        with torch.no_grad():
            recon = self.model.decoder(z)

        # Convert (1, 3, 64, 64) -> (64, 64, 3) numpy array in [0, 255]
        img_array = recon.squeeze(0).permute(1, 2, 0).numpy()
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_array)
        img = img.resize((IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo)


def main():
    root = tk.Tk()
    VAEApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
