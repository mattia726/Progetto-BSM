"""Tkinter app to draw a digit and classify it with the Bayesian MNIST model."""

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk

from bnn_mnist import BayesianMLP, MNIST_MEAN, MNIST_STD, predict_probabilities, set_seed


DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "bnn_mnist.pt"
DEFAULT_CANVAS_SIZE = 280
DEFAULT_BRUSH_SIZE = 18
DEFAULT_TEST_SAMPLES = 25
MNIST_IMAGE_SIZE = 28
MNIST_DIGIT_BOX = 20


def infer_hidden_dim(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer the hidden layer width from the checkpoint weights."""

    first_layer_weights = state_dict.get("layer1.weight_mu")
    if first_layer_weights is None:
        raise KeyError("Checkpoint is missing 'layer1.weight_mu'.")
    return int(first_layer_weights.shape[0])


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> BayesianMLP:
    """Load a trained Bayesian MLP from a checkpoint file."""

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        hidden_dim = int(checkpoint.get("hidden_dim", infer_hidden_dim(state_dict)))
        prior_sigma = float(checkpoint.get("prior_sigma", 1.0))
    else:
        state_dict = checkpoint
        hidden_dim = infer_hidden_dim(state_dict)
        prior_sigma = 1.0

    model = BayesianMLP(hidden_dim=hidden_dim, prior_sigma=prior_sigma).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_drawing(image: Image.Image) -> tuple[Image.Image, torch.Tensor | None]:
    """Convert a user drawing into an MNIST-like 28x28 tensor."""

    pixels = np.asarray(image, dtype=np.uint8)
    non_zero = np.argwhere(pixels > 0)

    if non_zero.size == 0:
        blank_image = Image.new("L", (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), 0)
        return blank_image, None

    top_left = non_zero.min(axis=0)
    bottom_right = non_zero.max(axis=0) + 1

    top = int(top_left[0])
    left = int(top_left[1])
    bottom = int(bottom_right[0])
    right = int(bottom_right[1])

    cropped_pixels = pixels[top:bottom, left:right]
    cropped_image = Image.fromarray(cropped_pixels, mode="L")

    cropped_width, cropped_height = cropped_image.size
    square_side = max(cropped_width, cropped_height)
    padding = max(4, square_side // 4)

    square_image = Image.new("L", (square_side + 2 * padding, square_side + 2 * padding), 0)
    paste_x = (square_image.width - cropped_width) // 2
    paste_y = (square_image.height - cropped_height) // 2
    square_image.paste(cropped_image, (paste_x, paste_y))

    resized_digit = square_image.resize((MNIST_DIGIT_BOX, MNIST_DIGIT_BOX), Image.Resampling.LANCZOS)
    mnist_image = Image.new("L", (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), 0)
    mnist_offset = (MNIST_IMAGE_SIZE - MNIST_DIGIT_BOX) // 2
    mnist_image.paste(resized_digit, (mnist_offset, mnist_offset))

    tensor = torch.from_numpy(np.asarray(mnist_image, dtype=np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return mnist_image, tensor


class DigitDrawApp:
    """Small desktop interface for drawing and classifying digits."""

    def __init__(
        self,
        model: BayesianMLP,
        device: torch.device,
        checkpoint_path: Path,
        test_samples: int,
        canvas_size: int,
        brush_size: int,
    ) -> None:
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.test_samples = test_samples
        self.canvas_size = canvas_size
        self.brush_size = brush_size

        self.root = tk.Tk()
        self.root.title("Bayesian MNIST Digit Drawer")
        self.root.resizable(False, False)

        self.canvas_image = Image.new("L", (canvas_size, canvas_size), 0)
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        self.last_x: int | None = None
        self.last_y: int | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.prediction_var = tk.StringVar(value="Prediction: -")
        self.status_var = tk.StringVar(
            value=f"Checkpoint: {checkpoint_path} | Monte Carlo samples: {test_samples}"
        )
        self.probability_vars = [tk.DoubleVar(value=0.0) for _ in range(10)]
        self.probability_text_vars = [tk.StringVar(value="0.0%") for _ in range(10)]

        self._build_layout()
        self._reset_preview()

    def _build_layout(self) -> None:
        """Create the Tk widgets."""

        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nw")

        right_frame = ttk.Frame(main_frame, padding=(16, 0, 0, 0))
        right_frame.grid(row=0, column=1, sticky="ne")

        title_label = ttk.Label(left_frame, text="Draw a digit", font=("Segoe UI", 16, "bold"))
        title_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        instruction_label = ttk.Label(
            left_frame,
            text="Use the left mouse button. Click Predict to run the Bayesian network.",
            wraplength=self.canvas_size,
        )
        instruction_label.grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#666666",
            cursor="crosshair",
        )
        self.canvas.grid(row=2, column=0)
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._draw_stroke)
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)

        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        predict_button = ttk.Button(buttons_frame, text="Predict", command=self.predict_digit)
        predict_button.grid(row=0, column=0, padx=(0, 8))

        clear_button = ttk.Button(buttons_frame, text="Clear", command=self.clear_canvas)
        clear_button.grid(row=0, column=1)

        self.root.bind("<Return>", lambda _event: self.predict_digit())
        self.root.bind("<Escape>", lambda _event: self.clear_canvas())

        prediction_label = ttk.Label(right_frame, textvariable=self.prediction_var, font=("Segoe UI", 16, "bold"))
        prediction_label.grid(row=0, column=0, sticky="w")

        preview_title = ttk.Label(right_frame, text="Processed 28x28 input", font=("Segoe UI", 10, "bold"))
        preview_title.grid(row=1, column=0, sticky="w", pady=(12, 4))

        self.preview_label = ttk.Label(right_frame)
        self.preview_label.grid(row=2, column=0, sticky="w")

        probabilities_title = ttk.Label(right_frame, text="Digit probabilities", font=("Segoe UI", 10, "bold"))
        probabilities_title.grid(row=3, column=0, sticky="w", pady=(12, 4))

        probabilities_frame = ttk.Frame(right_frame)
        probabilities_frame.grid(row=4, column=0, sticky="w")

        for digit in range(10):
            digit_label = ttk.Label(probabilities_frame, text=str(digit), width=2)
            digit_label.grid(row=digit, column=0, sticky="w", pady=2)

            progress = ttk.Progressbar(
                probabilities_frame,
                maximum=100.0,
                variable=self.probability_vars[digit],
                length=180,
            )
            progress.grid(row=digit, column=1, padx=(0, 8), pady=2)

            value_label = ttk.Label(probabilities_frame, textvariable=self.probability_text_vars[digit], width=7)
            value_label.grid(row=digit, column=2, sticky="e", pady=2)

        status_label = ttk.Label(right_frame, textvariable=self.status_var, wraplength=280)
        status_label.grid(row=5, column=0, sticky="w", pady=(12, 0))

    def _draw_circle(self, x: int, y: int) -> None:
        """Draw a filled circle on both the visible canvas and the backing image."""

        radius = self.brush_size // 2
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white")
        self.canvas_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    def _start_stroke(self, event: tk.Event) -> None:
        """Start a new brush stroke."""

        self.last_x = int(event.x)
        self.last_y = int(event.y)
        self._draw_circle(self.last_x, self.last_y)

    def _draw_stroke(self, event: tk.Event) -> None:
        """Continue drawing while the left mouse button is pressed."""

        current_x = int(event.x)
        current_y = int(event.y)

        if self.last_x is None or self.last_y is None:
            self.last_x = current_x
            self.last_y = current_y

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            current_x,
            current_y,
            fill="white",
            width=self.brush_size,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.canvas_draw.line((self.last_x, self.last_y, current_x, current_y), fill=255, width=self.brush_size)
        self._draw_circle(current_x, current_y)

        self.last_x = current_x
        self.last_y = current_y

    def _end_stroke(self, _event: tk.Event) -> None:
        """Finish the current stroke."""

        self.last_x = None
        self.last_y = None

    def _reset_preview(self) -> None:
        """Show an empty 28x28 preview and clear displayed probabilities."""

        blank_image = Image.new("L", (MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE), 0)
        preview_image = blank_image.resize((MNIST_IMAGE_SIZE * 8, MNIST_IMAGE_SIZE * 8), Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(preview_image)
        self.preview_label.configure(image=self.preview_photo)

        self.prediction_var.set("Prediction: -")
        for digit in range(10):
            self.probability_vars[digit].set(0.0)
            self.probability_text_vars[digit].set("0.0%")

    def clear_canvas(self) -> None:
        """Erase the current drawing and reset the outputs."""

        self.canvas.delete("all")
        self.canvas_image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.canvas_draw = ImageDraw.Draw(self.canvas_image)
        self.last_x = None
        self.last_y = None
        self.status_var.set(
            f"Checkpoint: {self.checkpoint_path} | Monte Carlo samples: {self.test_samples}"
        )
        self._reset_preview()

    def predict_digit(self) -> None:
        """Preprocess the drawing, run the BNN, and update the GUI."""

        mnist_image, input_tensor = preprocess_drawing(self.canvas_image)
        preview_image = mnist_image.resize((MNIST_IMAGE_SIZE * 8, MNIST_IMAGE_SIZE * 8), Image.Resampling.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(preview_image)
        self.preview_label.configure(image=self.preview_photo)

        if input_tensor is None:
            self.prediction_var.set("Prediction: -")
            self.status_var.set("Draw a digit before running prediction.")
            for digit in range(10):
                self.probability_vars[digit].set(0.0)
                self.probability_text_vars[digit].set("0.0%")
            return

        with torch.no_grad():
            probabilities, _ = predict_probabilities(
                self.model,
                input_tensor.to(self.device),
                num_samples=self.test_samples,
            )

        probability_vector = probabilities.squeeze(0).cpu().numpy()
        predicted_digit = int(probability_vector.argmax())
        predicted_probability = float(probability_vector[predicted_digit])

        self.prediction_var.set(
            f"Prediction: {predicted_digit} ({predicted_probability * 100:.1f}%)"
        )
        self.status_var.set(
            f"Averaged over {self.test_samples} weight samples from {self.checkpoint_path.name}."
        )

        for digit, probability in enumerate(probability_vector):
            probability_percent = float(probability * 100.0)
            self.probability_vars[digit].set(probability_percent)
            self.probability_text_vars[digit].set(f"{probability_percent:4.1f}%")

    def run(self) -> None:
        """Start the Tk event loop."""

        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    """Parse command line options for the drawing app."""

    parser = argparse.ArgumentParser(description="Draw a digit and classify it with the Bayesian MNIST model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to a saved checkpoint produced by bnn_mnist.py.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help="Number of Bayesian weight samples to average for each prediction.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=DEFAULT_CANVAS_SIZE,
        help="Size of the square drawing canvas in pixels.",
    )
    parser.add_argument(
        "--brush-size",
        type=int,
        default=DEFAULT_BRUSH_SIZE,
        help="Brush width used while drawing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """Load the checkpoint and launch the GUI."""

    args = parse_args()
    set_seed(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Train the model first with "
            f"'python bnn_mnist.py --save-path {args.checkpoint}'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, device=device)

    app = DigitDrawApp(
        model=model,
        device=device,
        checkpoint_path=args.checkpoint,
        test_samples=args.test_samples,
        canvas_size=args.canvas_size,
        brush_size=args.brush_size,
    )
    app.run()


if __name__ == "__main__":
    main()
