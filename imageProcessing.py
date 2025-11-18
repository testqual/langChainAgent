from PIL import Image, ImageFilter
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
import tkinter as tk
from tkinter import filedialog


def process_image(image_path: Path, output_path: Path):
    img = Image.open(image_path)

    # Resize
    resized = img.resize((800, 600))

    # Crop (left, upper, right, lower)
    cropped = resized.crop((100, 100, 700, 500))

    # Rotate
    rotated = cropped.rotate(90)

    # Apply filter
    filtered = rotated.filter(ImageFilter.CONTOUR)

    # Convert format
    filtered.save(output_path.with_suffix(".png"), format="PNG")
    print(f"Processed image saved to {output_path.with_suffix('.png')}")


def generate_caption(image_path: Path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption: {caption}")
    return caption


def preprocess_for_ml(image_path: Path, output_path: Path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img.save(output_path)
    print(f"Preprocessed image saved to {output_path}")


def full_pipeline(image_path: Path):
    output_path = image_path.parent / f"{image_path.stem}_processed"

    process_image(image_path, output_path)
    caption = generate_caption(image_path)
    preprocess_for_ml(image_path, output_path.with_suffix(".jpg"))

    return caption


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
    )
    if file_path:
        image_path = Path(file_path)
        caption = full_pipeline(image_path)
        print(f"\nFinal Caption: {caption}")
    else:
        print("No file selected.")
