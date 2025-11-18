# pip install pillow transformers torch
from PIL import Image, ImageFilter, ImageTk
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Text, Checkbutton, Entry, LabelFrame, Canvas, Scrollbar

# Load the BLIP model once at startup (takes ~30-60 seconds the first time)
print("Loading BLIP model... This may take a moment.")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Model loaded successfully!")

# Global variable to store the selected image path
image_path = None

def generate_caption(img: Image.Image) -> str:
    raw_image = img.convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption: {caption}")
    return caption

def preprocess_for_ml(img: Image.Image, output_path: Path):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img.save(output_path)
    print(f"Preprocessed image saved to {output_path}")

def process_image(image_path: Path, output_path: Path, crop_box=None, rotate_angle: float = 0):
    img = Image.open(image_path)
    resized = img.resize((800, 600))
    current = resized

    if crop_box is not None:
        current = current.crop(crop_box)

    if rotate_angle != 0:
        current = current.rotate(rotate_angle, expand=True, resample=Image.BICUBIC)

    filtered = current.filter(ImageFilter.CONTOUR)
    png_path = output_path.with_suffix(".png")
    filtered.save(png_path, "PNG")
    print(f"Processed image saved to {png_path}")
    return current, filtered  # current = cropped/rotated (no filter), filtered = with contour

# GUI Setup
root = tk.Tk()
root.title("Interactive Image Processing & Captioning App")
root.geometry("1200x800")
root.resizable(True, True)  # Make window resizable for responsiveness

# Create a canvas for scrolling
canvas = Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Add scrollbar
scrollbar = Scrollbar(root, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the canvas to hold all widgets
main_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=main_frame, anchor="nw")

# Bind mouse wheel for scrolling
def on_mouse_wheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
root.bind("<MouseWheel>", on_mouse_wheel)  # For Windows/macOS
root.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # For Linux up
root.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))  # For Linux down

# Now pack all widgets into main_frame instead of root
tk.Label(main_frame, text="Interactive Image Processing App", font=("Helvetica", 16, "bold")).pack(pady=10)

button_select = Button(main_frame, text="Select Image", font=("Helvetica", 12), command=lambda: open_file())
button_select.pack(pady=5)

note_label = tk.Label(main_frame, text="Note: Image is resized to 800x600 for processing.\n"
                                      "Crop coordinates are relative to this resized version (width 0-800, height 0-600).\n"
                                      "Rotation: positive degrees = counter-clockwise.", fg="blue")
note_label.pack(pady=5)

# Crop Options
crop_frame = LabelFrame(main_frame, text="Crop Options (optional)")
crop_frame.pack(pady=10)
var_crop = tk.BooleanVar()
check_crop = Checkbutton(crop_frame, text="Apply Custom Crop", variable=var_crop)
check_crop.grid(row=0, column=0, columnspan=4, pady=5)

tk.Label(crop_frame, text="Left:").grid(row=1, column=0)
entry_left = Entry(crop_frame, width=10)
entry_left.grid(row=1, column=1)
tk.Label(crop_frame, text="Upper:").grid(row=1, column=2)
entry_upper = Entry(crop_frame, width=10)
entry_upper.grid(row=1, column=3)

tk.Label(crop_frame, text="Right:").grid(row=2, column=0)
entry_right = Entry(crop_frame, width=10)
entry_right.grid(row=2, column=1)
tk.Label(crop_frame, text="Lower:").grid(row=2, column=2)
entry_lower = Entry(crop_frame, width=10)
entry_lower.grid(row=2, column=3)

# Rotate Options
rotate_frame = LabelFrame(main_frame, text="Rotate Options (optional)")
rotate_frame.pack(pady=10)
var_rotate = tk.BooleanVar()
check_rotate = Checkbutton(rotate_frame, text="Apply Rotation", variable=var_rotate)
check_rotate.pack()
tk.Label(rotate_frame, text="Angle (degrees):").pack()
entry_angle = Entry(rotate_frame, width=10)
entry_angle.pack()

# Process Button (moved higher, after options for better visibility)
button_process = Button(main_frame, text="Process Image & Generate Caption", font=("Helvetica", 12), state=tk.DISABLED)
button_process.pack(pady=15)

# Preview section
preview_frame = tk.Frame(main_frame)
preview_frame.pack(pady=10)

original_frame = LabelFrame(preview_frame, text="Original Image Preview")
original_frame.grid(row=0, column=0, padx=10)
original_label = Label(original_frame)
original_label.pack()

processed_png_frame = LabelFrame(preview_frame, text="Processed PNG Preview (with Contour Filter)")
processed_png_frame.grid(row=0, column=1, padx=10)
processed_png_label = Label(processed_png_frame)
processed_png_label.pack()

processed_jpg_frame = LabelFrame(preview_frame, text="Processed JPG Preview (for ML, 224x224)")
processed_jpg_frame.grid(row=0, column=2, padx=10)
processed_jpg_label = Label(processed_jpg_frame)
processed_jpg_label.pack()

# Results
result_frame = LabelFrame(main_frame, text="Results & Saved Files")
result_frame.pack(pady=10, fill=tk.BOTH, expand=True)
result_text = Text(result_frame, height=8, wrap=tk.WORD)
result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Update the scroll region after all widgets are added
def update_scroll_region(event=None):
    main_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

main_frame.bind("<Configure>", update_scroll_region)
update_scroll_region()  # Initial call

# Toggle functions for enabling/disabling inputs
def toggle_crop():
    state = tk.NORMAL if var_crop.get() else tk.DISABLED
    entry_left.config(state=state)
    entry_upper.config(state=state)
    entry_right.config(state=state)
    entry_lower.config(state=state)
    if state == tk.NORMAL and entry_left.get() == "":
        entry_left.insert(0, "100")
        entry_upper.insert(0, "100")
        entry_right.insert(0, "700")
        entry_lower.insert(0, "500")

def toggle_rotate():
    state = tk.NORMAL if var_rotate.get() else tk.DISABLED
    entry_angle.config(state=state)
    if state == tk.NORMAL and entry_angle.get() == "":
        entry_angle.insert(0, "90")

check_crop.config(command=toggle_crop)
check_rotate.config(command=toggle_rotate)
toggle_crop()      # Initial disable
toggle_rotate()    # Initial disable

def open_file():
    global image_path
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
    )
    if file_path:
        image_path = Path(file_path)
        
        # Reset options
        var_crop.set(False)
        var_rotate.set(False)
        toggle_crop()
        toggle_rotate()
        entry_left.delete(0, tk.END)
        entry_upper.delete(0, tk.END)
        entry_right.delete(0, tk.END)
        entry_lower.delete(0, tk.END)
        entry_angle.delete(0, tk.END)
        
        # Clear previous results
        processed_png_label.config(image="")
        processed_jpg_label.config(image="")
        result_text.delete(1.0, tk.END)
        
        # Show original preview
        img = Image.open(image_path)
        thumb = img.resize((400, 300))
        photo = ImageTk.PhotoImage(thumb)
        original_label.config(image=photo)
        original_label.image = photo
        
        button_process.config(state=tk.NORMAL, command=process_and_caption)
        result_text.insert(tk.END, f"Image loaded: {image_path.name}\n"
                                   "Adjust crop/rotate if needed, then click 'Process Image & Generate Caption'.")

def process_and_caption():
    if not image_path:
        return
    
    try:
        # Get crop
        crop_box = None
        if var_crop.get():
            left = int(entry_left.get() or 0)
            upper = int(entry_upper.get() or 0)
            right = int(entry_right.get() or 800)
            lower = int(entry_lower.get() or 600)
            if left < 0 or upper < 0 or right > 800 or lower > 600 or left >= right or upper >= lower:
                raise ValueError("Invalid crop values!\nUse: 0 ≤ left < right ≤ 800 and 0 ≤ upper < lower ≤ 600")
            crop_box = (left, upper, right, lower)
        
        # Get rotation
        rotate_angle = 0
        if var_rotate.get():
            angle_str = entry_angle.get().strip()
            if angle_str:
                rotate_angle = float(angle_str)
        
        output_path = image_path.parent / f"{image_path.stem}_processed"
        
        # Process image
        processed_img, filtered_img = process_image(image_path, output_path, crop_box, rotate_angle)
        
        # Generate caption on cropped/rotated version (before contour filter)
        result_text.insert(tk.END, "\n\nGenerating caption (this may take 5-15 seconds)...")
        root.update()
        caption = generate_caption(processed_img)
        
        # Preprocess for ML
        ml_path = output_path.with_suffix(".jpg")
        preprocess_for_ml(processed_img, ml_path)
        
        # Show processed PNG preview
        filtered_thumb = filtered_img.resize((400, 300))
        photo_png = ImageTk.PhotoImage(filtered_thumb)
        processed_png_label.config(image=photo_png)
        processed_png_label.image = photo_png
        
        # Show processed JPG preview
        jpg_img = Image.open(ml_path)
        jpg_thumb = jpg_img.resize((400, 300))  # Resize for preview even though original is 224x224
        photo_jpg = ImageTk.PhotoImage(jpg_thumb)
        processed_jpg_label.config(image=photo_jpg)
        processed_jpg_label.image = photo_jpg
        
        # Show final results
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Final Caption: {caption}\n\n"
                                   f"Saved files (in same folder as original):\n"
                                   f"• Contour filtered PNG: {output_path.with_suffix('.png')}\n"
                                   f"• Preprocessed for ML (224x224 JPG): {ml_path}\n\n"
                                   f"Caption was generated on your cropped/rotated version (without the contour filter).")
        
    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"Processing failed:\n{str(e)}")

root.mainloop()
