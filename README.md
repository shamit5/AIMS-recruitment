Scene Localization in Dense Images via YOLO + CLIP

This repository contains a YOLO + CLIP-based pipeline for localizing specific interactions in dense images using natural language queries. Given an image with multiple activities, the system returns the most relevant cropped region corresponding to a text description.

scene-localization/
├── Untitled2_with_finetune.ipynb   # Main Colab notebook
├── requirements.txt                 # Python dependencies
├── voc_dark_data/                   # Custom YOLO dataset
└── weights/
    └── best_yolo.pt                 # Fine-tuned YOLO weights
⚙️ Features

YOLOv8 for fast object detection (fine-tunable on custom dataset).

CLIP (ViT-B/32) for ranking detected regions against a text query.

Supports custom training of YOLO on your annotated dataset.

Outputs:

Best matching crop (image + numpy array)

Original image annotated with bounding box

Optional top-K ranked crops

🛠 Setup Instructions

Open the notebook in Colab:

Open in Colab

Install dependencies:

!pip install -q ultralytics ftfy regex tqdm
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q pillow matplotlib opencv-python


Restart the runtime if prompted.

🚀 How to Use
1. Use Pretrained or Fine-tuned YOLO Weights

Pretrained weights: yolov8n.pt (downloaded automatically by Ultralytics).

Fine-tuned weights: weights/best_yolo.pt (included in this repo).

Load the model in the notebook:

from ultralytics import YOLO
yolo = YOLO("weights/best_yolo.pt")  # Fine-tuned YOLO

2. Run Inference

Upload a dense image:

from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]


Enter a text query describing the target interaction:

query_text = "a person selling vegetables"


The notebook will:

Detect candidate boxes with YOLO.

Crop each box and compute similarity with the query using CLIP.

Return the best matching crop, save it, and display annotated original image.

Optionally show top-K crops ranked by similarity.

📝 Output

output/best_crop.jpg → best matching crop.

output/best_crop.npy → numpy array of the crop.

output/crop_1.jpg, crop_2.jpg, ... → top-K ranked crops.

Annotated original image shows bounding box with similarity score.

📚 Fine-tuning YOLO on Your Custom Dataset

If you want to improve detection on your specific scenes:

Prepare dataset in YOLO format:

images/
    train/
    val/
labels/
    train/
    val/


Each image must have a corresponding .txt file with YOLO annotations.

Class names in a YAML or classes.txt file.

Create YAML config:

train: /path/to/train/images
val: /path/to/val/images
names:
  0: Person
  1: Dog
  2: Bicycle
  ...


Train YOLO (in Colab notebook):

from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # or yolov8s.pt for a bigger model

model.train(
    data="/path/to/dataset.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    project="runs/detect",
    name="custom_yolo_model"
)


Save weights:

model.save("weights/best_yolo.pt")


Use fine-tuned weights in inference as shown above.

🔹 Fine-tuning improves detection accuracy for your specific queries (e.g., “person selling vegetables”).

📝 Key Dependencies

Ultralytics YOLOv8

OpenAI CLIP

Python libraries: torch, opencv-python, Pillow, matplotlib, tqdm, ftfy, regex

💡 Notes

GPU runtime in Colab is recommended.

Adjust YOLO parameters (epochs, batch, imgsz) depending on dataset size and GPU memory.

CLIP is used as a frozen model for ranking; no fine-tuning applied.

Fine-tuning YOLO on your dataset improves detection and overall pipeline performance.

🙏 Acknowledgements

AIMS 2K28 Recruitments for the opportunity to work on this project.

Open-source projects: YOLOv8 and CLIP.

Google Colab for free GPU resources.
