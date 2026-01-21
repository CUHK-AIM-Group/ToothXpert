# 🦷 ToothXpert

**ToothXpert** is a multimodal AI model for comprehensive dental X-ray (OPG) analysis, combining vision and language understanding for automatic diagnosis and condition detection.

## ✨ Key Features

- **🎯 Multimodal Understanding**: Analyzes dental X-rays and generates detailed clinical descriptions
- **🔍 Multi-Condition Detection**: Detects 11 different dental conditions automatically
- **🧠 Guided Mixture of LoRA Experts**: Efficient model architecture for scalable adaptation
- **🔬 Segmentation Capabilities**: Advanced tooth segmentation using SAM integration
- **⚡ Easy-to-Use Interface**: Simple command-line tools for quick inference

---

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🔧 Requirements

- **GPU**: NVIDIA GPU with at least 16GB VRAM (tested on L40)
- **Python**: 3.11 (recommended)
- **CUDA**: 12.1 or compatible
- **Storage**: ~20GB for model and dependencies

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/CUHK-AIM-Group/ToothXpert.git
cd ToothXpert
```

### Step 2: Set Up Environment

We recommend using conda:

```bash
# Create conda environment
conda create -n toothxpert python=3.11
conda activate toothxpert

# Install PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install additional required packages
pip install medpy
```

### Step 3: Download Pre-trained Model

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download model (~15GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jeffrey423/ToothXpert', local_dir='./ToothXpert_pretrained')"
```

### Step 4: Download SAM Checkpoint

```bash
# Download SAM ViT-H checkpoint (~2.3GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Step 5: Download Data (Optional)

```bash
# Download sample test images and annotations
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jeffrey423/ToothXpert.MM-OPG-Annotations', repo_type='dataset', local_dir='./mm-opg')"
```

---

## ⚡ Quick Start

### Test Single Image (Recommended)

Run inference on a dental X-ray image with our simplified script:

```bash
# Use demo image (included in repo)
python test_comprehensive.py

# Or specify your own image
python test_comprehensive.py --image_path /path/to/your/xray.png

# Specify custom model path
python test_comprehensive.py \
    --model_path /path/to/model \
    --image_path /path/to/xray.png
```

**What it does:**
- Generates a clinical summary of the X-ray
- Detects 11 dental conditions:
  1. Amalgam restorations
  2. Caries (R/L)
  3. Crestal bone loss (mandible)
  4. Crestal bone loss (maxillary)
  5. Implant-supported bridge
  6. Dental implant
  7. Metallic/non-metallic post
  8. Non-metallic restorations
  9. Periapical radiolucency
  10. Root canal treated teeth
  11. Tooth-supported bridge

**Example output:**
```
================================================================================
ToothXpert Single Image Inference
Summary + 11 Dental Condition Questions
================================================================================

✓ Image: /path/to/image.png
  Testing 12 questions (1 summary + 11 conditions)

Loading tokenizer...
✓ Tokenizer loaded

Loading model (this takes 2-3 minutes)...
✓ Model loaded and ready on cuda:0

Loading image...
✓ Image loaded: (320, 640, 3)

Running 12 inferences (1 summary + 11 conditions)...

[1/12] SUMMARY
Q: Can you describe the image for me?
A: This is a dental x-ray image (OPG). Several symptoms are observed...

[2/12] Condition 1
Q: Is there any amalgam restorations in the image?
A: No, there is no amalgam restorations.

...
```

### Test Tooth Segmentation

Test the tooth segmentation capabilities:

```bash
# Use demo image (included in repo)
python test_segmentation.py

# Or specify your own image
python test_segmentation.py --image_path /path/to/your/xray.png

# Specify custom output directory
python test_segmentation.py \
    --image_path /path/to/xray.png \
    --output_dir ./my_seg_results
```

**What it does:**
- Tests tooth segmentation
- Tries multiple segmentation prompts
- Generates visualization with segmentation masks overlaid in red
- Saves results to `./segmentation_output/` (or custom directory)

**Example output:**
```
================================================================================
ToothXpert Segmentation Test
SAM-based Tooth Segmentation
================================================================================

✓ Image: /path/to/image.png
  Testing 3 segmentation prompts

Running segmentation tests...

[1/3] Segmentation Test
Q: Can you segment all the teeth in this image?
A: Sure, it is [SEG].
   Masks generated: 1
   Saved: ./segmentation_output/seg_test_1.png

```

### Advanced Inference

For batch processing or custom questions:

```bash
python inference_toothxpert.py \
    --version ./ToothXpert_pretrained \
    --question_file /path/to/questions.json \
    --image_path /path/to/images \
    --precision bf16
```

### Web Interface (Gradio)

```bash
python app.py --version ./ToothXpert
```

Then open your browser to the displayed URL (typically `http://localhost:7860`).

---

## 💻 Usage

### Command-Line Arguments

#### `test_comprehensive.py` (Simplified Single Image Inference)

```bash
python test_comprehensive.py [OPTIONS]

Options:
  --model_path PATH     Path to model directory (default: ./ToothXpert)
  --image_path PATH     Path to dental X-ray image (default: ./demo/example_image_1.png)
  --device DEVICE       Device to use (default: cuda:0)
```

#### `test_segmentation.py` (Tooth Segmentation Test)

```bash
python test_segmentation.py [OPTIONS]

Options:
  --model_path PATH     Path to model directory (default: ./ToothXpert)
  --image_path PATH     Path to dental X-ray image (default: ./demo/example_image_2.png)
  --device DEVICE       Device to use (default: cuda:0)
  --output_dir PATH     Output directory for results (default: ./segmentation_output)
```

---

## 🏗️ Model Details

### Architecture

- **Base Model**: LLaVA-1.5-7B with medical alignment
- **Vision Encoder**: CLIP ViT-L/14
- **Segmentation**: SAM (Segment Anything Model) ViT-H
- **Adaptation**: Guided Mixture of LoRA Experts (G-MoLE)
- **Precision**: bfloat16 (recommended), fp16, fp32

### Model Components

```
ToothXpert/
├── model/
│   ├── ToothXpert_MOE.py         # Main model architecture
│   ├── llava/                    # LLaVA implementation
│   └── segment_anything/         # SAM integration
├── mypeft/                       # Custom PEFT with G-MoLE
├── utils/                        # Data processing utilities
├── demo/                         # Example dental X-ray images
│   ├── example_image_1.png       # Demo OPG image 1
│   └── example_image_2.png       # Demo OPG image 2
├── test_comprehensive.py         # Simple inference script ⭐
├── test_segmentation.py          # Segmentation test script ⭐
├── inference_toothxpert.py       # Full inference pipeline
├── app.py                        # Gradio web interface
└── train_lora_base.py           # Training script
```

### Performance

- **GPU Memory**: ~14GB during inference (bf16)
- **Inference Speed**: ~5-10 seconds per question (L40 GPU)
- **Supported Image Formats**: PNG, JPG, JPEG

---

## 📊 Citation

If you use ToothXpert in your research, please cite:

```bibtex
@article{liu2026toothxpert,
  title={Developing and Evaluating Multimodal Large Language Model for Orthopantomography Analysis to Support Clinical Dentistry},
  author={Liu, Xinyu and Hung, Kuo Feng and Yu, Weihao and Ng, Ray Anthony W T and Li, Wuyang and Niu, Tianye and Chen, Hui and Yuan, Yixuan},
  journal={Cell Reports Medicine},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/CUHK-AIM-Group/ToothXpert/issues)
- Contact the authors (xinyuliu@link.cuhk.edu.hk)

---

## 🙏 Acknowledgments

- LLaVA for the base multimodal architecture
- LISA for the SAM integration model
- HuggingFace for model hosting and tools
