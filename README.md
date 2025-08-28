# 🦷 ToothXpert

**ToothXpert** is a multimodal model for tooth analysis and segmentation for OPGs.

## ✨ Features

- **🎯 Multimodal Understanding**: Combines vision and language for comprehensive OPG analysis
- **🧠 Mixture of Experts**: Utilizes Guided Mixture of LoRA Experts for efficient and scalable model adaptation
- **🔬 Segmentation**: Advanced tooth segmentation capabilities for dental imaging
- **🌐 Interactive Interface**: Gradio-based web interface for easy interaction

## 🚀 Installation

1. **📥 Clone the repository:**
```bash
git clone https://github.com/CUHK-AIM-Group/ToothXpert.git
cd ToothXpert
```

2. **📦 Install dependencies:**
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

3. **⬇️ Download SAM checkpoint:**
   Download SAM from [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

## 🤗 Pre-trained Model

A trained ToothXpert model is available on Hugging Face:

**📥 Download Pre-trained Model:**
```bash
pip install huggingface_hub
huggingface-cli download jeffrey423/ToothXpert --local-dir ./ToothXpert_pretrained
```

**🌐 Quick Start with Web Interface:**
```bash
python app.py --version ./ToothXpert_pretrained
```

## 💻 Usage

### 📈 Training
```bash
bash train.sh
```

### 🔮 Inference

#### Option 1: Use Pre-trained Model (Recommended)
```bash
python inference_toothxpert.py --version ./ToothXpert_pretrained
```

#### Option 2: Train Your Own Model

Before running inference, you need to convert and merge the trained model. Follow these steps:

#### Step 1: Convert Checkpoint to FP32 (if needed)
If your trained checkpoint is in a different precision format, convert it to FP32:
```bash
# Note: zero_to_fp32.py may need to be obtained from your training framework
python zero_to_fp32.py /path/to/trained/checkpoint.bin /path/to/output/fp32_checkpoint.bin
```

#### Step 2: Merge LoRA Weights and Save HF Model
Merge the LoRA weights with the base model and save in HuggingFace format:
```bash
python merge_lora_weights_and_save_hf_model.py \
    --weight "/path/to/your/trained/model.bin" \
    --vision_pretrained "PATH_TO_SAM_ViT-H/sam_vit_h_4b8939.pth" \
```

#### Step 3: Run Inference
Once you have the merged HF model, run inference:
```bash
python inference_toothxpert.py \
    --version "/path/to/saved/hf/model" \
    --question_file "/path/to/questions.json" \
    --image_path "/path/to/images" \
    --vision_pretrained "PATH_TO_SAM_ViT-H/sam_vit_h_4b8939.pth"
```

#### ⚠️ Important Notes:
- **Memory Requirements**: Ensure you have sufficient GPU memory (at least 16GB recommended)
- **SAM Checkpoint**: Download the SAM ViT-H checkpoint and update the path accordingly
- **Output Directory**: The merged model will be saved in `{weight_path}_hf` directory


## 🏗️ Key Components

- **`model/ToothXpert_MOE.py`**: 🦷 The ToothXpert model implementation
- **`mypeft/`**: 🔧 Custom PEFT implementation with G-MoLE
- **`utils/`**: 📊 Data processing and evaluation utilities

## �📄 License

This project is licensed under the Apache License.
