# ComfyUI DeepGen API Extension

A powerful suite of custom nodes for ComfyUI that integrates seamlessly with the **DeepGen API**. This extension allows you to leverage state-of-the-art AI models for high-quality image generation, video creation, large language models (LLMs), vision-language models (VLMs), and more, all within your ComfyUI workflows.

## Features

- 🎨 **Image Generation**: High-quality image generation using models like Flux Schnell and other state-of-the-art architectures.
- 🎬 **Video Generation**: Create stunning videos directly from text or image prompts.
- 🧠 **LLM & VLM**: Integrated nodes for advanced text processing and image understanding.
- 🛠 **Training & Upscaling**: Dedicated nodes for training custom models and high-resolution upscaling.
- ⚙️ **Dynamic Configuration**: Easily switch between models and endpoints.

---

## Installation Guide

Follow these steps to set up ComfyUI and the DeepGen extension.

### 1. Install ComfyUI

If you haven't installed ComfyUI yet:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    ```
2.  **Install Dependencies**:
    Follow the official installation guide for your operating system (Windows, macOS, or Linux) found on the [ComfyUI GitHub Page](https://github.com/comfyanonymous/ComfyUI).

### 2. Install ComfyUI Manager

ComfyUI Manager is a highly recommended extension for managing custom nodes.

1.  Navigate to your `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone the ComfyUI Manager repository:
    ```bash
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    ```
3.  **Restart ComfyUI**. You will now see a **Manager** button in the ComfyUI interface.

### 3. Install DeepGen Extension

#### Option A: Using ComfyUI Manager (Recommended)

1.  Open ComfyUI in your browser.
2.  Click the **Manager** button.
3.  Click **Custom Nodes Manager**.
4.  Search for `ComfyUI-DeepGen-API`.
5.  Click **Install**.
6.  Restart ComfyUI when prompted.

#### Option B: Manual Installation

1.  Navigate to your `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/deepiksdev/ComfyUI-DeepGen-API.git
    ```
3.  Install the required Python dependencies:
    ```bash
    cd ComfyUI-DeepGen-API
    pip install -r requirements.txt
    ```
4.  **Restart ComfyUI**.

---

## Configuration

To use the DeepGen nodes, you need a DeepGen API key.

1.  **Get an API Key**: Sign up at [DeepGen.app](https://deepgen.app) to obtain your key.
2.  **Configure in ComfyUI**:
    - Open ComfyUI and click the **Settings** (gear icon).
    - Locate the **DeepGen API Key** field and enter your key.
    - Your configuration will be saved automatically to `ComfyUI/user/deepgen/config.json`.

Alternatively, you can manually create the configuration file:
```json
{
    "DEEPGEN_API_KEY": "your_api_key_here",
    "DEEPGEN_API_URL": "https://api.deepgen.app"
}
```

---

## Usage

Once installed, you can find the DeepGen nodes under the **DeepGen** category:

- **DeepGen/Image**: Nodes for image generation.
- **DeepGen/Video**: Nodes for video generation.
- **DeepGen/LLM**: Text and Vision-Language models.
- **DeepGen/Utils**: Helper nodes for display and processing.

## Support

For issues, feature requests, or contributions, please visit our [GitHub repository](https://github.com/deepiksdev/ComfyUI-DeepGen-API).
