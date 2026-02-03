import math
from typing import List, Optional, TypeAlias, Union

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

ImageInput: TypeAlias = Union[Image.Image, List[Image.Image]]
BatchImageInput: TypeAlias = Union[List[Image.Image], List[List[Image.Image]]]


class OpsMMEmbeddingV1(nn.Module):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: Optional[int] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.default_instruction = "You are a helpful assistant."
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        self.processor.tokenizer.padding_side = "left"
        self.eval()

    def encode_input(self, input):
        hidden_states = self.base_model(**input, return_dict=True, output_hidden_states=True)
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states)
        return pooled_output

    def _pooling(self, last_hidden_state):
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[torch.arange(batch_size), -1, :]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def _validate_instructions(
        self,
        texts: Optional[List[str]],
        images: Optional[BatchImageInput],
        instruction: Optional[Union[str, List[str]]],
    ) -> List[str]:
        """Validate and format instructions to match batch size"""
        batch_size = max(len(x) if x is not None else 0 for x in [texts, images])

        if instruction is None:
            return [self.default_instruction] * batch_size

        if isinstance(instruction, str):
            return [instruction] * batch_size

        if isinstance(instruction, list):
            if len(instruction) != batch_size:
                raise ValueError(f"Length of instruction list ({len(instruction)}) must match batch size ({batch_size}) when texts/images are provided")
            return instruction

        raise TypeError("instruction must be str, List[str] or None")

    def _process_images(self, images: ImageInput) -> List[Image.Image]:
        """Convert single image or list of images to processed format"""
        if isinstance(images, Image.Image) or isinstance(images, str):
            return [fetch_image(images)]
        return [fetch_image(i) for i in images]

    def embed(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[BatchImageInput] = None,
        instruction: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate embeddings for text, images, or combined inputs.

        Args:
            texts: List of text inputs (optional)
            images: Can be:
                - List[Image.Image]: Single image per input
                - List[List[Image.Image]]: Multiple images per input
            instruction: Instruction(s) for the model. Can be:
                - None: use default instruction
                - str: use same instruction for all inputs
                - List[str]: per-input instructions (must match batch size)
        """
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        instructions = self._validate_instructions(texts, images, instruction)

        # Determine batch size
        batch_size = len(texts) if texts is not None else len(images)  # type: ignore

        input_texts, input_images = [], []
        for i in range(batch_size):
            text = texts[i] if texts is not None else None
            image = images[i] if images is not None else None

            input_str = ""
            processed_image = None
            if image is not None:
                processed_image = self._process_images(image)
                input_str += "<|vision_start|><|image_pad|><|vision_end|>" * len(processed_image)

            if text is not None:
                input_str += text

            msg = f"<|im_start|>system\n{instructions[i]}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"

            input_texts.append(msg)
            input_images.append(processed_image)

        # Only pass to processor if we actually have images
        processed_images = input_images if any(img is not None for img in input_images) else None

        inputs = self.processor(
            text=input_texts,
            images=processed_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            embeddings = self.encode_input(inputs)

        return embeddings

    def get_text_embeddings(
        self,
        texts: List[str],
        instruction: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Convenience method for text-only embeddings"""
        return self.get_fused_embeddings(texts=texts, instruction=instruction, **kwargs)

    def get_image_embeddings(
        self,
        images: BatchImageInput,
        instruction: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Convenience method for image-only embeddings.

        Args:
            images: Can be:
                - List[Image.Image]: Single image per input
                - List[List[Image.Image]]: Multiple images per input
        """
        return self.get_fused_embeddings(images=images, instruction=instruction, **kwargs)

    def get_fused_embeddings(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[BatchImageInput] = None,
        instruction: Optional[Union[str, List[str]]] = None,
        batch_size: int = 8,
        show_progress: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Batch processing for large collections of texts/images.

        Args:
            texts: List of text inputs (optional)
            images: Can be:
                - List[Image.Image]: Single image per input
                - List[List[Image.Image]]: Multiple images per input
            instruction: Instruction(s) for the model
            batch_size: Number of items to process at once
            show_progress: Whether to display progress bar
        """

        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        total_items = len(texts) if texts is not None else len(images)  # type: ignore
        num_batches = math.ceil(total_items / batch_size)

        all_embeddings = []
        progress = tqdm(total=num_batches, disable=not show_progress, desc="Processing")

        for i in range(0, total_items, batch_size):
            batch_texts = texts[i : i + batch_size] if texts is not None else None
            batch_images = images[i : i + batch_size] if images is not None else None
            batch_emb = self.embed(texts=batch_texts, images=batch_images, instruction=instruction)

            all_embeddings.append(batch_emb.cpu())
            progress.update(1)

        progress.close()
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def forward(self, **inputs) -> torch.Tensor:
        """Alias for encode_input"""
        return self.encode_input(inputs)


### Modified from qwen_vl_utils.vision_process.py
import base64
import logging
import math
from io import BytesIO

import requests

IMAGE_FACTOR = 28
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int | float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int | float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if max(h_bar, w_bar) / min(h_bar, w_bar) > MAX_RATIO:
        logging.warning(f"Absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(h_bar, w_bar) / min(h_bar, w_bar)}")
        if h_bar > w_bar:
            h_bar = w_bar * MAX_RATIO
        else:
            w_bar = h_bar * MAX_RATIO
    return h_bar, w_bar


def fetch_image(
    image: str | Image.Image,
    size_factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Image.Image:
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)  # type: ignore
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    return image


###
