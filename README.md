# FoodSeg103 SegFormer-B5

SegFormer-B5 fine-tuned for semantic segmentation of food images on [FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103) (103 food classes + background = 104 labels).

## Results (full validation set, 2135 images)

| Metric | Value |
|---|---|
| mIoU | 0.415 |
| Pixel accuracy | 0.805 |
| Rice IoU | 0.739 |

Common classes (rice, bread, soup, large vegetables) segment well. A tail of rare or visually similar classes scores near zero IoU, for example pudding, peanut, hamburg, salad, king oyster mushroom, oyster mushroom, bamboo shoots and kelp.

## Usage

```python
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

REPO = "your-username/foodseg103-segformer-b5"   # edit this
model = SegformerForSemanticSegmentation.from_pretrained(REPO).eval()
proc = SegformerImageProcessor.from_pretrained(REPO)

img = Image.open("photo.jpg").convert("RGB")
inputs = proc(images=img, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
logits = torch.nn.functional.interpolate(
    logits, size=img.size[::-1], mode="bilinear", align_corners=False)
mask = logits.argmax(1)[0].numpy()          # class id per pixel
print({model.config.id2label[int(i)] for i in set(mask.flatten())})
```

Class names are stored in `config.json` (`id2label`); 0 is background.

## Training

- **Base model:** `nvidia/segformer-b5-finetuned-ade-640-640` (ADE20K head replaced with a new 104-class head)
- **Data:** FoodSeg103 train split (4983 images)
- **Input size:** 640x640 (squash-resize)
- **Augmentation:** random scale, crop, horizontal flip, color jitter
- **Rare-class oversampling:** repeat-factor sampling (T = 0.03), 4983 -> 6627 images per epoch
- **Schedule:** 25 epochs, learning rate 6e-5, effective batch size 16, fp16
- **Hardware:** 2x NVIDIA T4 on Kaggle (DDP), about 6 hours
- **Model selection:** best checkpoint by mIoU on 1000 validation images

## Limitations

- Several classes are visually near-identical (mushroom varieties, salad vs lettuce, meat types), so expect confusion between them.
- Trained on FoodSeg103 photo styles; performance on very different images (heavy filters, unusual angles, packaged food) is untested.
- Rare classes have few training examples and score poorly.

## Credits

- Model: [SegFormer](https://arxiv.org/abs/2105.15203) (Xie et al., 2021), via Hugging Face `transformers`
- Dataset: [FoodSeg103](https://xiongweiwu.github.io/foodseg103.html) (Wu et al., 2021)
