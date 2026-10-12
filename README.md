# OCR-Honours-Project

Code and documentation for our Honour's Project, created for uOttawa CSI 4900. Working title "Automating sample and label digitization using OCR". Intended to suplement a report for tools to digitize labels found within various beetle/insect samples. Labels may occasionally be obscured, written in various forms of handwritting.

## Main Pipeline Description

For easy testing of the OCR model capable of intepreting text on partially-obscured labels, we have decided to construct a pipeline to generate a variety of labels using a custom pipeline.

Since the beetle sample photographs will likely be partially obscured by either the beetle or due to poor-photography, the generated samples will need to include labels with obscured text.

Our sample generation pipeline will have the following steps:
1) Take in a list of beetle names, possibly information on these beetles to include on labels
2) Use a text to handwritten conversion model to generate artificial labels for a variety of beetle speecies.
3) Use Projective Projection on the generate handwritten text to simulate photographs being taken from different directions. Some labels may be run through this multiple times to capture cases where some text may be obscured from one angle but not from another.
4) Introduce occlusions to the image (either portions of a beetle or with obscuring bars) to ensure the OCR model is trained to interpret partially obscured text

[TO RESEARCH]: DiffusionPen https://github.com/koninik/DiffusionPen https://arxiv.org/abs/2409.06065

## Setting up .venv for DiffusionPen

```
python -m venv .venv
source .venv/Scripts/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate einops pillow scikit-image opencv-python numpy timm tqdm wandb
```

## References

Experimenting with the DiffusionPen model for generating handwritten text from images.

@article{nikolaidou2024diffusionpen,
  title={DiffusionPen: Towards Controlling the Style of Handwritten Text Generation},
  author={Nikolaidou, Konstantina and Retsinas, George and Sfikas, Giorgos and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2409.06065},
  year={2024}
}

See "AI_Attributions.md" for more in-depth conversation histories