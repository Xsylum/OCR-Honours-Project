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

## Setting up for DiffusionPen (using the IAM handwritting dataset)

[TODO] cleanup this section, the following should be enough to setup and run?

```
pip install -r requirements.txt
python DiffusionPen/download_models.py   # pretrained weights from Hugging Face (~1.4 GB)
python pipeline.py                       # also auto-downloads the Catalogue of Life data (~900 MB)
```

[TODO] Note that "iam_data/ is derived from the IAM Handwriting Database (Marti & Bunke, 2002) and carries its non-commercial terms."

1) Create .venv
```
python -m venv .venv
source .venv/Scripts/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate einops pillow scikit-image opencv-python numpy timm tqdm wandb huggingface_hub omegaconf
```

2) Get the IAM word images and metadata
[TODO]

3) Download model weights

```
hf download konnik/DiffusionPen --include "style_models/*" "diffusionpen_iam_model_path/*" --local-dir .
```

4) Fix the split-file path
```
cp -r utils/aachen_iam_split ./aachen_iam_split
mkdir -p image_samples
```

5) Make a copy of the script so if the DiffusionPen subtree updates there aren't any issues
```
cp train.py sample_names.py
```

and in this edit sample_names.py, after "--sampling_mode" argument, add:
```
  parser.add_argument('--text', type=str, default='text word')
  parser.add_argument('--style_id', type=int, default=12)
```
and in the "single_sampling" branch, replace the "x_test =" line, the "for" line and the "s = random.randint(...)" line with:
```
  for x_text in args.text.split():
      print('Word:', x_text)
      s = args.style_id
```
and in the "paragarph" branch set 
```
lines = args.text
s = args.style_id
```
and within that same branch remove the "clip_model=None" parameter from the "diffusion.sampling()" call.

[TODO] properly include this command:
fleur@XandersLaptop MINGW64 /c/_University/Y4S2/Honours/OCR-Honours-Project/DiffusionPen (main)
python sample_names.py --train_mode sampling --sampling_mode single_sampling --save_path ./diffusionpen_iam_model_path --style_path ./style_models/iam_style_diffusionpen.pth --stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5 --text "Pinacotarsus proximus" --style_id 25

## Catalogue of Life
using this to source beetle names

https://www.catalogueoflife.org/data/download

## References

Experimenting with the DiffusionPen model for generating handwritten text from images.

@article{nikolaidou2024diffusionpen,
  title={DiffusionPen: Towards Controlling the Style of Handwritten Text Generation},
  author={Nikolaidou, Konstantina and Retsinas, George and Sfikas, Giorgos and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2409.06065},
  year={2024}
}

Using the IAM Handwriting Database from (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) [NOTE: Bern university site, shows as unsecure]

IAM Handwriting Database asks we cite the following paper:
U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002.

See "AI_Attributions.md" for more in-depth conversation histories

DiffusionPen/extract_style_images.py is claude generated code, pretty sure from DiffusionPen github, but make sure to verify it.