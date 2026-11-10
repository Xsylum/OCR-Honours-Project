# Projection examples

Example output of the projection stage, for demos. Regenerate with:

```
python -m ProjectiveProjection.examples.make_examples
```

- `stages.jpg`: one label through the projection stage: generated label with word boxes, then a clean virtual-camera view, then the same view with a pin, a clear plastic sleeve (glare, fog, scratches), lens distortion, shadows, lighting, blur and noise, and finally the ground-truth boxes after projection. Red boxes are partly hidden by the pin; magenta boxes are washed out by sleeve glare.
- `long_names/`: three labels from the label generator's long-name run (`inputs/`). These came as images only, so the word boxes in `labels.json` are found from the ink, split at the widest gaps. `projected/` holds 4 camera views per label, all with the label in a plastic sleeve, `preview_sheet.jpg` and `projected_dataset_labels.json`.
- `sample_labels/`: one view of each of the 10 labels in `final_output_labels/`, using the boxes from `master_dataset_labels.json`. Some of those boxes are offset from the ink, which comes from the label stage, not projection.

Photos are saved as JPEG (quality 90) to keep the repo small. The pipeline itself writes PNG.
