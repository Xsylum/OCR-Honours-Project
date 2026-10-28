#pipeline.py
import json
import os

# Import functions from each file
from CatalogueOfLife.extract_beetles import get_names_list
from DiffusionPen.syllable_splitter import split_name_into_syllables
from DiffusionPen.label_generator import generate_handwritten_syllables, stitch_handwritten_words_to_solid

def run_dataset_pipeline():
	print("Starting dataset generation pipeline")

	raw_names = get_names_list()
	print(f"Loaded {len(raw_names)} names.")

	# Step 2 & 3: Process each name through splitting and image generation
	output_labels_dir = "./final_output_labels"
	os.makedirs(output_labels_dir, exist_ok=True)

	all_dataset_annotations = []

	for index, name in enumerate(raw_names):
		print(f"\n--- Processing: [{index+1}/{len(raw_names)}]: {name} ---")

		# Pass string into the syllable splitter
		split_data = split_name_into_syllables(name)
		
		try:
			syllable_image_paths = generate_handwritten_syllables(split_data)
		except Exception as e:
			print(f"Skipping '{name}' due to generation error: {e}")
			continue

		if not syllable_image_paths:
			print(f"Skipping '{name}' because no syllable images were generated.")
			continue
        
        # 3. Prepare the final destination path for the fully stitched image
		image_name = f"label_{index}.png"
		final_image_path = os.path.join(output_labels_dir, image_name, name)
        
        # 4. Pass the generated image paths to your OpenCV tool to stitch them together
        # Your stitch function returns the bounding boxes dictionary list
		ocr_boxes = stitch_handwritten_words_to_solid(syllable_image_paths, final_image_path)
		
		# Compile master annotations mapping image files to text coordinates
		all_dataset_annotations.append({
			"image_file": image_name,
			"ground_truth": name,
			"words": ocr_boxes
		})
		
	# Save a single master dataset index JSON file at the very end
	with open('master_dataset_labels.json', 'w') as f:
		json.dump(all_dataset_annotations, f, indent=4)
		
	print("Pipeline complete! Master dataset labels saved.")

if __name__ == "__main__":
    run_dataset_pipeline()