import os

def rename_images(folder_path, base_name="image"):
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    
    # List all image files in the folder
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    images.sort()  # Sort to ensure numbering is consistent

    # Rename images sequentially
    for index, image in enumerate(images, start=1):
        ext = os.path.splitext(image)[1]  # Get the file extension
        new_name = f"{base_name}_{index}{ext}"  # Construct new name
        old_path = os.path.join(folder_path, image)
        new_path = os.path.join(folder_path, new_name)

        # Skip renaming if the new name already exists
        if os.path.exists(new_path):
            print(f"Skipping: {image} → {new_name} (Already exists)")
            continue

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {image} → {new_name}")

# Example usage
folder_path = "C:\\Users\\Admin\\Desktop\\Projects\\SightVision_App\\Gender_Detection\\gender_dataset_face\\woman"  # Change this to your folder path
rename_images(folder_path, base_name="face")
