import json
import os

# Path to original notebook
original_notebook_path = 'notebooks/brain_tumor.ipynb'
new_notebook_path = 'notebooks/brain_tumor_colab.ipynb'

# Read the original notebook
with open(original_notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create the new cell for Colab setup
colab_setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "import shutil\n",
        "import os\n",
        "\n",
        "# Define source path in Drive (User needs to upload archive_1.zip here)\n",
        "drive_path = '/content/drive/MyDrive/BrainTumorProject/archive_1.zip'\n",
        "\n",
        "# Copy to local Colab environment\n",
        "if os.path.exists(drive_path):\n",
        "    print(f\"Copying {drive_path} to local runtime...\")\n",
        "    shutil.copy(drive_path, 'archive_1.zip')\n",
        "    print(\"Copy complete.\")\n",
        "else:\n",
        "    print(f\"File not found at {drive_path}. Please check the path.\")\n"
    ]
}

# Insert the setup cell at the beginning
notebook['cells'].insert(0, colab_setup_cell)

# Find the cell that calls augmented_data and inject directory creation
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "augmented_data(file_dir=yes_path" in source:
            # Inject directory creation before the function call
            new_source = [
                "import os\n",
                "os.makedirs('augmented_data/yes', exist_ok=True)\n",
                "os.makedirs('augmented_data/no', exist_ok=True)\n",
                "print(\"Created augmented_data directories.\")\n",
                "\n"
            ] + cell['source']
            cell['source'] = new_source
            break

# Write the new notebook
with open(new_notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Created {new_notebook_path}")
