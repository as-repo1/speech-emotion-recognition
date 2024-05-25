import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment

def ogg_to_wav(input_file, output_file):
    # Load the OGG audio file
    audio = AudioSegment.from_ogg(input_file)
    
    # Export the audio to WAV format
    audio.export(output_file, format="wav")

def select_file():
    # Open a file dialog window to select an OGG audio file
    input_file = filedialog.askopenfilename(filetypes=[("OGG files", "*.ogg")])
    if input_file:
        # Convert the selected OGG file to WAV format
        output_file = input_file.replace(".ogg", ".wav")
        ogg_to_wav(input_file, output_file)
        status_label.config(text=f"Conversion successful: {output_file}")

# Create a GUI window
root = tk.Tk()
root.title("OGG to WAV Converter")

# Create a button to select an OGG audio file
select_button = tk.Button(root, text="Select OGG File", command=select_file)
select_button.pack(pady=10)

# Create a label to display conversion status
status_label = tk.Label(root, text="")
status_label.pack()

# Start the GUI event loop
root.mainloop()
