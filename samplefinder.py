import os
import numpy as np
import librosa
from fastdtw import fastdtw
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import time
import threading
import audioop
import wave
import concurrent.futures
import tkinter.font as tkFont


# Function to load audio file
def load_audio(file_path):
    print(f"[DEBUG] Loading audio file: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        print(f"[DEBUG] Successfully loaded: {file_path}")
        return y, sr
    except Exception as e:
        print(f"[ERROR] Error loading {file_path}: {e}")
        return None, None

# Function to extract MFCC features from audio
def extract_mfcc(y, sr):
    print("[DEBUG] Extracting MFCC features...")
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print("[DEBUG] MFCC extraction complete.")
        return mfcc
    except Exception as e:
        print(f"[ERROR] Error extracting MFCC: {e}")
        return None

# Function to traverse the directory and get all audio files
def get_audio_files(directory):
    print(f"[DEBUG] Checking if directory exists: {directory}")
    if not os.path.exists(directory):
        print(f"[ERROR] Directory does not exist: {directory}")
        return []

    print(f"[DEBUG] Scanning directory: {directory}")
    audio_files = []
    try:
        print("[DEBUG] Listing files in the directory:")
        for root, _, files in os.walk(directory):
            print(f"[DEBUG] Root: {root}, Files: {files}")  # List files in each folder
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.aif', '.m4a')):  # Added .aif and .m4a
                    audio_files.append(os.path.join(root, file))
        print(f"[DEBUG] Found {len(audio_files)} audio files in {directory}")
    except Exception as e:
        print(f"[ERROR] Error scanning directory {directory}: {e}")
    return audio_files

# Function to compare two audio files using DTW on their MFCCs
def compare_audio(file1, file2):
    print(f"[DEBUG] Comparing {file1} with {file2}")
    y1, sr1 = load_audio(file1)
    y2, sr2 = load_audio(file2)

    if y1 is None or y2 is None:
        return float('inf')  # Return a very high distance if audio loading failed

    mfcc1 = extract_mfcc(y1, sr1)
    mfcc2 = extract_mfcc(y2, sr2)

    if mfcc1 is None or mfcc2 is None:
        return float('inf')  # Return a very high distance if MFCC extraction failed

    # Reshape MFCCs to 2D array (time, features) for DTW
    mfcc1 = np.transpose(mfcc1)
    mfcc2 = np.transpose(mfcc2)

    # Calculate DTW distance (lower distance means more similarity)
    try:
        distance, _ = fastdtw(mfcc1, mfcc2)
        print(f"[DEBUG] DTW distance between {file1} and {file2}: {distance}")
        return distance
    except Exception as e:
        print(f"[ERROR] Error during DTW comparison: {e}")
        return float('inf')  # Return a high distance on error

# Function to find matching audio files based on a similarity threshold
def find_matching_files(target_file, directory, threshold=0.25, update_progress=None, update_eta=None):
    print(f"[DEBUG] Finding matching files for {target_file} in {directory}")
    audio_files = get_audio_files(directory)
    matches = []

    total_files = len(audio_files)
    start_time = time.time()  # Start tracking time

    for idx, file in enumerate(audio_files):
        distance = compare_audio(target_file, file)
        similarity_score = 1 - distance  # Convert distance to similarity (higher is better)

        print(f"[DEBUG] Similarity score between {target_file} and {file}: {similarity_score * 100:.2f}%")

        if similarity_score >= threshold:
            matches.append((file, similarity_score))

        # Update progress and ETA in the GUI
        if update_progress:
            update_progress(idx + 1, total_files)
        
        if update_eta:
            elapsed_time = time.time() - start_time
            remaining_files = total_files - (idx + 1)
            avg_time_per_file = elapsed_time / (idx + 1)
            eta_seconds = avg_time_per_file * remaining_files
            update_eta(eta_seconds)

    # Sort matches by similarity score (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

# Function to open file paths in File Explorer
def open_in_explorer(file_path):
    try:
        os.startfile(file_path)
    except Exception as e:
        print(f"[ERROR] Unable to open file: {e}")

# Function to update debug box
def update_debug_box(message, text_box):
    text_box.insert(tk.END, message + "\n")
    text_box.yview(tk.END)

# GUI functionality
class AudioMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Patches' Sample Finder")  # Update the window title
        self.root.geometry("800x800")  # Set a default size to ensure the window is resizable
        self.create_widgets()

    def create_widgets(self):
        # File selection buttons
        self.target_file = ""
        self.directory = ""
        
        # Top right label for credit
        self.credit_label = tk.Label(self.root, text="made by @patchestheproducer", anchor='ne')
        self.credit_label.pack(pady=10, padx=10, side=tk.TOP, anchor="ne")

        # Audio file selection button
        self.audio_file_btn = tk.Button(self.root, text="Select Audio File", command=self.select_audio_file)
        self.audio_file_btn.pack(pady=10)

        # Directory selection button
        self.directory_btn = tk.Button(self.root, text="Select Directory", command=self.select_directory)
        self.directory_btn.pack(pady=10)

        # Threshold slider
        self.threshold_var = tk.DoubleVar(value=0.25)  # Store the threshold value
    
        # Threshold label and entry box
        threshold_frame = tk.Frame(self.root)
        threshold_frame.pack(pady=10)
    
        threshold_label = tk.Label(threshold_frame, text="Threshold (%)")
        threshold_label.pack(side=tk.LEFT, padx=5)

        self.threshold_entry = tk.Entry(threshold_frame, textvariable=self.threshold_var, width=6)
        self.threshold_entry.pack(side=tk.LEFT)

        # Threshold slider
        self.threshold_slider = tk.Scale(self.root, from_=0, to=100, resolution=1, 
                                      orient=tk.HORIZONTAL, variable=self.threshold_var)
        self.threshold_slider.pack(pady=10)

        # Sync changes between slider and text box
        self.threshold_var.trace_add("write", self.sync_threshold)

        # Debug box for messages
        self.debug_box = tk.Text(self.root, height=10, width=60)
        self.debug_box.pack(pady=10, fill=tk.BOTH, expand=True)

        # ETA label
        self.eta_label = tk.Label(self.root, text="ETA: N/A")
        self.eta_label.pack(pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack(pady=10)

        # Find matches button
        self.find_matches_btn = tk.Button(self.root, text="Find Matching Files", command=self.find_matches)
        self.find_matches_btn.pack(pady=10)

        # Container for links and similarity results
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

    def update_slider_from_entry(self):
        try:
            value = int(self.threshold_entry.get())
            if 0 <= value <= 100:
                self.threshold_slider.set(value)
            else:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid threshold (0-100).")

    def sync_threshold(self, *args):
        """Sync the value between the slider and the text box."""
        try:
            value = float(self.threshold_var.get())
            if 0 <= value <= 100:
                self.threshold_var.set(value)  # Clamp to valid range if necessary
            else:
                raise ValueError
        except ValueError:
            self.update_debug("Invalid threshold value. Please enter a number between 0 and 100.")
            self.threshold_var.set(25)  # Reset to default if invalid input

    
    
    def update_entry_from_slider(self, value):
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(0, str(int(float(value))))

    def select_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.aif *.m4a")])
        if file_path:
            self.target_file = file_path
            file_name = os.path.basename(file_path)
            self.audio_file_btn.config(text=file_name)
            self.update_debug(f"Selected audio file: {self.target_file}")

    def select_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.directory = dir_path

            # Normalize path separators and split into parts
            normalized_path = dir_path.replace("\\", "/")  # Replace backslashes with forward slashes
            dir_parts = normalized_path.strip("/").split("/")  # Split by normalized slash and strip excess

            # Extract the last three folders
            dir_parts_last_three = dir_parts[-3:] if len(dir_parts) >= 3 else dir_parts  # Handle short paths
            shortened_path = "/".join(dir_parts_last_three)

            # Debug outputs
            self.update_debug(f"Full directory path: {self.directory}")
            self.update_debug(f"Extracted last 3 parts: {dir_parts_last_three}")
            self.update_debug(f"Shortened path: {shortened_path}")

            # Update the button text with the shortened path
            self.directory_btn.config(text=shortened_path)

            self.update_debug(f"Selected directory: {self.directory}")



    def update_debug(self, message):
        update_debug_box(message, self.debug_box)

    def update_progress(self, processed, total):
        progress_value = (processed / total) * 100
        self.progress['value'] = progress_value
    
    
    def update_eta(self, eta_seconds):
        # Calculate ETA in minutes and seconds
        eta_minutes = int(eta_seconds // 60)
        eta_seconds = int(eta_seconds % 60)
        self.eta_label.config(text=f"ETA: {eta_minutes}m {eta_seconds}s")

    def find_matches(self):
        if not self.target_file or not self.directory:
            messagebox.showerror("Error", "Please select both an audio file and a directory.")
            return
        
        threshold = self.threshold_slider.get() / 100
        self.update_debug(f"Starting search with threshold: {threshold * 100:.2f}%")

        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Run the matching process in a separate thread
        def run_find_matches():
            # Update progress and ETA functions
            def update_progress(processed, total):
                self.update_progress(processed, total)

            matches = find_matching_files(self.target_file, self.directory, threshold, update_progress, self.update_eta)

            if matches:
                self.update_debug(f"\nFound {len(matches)} matching files:")
                for match, score in matches:
                    self.update_debug(f"Match: {match} (Similarity: {score * 100:.2f}%)")
                    
                    # Extract the last 3 folders
                    normalized_path = match.replace("\\", "/")
                    dir_parts = normalized_path.strip("/").split("/")
                    last_three_folders = dir_parts[-3:]  # Get the last 3 folders
                    shortened_path = "/".join(last_three_folders)  # Join them to form the shortened path

                    match_frame = tk.Frame(self.results_frame)
                    match_frame.pack(fill=tk.X, pady=2)

                    # Display the shortened path as a clickable link
                    link = tk.Label(match_frame, text=shortened_path, fg="blue", cursor="hand2")
                    link.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    link.bind("<Button-1>", lambda event, path=match: open_in_explorer(path))

                    # Display similarity score
                    similarity_label = tk.Label(match_frame, text=f"{score * 100:.2f}%", fg="green")
                    similarity_label.pack(side=tk.RIGHT)
            else:
                self.update_debug("No matching files found")

        threading.Thread(target=run_find_matches, daemon=True).start()



# Run the application
root = tk.Tk()
app = AudioMatcherGUI(root)
root.mainloop()
