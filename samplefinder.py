import os
import numpy as np
import librosa
from fastdtw import fastdtw
import tkinter as tk
from tkinter import filedialog, messagebox
import webbrowser
import time
import threading

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
        self.root.geometry("600x600")  # Set a default size to ensure the window is resizable
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
        self.threshold_slider = tk.Scale(self.root, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, label="Threshold")
        self.threshold_slider.set(0.25)
        self.threshold_slider.pack(pady=10)

        # Debug box for messages
        self.debug_box = tk.Text(self.root, height=10, width=60)
        self.debug_box.pack(pady=10, fill=tk.BOTH, expand=True)

        # ETA label
        self.eta_label = tk.Label(self.root, text="ETA: N/A")
        self.eta_label.pack(pady=5)

        # Find matches button
        self.find_matches_btn = tk.Button(self.root, text="Find Matching Files", command=self.find_matches)
        self.find_matches_btn.pack(pady=10)

        # Container for links and similarity results
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

    def select_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.aif *.m4a")])
        if file_path:
            self.target_file = file_path
            self.update_debug(f"Selected audio file: {self.target_file}")

    def select_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.directory = dir_path
            self.update_debug(f"Selected directory: {self.directory}")

    def update_debug(self, message):
        update_debug_box(message, self.debug_box)

    def update_eta(self, eta_seconds):
        # Calculate ETA in minutes and seconds
        eta_minutes = int(eta_seconds // 60)
        eta_seconds = int(eta_seconds % 60)
        self.eta_label.config(text=f"ETA: {eta_minutes}m {eta_seconds}s")

    def find_matches(self):
        if not self.target_file or not self.directory:
            messagebox.showerror("Error", "Please select both an audio file and a directory.")
            return
        
        threshold = self.threshold_slider.get()
        self.update_debug(f"Starting search with threshold: {threshold * 100:.2f}%")

        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Run the matching process in a separate thread
        def run_find_matches():
            # Update progress and ETA functions
            def update_progress(processed, total):
                self.update_debug(f"Processing Files: {processed} out of {total}")

            matches = find_matching_files(self.target_file, self.directory, threshold, update_progress, self.update_eta)

            if matches:
                self.update_debug(f"\nFound {len(matches)} matching files:")
                for match, score in matches:
                    self.update_debug(f"Match: {match} (Similarity: {score * 100:.2f}%)")
                    link = tk.Label(self.results_frame, text=match, fg="blue", cursor="hand2")
                    link.pack(fill=tk.X)
                    link.bind("<Button-1>", lambda e, path=match: open_in_explorer(path))
                    score_label = tk.Label(self.results_frame, text=f"Similarity: {score * 100:.2f}%", fg="green")
                    score_label.pack()
            else:
                self.update_debug("No matching files found", self.debug_box)
                no_match_label = tk.Label(self.results_frame, text="No matching files found", fg="red")
                no_match_label.pack()

        threading.Thread(target=run_find_matches, daemon=True).start()

# Create and run the application
root = tk.Tk()
app = AudioMatcherGUI(root)
root.mainloop()
