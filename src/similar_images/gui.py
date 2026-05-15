import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk
import threading
import webbrowser

from .classifier import compare_all
from .io import scan_images_from_folders
from .similarity import SimilarityWeights
from .report import build_html_report

# System settings
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class SimilarImagesGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Similar Images Finder")
        self.geometry("800x700")

        self.selected_folders = []
        self.scanning = False
        self.sliders_data = []

        # --- Grid Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Similar Images", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_optionemenu.set("System")

        # --- Main Frame ---
        self.main_frame = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        # 1. Folder Selection
        self.folder_label = ctk.CTkLabel(self.main_frame, text="Folders to Scan", font=ctk.CTkFont(size=16, weight="bold"))
        self.folder_label.grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        self.folder_listbox = tk.Listbox(self.main_frame, height=4, bg="#333333", fg="white", borderwidth=0, highlightthickness=0)
        self.folder_listbox.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.folder_btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.folder_btn_frame.grid(row=2, column=0, sticky="ew")
        
        self.add_folder_btn = ctk.CTkButton(self.folder_btn_frame, text="Add Folder", command=self.add_folder)
        self.add_folder_btn.pack(side="left", padx=(0, 10))
        
        self.clear_folders_btn = ctk.CTkButton(self.folder_btn_frame, text="Clear", command=self.clear_folders, fg_color="transparent", border_width=1)
        self.clear_folders_btn.pack(side="left")

        # 2. Thresholds
        self.thresh_label = ctk.CTkLabel(self.main_frame, text="Thresholds", font=ctk.CTkFont(size=16, weight="bold"))
        self.thresh_label.grid(row=3, column=0, sticky="w", pady=(20, 10))

        self.sim_thresh_slider = self.create_slider(self.main_frame, "Similar Threshold", 0.82, row=4)
        self.dup_thresh_slider = self.create_slider(self.main_frame, "Duplicate Threshold", 0.96, row=5)

        # 3. Weights Section Header
        self.weight_header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.weight_header_frame.grid(row=6, column=0, sticky="ew", pady=(20, 10))
        self.weight_header_frame.grid_columnconfigure(0, weight=1)

        self.weight_label = ctk.CTkLabel(self.weight_header_frame, text="Similarity Weights", font=ctk.CTkFont(size=16, weight="bold"))
        self.weight_label.grid(row=0, column=0, sticky="w")

        self.reset_btn = ctk.CTkButton(self.weight_header_frame, text="Reset to Defaults", command=self.reset_to_defaults, 
                                       fg_color="#A16207", hover_color="#854D0E", height=24, width=120)
        self.reset_btn.grid(row=0, column=1, sticky="e")

        # 4. Weight Sliders
        self.hist_weight = self.create_slider(self.main_frame, "Histogram", 0.3, row=7)
        self.phash_weight = self.create_slider(self.main_frame, "pHash", 0.2, row=8)
        self.dhash_weight = self.create_slider(self.main_frame, "dHash", 0.2, row=9)
        self.hog_weight = self.create_slider(self.main_frame, "HOG", 0.3, row=10)
        self.orb_weight = self.create_slider(self.main_frame, "ORB", 0.0, row=11)
        self.ssim_weight = self.create_slider(self.main_frame, "SSIM", 0.0, row=12)
        self.edge_weight = self.create_slider(self.main_frame, "Edge", 0.0, row=13)

        # 5. Report Options
        self.report_label = ctk.CTkLabel(self.main_frame, text="Report Options", font=ctk.CTkFont(size=16, weight="bold"))
        self.report_label.grid(row=14, column=0, sticky="w", pady=(20, 10))
        
        self.min_score_slider = self.create_slider(self.main_frame, "Min Score in Report", 0.3, row=15)

        # --- Progress & Actions ---
        self.action_frame = ctk.CTkFrame(self, height=150)
        self.action_frame.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")
        self.action_frame.grid_columnconfigure(0, weight=1)

        self.progress_label = ctk.CTkLabel(self.action_frame, text="Ready")
        self.progress_label.grid(row=0, column=0, pady=(10, 0))

        self.progressbar = ctk.CTkProgressBar(self.action_frame)
        self.progressbar.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.progressbar.set(0)

        self.scan_btn = ctk.CTkButton(self.action_frame, text="Start Scan", font=ctk.CTkFont(size=16, weight="bold"), height=40, command=self.start_scan)
        self.scan_btn.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")

    def create_slider(self, parent, label, default_val, row):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, sticky="ew", pady=5)
        frame.grid_columnconfigure(1, weight=1)

        lbl = ctk.CTkLabel(frame, text=label, width=150, anchor="w")
        lbl.grid(row=0, column=0)

        slider = ctk.CTkSlider(frame, from_=0, to=1, number_of_steps=100)
        slider.set(default_val)
        slider.grid(row=0, column=1, sticky="ew")

        val_lbl = ctk.CTkLabel(frame, text=f"{default_val:.2f}", width=50)
        val_lbl.grid(row=0, column=2, padx=(10, 0))

        slider.configure(command=lambda v: val_lbl.configure(text=f"{v:.2f}"))
        
        # Store for reset functionality
        self.sliders_data.append({"slider": slider, "label": val_lbl, "default": default_val})
        
        return slider

    def reset_to_defaults(self):
        for data in self.sliders_data:
            data["slider"].set(data["default"])
            data["label"].configure(text=f"{data['default']:.2f}")

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            path = Path(folder).resolve()
            if path not in self.selected_folders:
                self.selected_folders.append(path)
                self.folder_listbox.insert(tk.END, str(path))

    def clear_folders(self):
        self.selected_folders = []
        self.folder_listbox.delete(0, tk.END)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def start_scan(self):
        if not self.selected_folders:
            messagebox.showwarning("Warning", "Please add at least one folder to scan.")
            return
        
        if self.scanning:
            return

        self.scanning = True
        self.scan_btn.configure(state="disabled", text="Scanning...")
        self.progressbar.set(0)
        
        # Run in thread to keep GUI responsive
        threading.Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        try:
            records = scan_images_from_folders(self.selected_folders, recursive=True)
            if not records:
                self.after(0, lambda: self.finish_scan("No images found.", False))
                return

            weights = SimilarityWeights(
                histogram=self.hist_weight.get(),
                phash=self.phash_weight.get(),
                dhash=self.dhash_weight.get(),
                hog=self.hog_weight.get(),
                orb=self.orb_weight.get(),
                ssim=self.ssim_weight.get(),
                edge=self.edge_weight.get(),
            )

            if weights.total() <= 0:
                self.after(0, lambda: self.finish_scan("At least one weight must be > 0.", False))
                return

            def on_feature_progress(done, total):
                self.after(0, lambda: self.update_progress(f"Extracting features: {done}/{total}", done / total))

            def on_compare_start(total_pairs):
                self.after(0, lambda: self.update_progress(f"Comparing {total_pairs} pairs...", 0))

            def on_compare_progress(done, total):
                self.after(0, lambda: self.update_progress(f"Comparing: {done}/{total}", done / total))

            results, loaded_records = compare_all(
                records=records,
                similar_threshold=self.sim_thresh_slider.get(),
                duplicate_threshold=self.dup_thresh_slider.get(),
                weights=weights,
                on_feature_progress=on_feature_progress,
                on_compare_start=on_compare_start,
                on_compare_progress=on_compare_progress,
            )

            report_path = Path("report.html").resolve()
            min_score = self.min_score_slider.get()
            filtered_results = [r for r in results if r.score >= min_score]
            hidden_count = len(results) - len(filtered_results)

            build_html_report(
                scanned_folders=self.selected_folders,
                output_path=report_path,
                results=filtered_results,
                loaded_count=len(loaded_records),
                skipped_count=len(records) - len(loaded_records),
                similar_threshold=self.sim_thresh_slider.get(),
                duplicate_threshold=self.dup_thresh_slider.get(),
                weights=weights,
                report_min_score=min_score,
                report_max_rows=0,
                hidden_count=hidden_count,
            )

            self.after(0, lambda: self.finish_scan(f"Scan complete! Report saved to {report_path}", True, report_path))

        except Exception as e:
            self.after(0, lambda: self.finish_scan(f"Error: {str(e)}", False))

    def update_progress(self, text, val):
        self.progress_label.configure(text=text)
        self.progressbar.set(val)

    def finish_scan(self, message, success, report_path=None):
        self.scanning = False
        self.scan_btn.configure(state="normal", text="Start Scan")
        self.progress_label.configure(text="Ready")
        
        if success:
            if messagebox.askyesno("Success", f"{message}\n\nOpen report now?"):
                webbrowser.open(str(report_path))
        else:
            messagebox.showerror("Error", message)

def main():
    app = SimilarImagesGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
