
"""
A user-friendly graphical interface for the GSAT-based molecular toxicity analyzer
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import webbrowser
import numpy as np
from PIL import Image, ImageTk

# Set matplotlib backend for tkinter embedding
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for embedding in tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTk, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Import the analyzer functions
from fixed_smiles_toxicity_analyzer import (
    load_trained_gsat_model, predict_with_actual_model, 
    get_toxicity_level_info, preload_model, get_current_analysis_info
)

# Import the similarity representative analyzer
from similarity_representative import SimilarityRepresentativeAnalyzer

class SMILESToxicityGUI:
    def __init__(self, root):
        print("🏗️  Initializing GUI components...")
        self.root = root
        self.root.title("GSAT SMILES Toxicity Analyzer")
        self.root.geometry("1000x700")
        
        # Modern dark theme color scheme
        self.bg_primary = '#1e1e2e'     # Dark blue-gray
        self.bg_secondary = '#313244'   # Lighter dark blue-gray  
        self.bg_accent = '#89b4fa'      # Bright blue
        self.accent_color = '#89b4fa'   # Bright blue
        self.accent_bright = '#74c7ec'  # Light blue
        self.text_primary = '#cdd6f4'   # Light gray
        self.text_secondary = '#9399b2' # Medium gray
        self.success_color = '#a6e3a1'  # Light green
        self.warning_color = '#f9e2af'  # Light yellow
        self.error_color = '#f38ba8'    # Light red
        
        self.root.configure(bg=self.bg_primary)
        
        # Model components (loaded once)
        self.model_components = None
        self.loading = False
        
        print("🎨 Setting up user interface...")
        self.setup_ui()
        print("🧠 Starting model loading in background...")
        self.load_model_in_background()
    
    def setup_ui(self):
        """Set up the GUI components"""
        # Modern header with gradient effect
        header_frame = tk.Frame(self.root, bg=self.bg_primary, height=120)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Create gradient-like header background
        gradient_frame = tk.Frame(header_frame, bg=self.bg_secondary, height=80)
        gradient_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        
        # Animated title with modern typography
        title_label = tk.Label(gradient_frame, text="🧬 GSAT MOLECULAR ANALYZER", 
                              font=('Segoe UI', 24, 'bold'), 
                              bg=self.bg_secondary, fg=self.accent_bright)
        title_label.pack(pady=(15, 5))
        
        subtitle_label = tk.Label(gradient_frame, 
                                 text="⚡ Advanced Graph Neural Network • Real-time Toxicity Prediction • AI-Powered Analysis ⚡", 
                                 font=('Segoe UI', 11), 
                                 bg=self.bg_secondary, fg=self.text_secondary)
        subtitle_label.pack()
        
        # Add status indicator bar
        self.status_indicator = tk.Frame(header_frame, bg=self.warning_color, height=4)
        self.status_indicator.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Animate the title (color cycling)
        self.animate_title(title_label)
        
        # Create modern notebook for tabs
        notebook = ttk.Notebook(self.root, style='Modern.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Single Analysis Tab with modern styling
        single_frame = tk.Frame(notebook, bg=self.bg_primary)
        notebook.add(single_frame, text="🧪 SINGLE ANALYSIS")
        
        # Batch Analysis Tab with modern styling
        batch_frame = tk.Frame(notebook, bg=self.bg_primary)
        notebook.add(batch_frame, text="📊 BATCH PROCESSING")
        
        # Model Explanation Tab with modern styling
        explanation_frame = tk.Frame(notebook, bg=self.bg_primary)
        notebook.add(explanation_frame, text="🧠 MODEL EXPLANATION")
        
        # Similarity Analysis Tab with modern styling
        similarity_frame = tk.Frame(notebook, bg=self.bg_primary)
        notebook.add(similarity_frame, text="🔬 SIMILARITY ANALYSIS")
        
        # Setup single analysis tab
        self.setup_single_analysis_tab(single_frame)
        
        # Setup batch analysis tab
        self.setup_batch_analysis_tab(batch_frame)
        
        # Setup model explanation tab
        self.setup_model_explanation_tab(explanation_frame)
        
        # Setup similarity analysis tab
        self.setup_similarity_analysis_tab(similarity_frame)
        
        # Add status bar at the bottom
        self.setup_status_bar()
        
    def animate_title(self, label):
        """Create animated title effect"""
        colors = [self.accent_bright, self.bg_accent, '#cba6f7', '#f5c2e7', self.accent_bright]
        self.title_color_index = getattr(self, 'title_color_index', 0)
        
        def cycle_color():
            label.configure(fg=colors[self.title_color_index])
            self.title_color_index = (self.title_color_index + 1) % len(colors)
            self.root.after(2000, cycle_color)  # Change color every 2 seconds
        
        cycle_color()
        
    def setup_single_analysis_tab(self, main_frame):
        """Setup the single molecule analysis tab"""
        # Modern input section with card-like design
        input_card = tk.Frame(main_frame, bg=self.bg_secondary, relief='flat', bd=0)
        input_card.pack(fill=tk.X, pady=10, padx=20)
        
        # Add subtle border effect
        border_frame = tk.Frame(input_card, bg=self.bg_accent, height=3)
        border_frame.pack(fill=tk.X)
        
        input_content = tk.Frame(input_card, bg=self.bg_secondary, padx=20, pady=20)
        input_content.pack(fill=tk.X)
        
        # Modern header for input section
        input_header = tk.Label(input_content, text="🧪 MOLECULAR INPUT", 
                               font=('Segoe UI', 14, 'bold'), 
                               bg=self.bg_secondary, fg=self.accent_bright)
        input_header.pack(anchor='w', pady=(0, 15))
        
        # SMILES input with modern styling
        smiles_label = tk.Label(input_content, text="💊 SMILES String:", 
                               font=('Segoe UI', 11, 'bold'), 
                               bg=self.bg_secondary, fg=self.text_primary)
        smiles_label.pack(anchor='w', pady=(0, 8))
        
        self.smiles_var = tk.StringVar(value="c1ccc(cc1)O")  # Default: phenol
        smiles_frame = tk.Frame(input_content, bg=self.bg_secondary)
        smiles_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Modern entry with custom styling
        self.smiles_entry = tk.Entry(smiles_frame, textvariable=self.smiles_var, 
                                    font=('Consolas', 14, 'bold'), width=40,
                                    bg=self.bg_primary, fg=self.text_primary,
                                    insertbackground=self.accent_bright,
                                    relief='flat', bd=5,
                                    highlightthickness=2,
                                    highlightcolor=self.bg_accent,
                                    highlightbackground=self.bg_primary)
        self.smiles_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        
        # Example molecules dropdown with modern styling
        example_label = tk.Label(input_content, text="📚 Select Example Molecule:", 
                                font=('Segoe UI', 11, 'bold'), 
                                bg=self.bg_secondary, fg=self.text_primary)
        example_label.pack(anchor='w', pady=(15, 8))
        
        self.example_var = tk.StringVar()
        examples = [
            ("Phenol", "c1ccc(cc1)O"),
            ("Benzene", "c1ccccc1"),
            ("Chlorobenzene", "c1ccc(cc1)Cl"),
            ("Toluene", "Cc1ccccc1"),
            ("4-Chlorophenol", "Oc1ccc(Cl)cc1"),
            ("Aniline", "Nc1ccccc1"),
            ("Benzoic acid", "OC(=O)c1ccccc1")
        ]
        
        example_combo = ttk.Combobox(input_content, textvariable=self.example_var, 
                                    values=[f"{name}: {smiles}" for name, smiles in examples],
                                    state="readonly", width=50, font=('Segoe UI', 10))
        example_combo.pack(fill=tk.X, pady=(0, 15))
        example_combo.bind('<<ComboboxSelected>>', self.on_example_selected)
        
        # Output directory with modern styling
        output_frame = tk.Frame(input_content, bg=self.bg_secondary)
        output_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(output_frame, text="📁 Output Directory:", font=('Segoe UI', 11, 'bold'), 
                bg=self.bg_secondary, fg=self.text_primary).pack(anchor='w', pady=(0, 8))
        
        dir_frame = tk.Frame(output_frame, bg=self.bg_secondary)
        dir_frame.pack(fill=tk.X, pady=(0, 0))
        
        self.output_dir_var = tk.StringVar(value=os.getcwd())
        self.output_dir_entry = tk.Entry(dir_frame, textvariable=self.output_dir_var, 
                                        font=('Consolas', 10), state='readonly',
                                        bg=self.bg_primary, fg=self.text_secondary,
                                        relief='flat', bd=3)
        self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        
        browse_btn = tk.Button(dir_frame, text="📂 BROWSE", command=self.browse_output_dir,
                              bg=self.accent_color, fg=self.bg_primary, 
                              font=('Segoe UI', 9, 'bold'),
                              relief='flat', bd=0, padx=15, pady=5, cursor='hand2')
        browse_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Modern control buttons section
        button_section = tk.Frame(input_content, bg=self.bg_secondary)
        button_section.pack(fill=tk.X, pady=(10, 0))
        
        button_frame = tk.Frame(button_section, bg=self.bg_secondary)
        button_frame.pack()
        
        # Primary analyze button with modern styling
        self.analyze_btn = tk.Button(button_frame, text="⚡ ANALYZE MOLECULE", 
                                    command=self.analyze_molecule, 
                                    font=('Segoe UI', 14, 'bold'),
                                    bg=self.bg_accent, fg=self.bg_primary, 
                                    activebackground=self.accent_bright,
                                    activeforeground=self.bg_primary,
                                    relief='flat', bd=0, padx=30, pady=12,
                                    cursor='hand2', state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        # Secondary clear button
        self.clear_btn = tk.Button(button_frame, text="🗑️ CLEAR", 
                                  command=self.clear_results, 
                                  font=('Segoe UI', 11, 'bold'),
                                  bg=self.bg_primary, fg='black',
                                  activebackground=self.error_color,
                                  activeforeground='black',
                                  relief='flat', bd=2, padx=20, pady=10,
                                  cursor='hand2')
        self.clear_btn.pack(side=tk.LEFT)
        
        # Add hover effects to buttons
        self.add_button_hover_effects()
        
        # Create progress bar and results sections
        self.setup_progress_and_results_sections(main_frame)
        
    def add_button_hover_effects(self):
        """Add hover effects to main control buttons"""
        # Main action buttons hover effects
        def on_analyze_enter(e):
            if self.analyze_btn['state'] != 'disabled':
                self.analyze_btn.configure(bg=self.accent_bright, fg=self.bg_primary)
        
        def on_analyze_leave(e):
            if self.analyze_btn['state'] != 'disabled':
                self.analyze_btn.configure(bg=self.bg_accent, fg=self.bg_primary)
        
        def on_clear_enter(e):
            self.clear_btn.configure(bg=self.error_color, fg='black')
        
        def on_clear_leave(e):
            self.clear_btn.configure(bg=self.bg_primary, fg='black')
        
        # Bind hover effects to main buttons
        self.analyze_btn.bind('<Enter>', on_analyze_enter)
        self.analyze_btn.bind('<Leave>', on_analyze_leave)
        self.clear_btn.bind('<Enter>', on_clear_enter)
        self.clear_btn.bind('<Leave>', on_clear_leave)
        
    def setup_progress_and_results_sections(self, parent_frame):
        """Set up progress bar and results sections"""
        # Modern progress bar section with dark theme
        self.progress_frame = tk.Frame(parent_frame, bg=self.bg_secondary, relief='flat', bd=2)
        self.progress_frame.pack(fill=tk.X, pady=(15, 0), padx=20)
        
        self.progress_var = tk.StringVar(value="⚡ Loading GSAT model components...")
        self.progress_label = tk.Label(self.progress_frame, textvariable=self.progress_var, 
                                      font=('Segoe UI', 11, 'normal'), bg=self.bg_secondary, 
                                      fg=self.accent_bright, pady=8)
        self.progress_label.pack()
        
        # Create modern styled progress bar
        progress_style = ttk.Style()
        progress_style.theme_use('clam')
        progress_style.configure("Modern.Horizontal.TProgressbar",
                               background=self.accent_color,
                               troughcolor=self.bg_primary,
                               borderwidth=0,
                               lightcolor=self.accent_bright,
                               darkcolor=self.accent_color)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate',
                                          style="Modern.Horizontal.TProgressbar",
                                          length=400)
        self.progress_bar.pack(fill=tk.X, pady=(5, 8), padx=10)
        self.progress_bar.start()
        
        # Modern results section with dark theme
        results_frame = tk.Frame(parent_frame, bg=self.bg_secondary, relief='flat', bd=0)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0), padx=20)
        
        # Results header with modern styling
        results_header = tk.Frame(results_frame, bg=self.bg_secondary)
        results_header.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(results_header, text="📊 ANALYSIS RESULTS & VISUALIZATION", 
                font=('Segoe UI', 14, 'bold'), bg=self.bg_secondary, 
                fg=self.text_primary).pack(anchor='w')
        
        tk.Label(results_header, text="Real-time molecular toxicity prediction with attention mapping", 
                font=('Segoe UI', 10), bg=self.bg_secondary, 
                fg=self.text_secondary).pack(anchor='w', pady=(2, 0))
        
        # Create modern paned window for results and visualization
        results_paned = tk.PanedWindow(results_frame, orient=tk.HORIZONTAL, 
                                     bg=self.bg_secondary, bd=0, sashwidth=8,
                                     sashrelief='flat')
        results_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel: Modern text results
        text_frame = tk.Frame(results_paned, bg=self.bg_primary, relief='flat', bd=1)
        results_paned.add(text_frame, minsize=300)
        
        text_header = tk.Frame(text_frame, bg=self.bg_primary)
        text_header.pack(fill=tk.X, pady=8, padx=10)
        
        tk.Label(text_header, text="📊 Analysis Summary", font=('Segoe UI', 12, 'bold'), 
                bg=self.bg_primary, fg=self.accent_bright).pack(anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(text_frame, height=12, width=40,
                                                     font=('Consolas', 10), bg=self.bg_primary,
                                                     fg=self.text_primary, wrap=tk.WORD,
                                                     insertbackground=self.accent_color,
                                                     selectbackground=self.accent_color,
                                                     relief='flat', bd=0)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Right panel: Modern visualization panel (increased minimum size)
        viz_frame = tk.Frame(results_paned, bg=self.bg_primary, relief='flat', bd=1)
        results_paned.add(viz_frame, minsize=500)
        
        viz_header = tk.Frame(viz_frame, bg=self.bg_primary)
        viz_header.pack(fill=tk.X, pady=8, padx=10)
        
        tk.Label(viz_header, text="🧬 Molecular Toxicity Map", font=('Segoe UI', 12, 'bold'), 
                bg=self.bg_primary, fg=self.accent_bright).pack(anchor='w')
        
        tk.Label(viz_header, text="💡 Double-click visualization for full resolution view", 
                font=('Segoe UI', 9), bg=self.bg_primary, fg=self.text_secondary).pack(anchor='w', pady=(2, 0))
        
        # Modern image display with dark theme
        self.image_display_frame = tk.Frame(viz_frame, bg=self.bg_secondary, relief='flat', bd=1)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create modern styled label to hold the image
        self.image_label = tk.Label(self.image_display_frame, 
                                   text="🧬 Ready for Analysis\n\nMolecular visualization will appear here\nafter running toxicity prediction",
                                   bg=self.bg_secondary, fg=self.text_secondary, 
                                   font=('Segoe UI', 12),
                                   relief='flat', bd=0, pady=20)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Keep matplotlib as backup (hidden)
        self.viz_figure = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        self.viz_canvas = FigureCanvasTk(self.viz_figure, viz_frame)
        # Don't pack the matplotlib canvas - keep it hidden for now
        
        print(f"📊 Direct tkinter image display created")
        
        # Initialize with placeholder
        self.show_placeholder_visualization()
        
        # Modern results action buttons
        results_btn_frame = tk.Frame(results_frame, bg=self.bg_secondary)
        results_btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Action buttons with modern styling
        self.view_image_btn = tk.Button(results_btn_frame, text="🖼️ VIEW VISUALIZATION", 
                                       command=self.view_visualization, font=('Segoe UI', 10, 'bold'),
                                       bg=self.success_color, fg='black', 
                                       state=tk.DISABLED, relief='flat', bd=0,
                                       padx=15, pady=8, cursor='hand2')
        self.view_image_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.view_csv_btn = tk.Button(results_btn_frame, text="📊 VIEW CSV DATA", 
                                     command=self.view_csv_data, font=('Segoe UI', 10, 'bold'),
                                     bg=self.warning_color, fg='black', 
                                     state=tk.DISABLED, relief='flat', bd=0,
                                     padx=15, pady=8, cursor='hand2')
        self.view_csv_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.open_folder_btn = tk.Button(results_btn_frame, text="📁 OPEN OUTPUT FOLDER", 
                                        command=self.open_output_folder, font=('Segoe UI', 10, 'bold'),
                                        bg=self.accent_color, fg='black', 
                                        relief='flat', bd=0, padx=15, pady=8, cursor='hand2')
        self.open_folder_btn.pack(side=tk.RIGHT)
        
        # Quick file access section with modern styling
        file_btn_frame = tk.Frame(results_frame, bg=self.bg_secondary)
        file_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(file_btn_frame, text="⚡ QUICK FILE ACCESS:", font=('Segoe UI', 9, 'bold'), 
                bg=self.bg_secondary, fg=self.accent_bright).pack(side=tk.LEFT, padx=(0, 15))
        
        self.open_image_btn = tk.Button(file_btn_frame, text="🖼️ IMAGE", 
                                       command=self.open_image_file, font=('Segoe UI', 9),
                                       bg=self.bg_primary, fg='black', 
                                       state=tk.DISABLED, relief='flat', bd=1,
                                       padx=10, pady=4, cursor='hand2')
        self.open_image_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.open_csv_btn = tk.Button(file_btn_frame, text="📊 CSV", 
                                     command=self.open_csv_file, font=('Segoe UI', 9),
                                     bg=self.bg_primary, fg='black', 
                                     state=tk.DISABLED, relief='flat', bd=1,
                                     padx=10, pady=4, cursor='hand2')
        self.open_csv_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.open_report_btn = tk.Button(file_btn_frame, text="📄 REPORT", 
                                        command=self.open_report_file, font=('Segoe UI', 9),
                                        bg=self.bg_primary, fg='black', 
                                        state=tk.DISABLED, relief='flat', bd=1,
                                        padx=10, pady=4, cursor='hand2')
        self.open_report_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        # Initialize results variables
        self.last_visualization = None
        self.last_csv_file = None
        self.current_mol_data = None
        self.current_image_path = None
        self.current_csv_path = None
        self.current_report_path = None

    def setup_status_bar(self):
        """Set up the status bar at the bottom of the window"""
        # Modern status bar with dark theme
        self.status_var = tk.StringVar(value="⚡ Ready - GSAT Model Loaded")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief='flat', 
                             anchor=tk.W, font=('Segoe UI', 10), bg=self.bg_primary, 
                             fg=self.text_secondary, pady=8, padx=20, bd=0)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def show_placeholder_visualization(self):
        """Show placeholder visualization when no analysis is done"""
        print("🔧 Setting up DIRECT TKINTER placeholder...")
        
        # Update the tkinter label directly
        self.image_label.configure(
            text="🧬 MOLECULAR TOXICITY MAP\n\n⚠️ READY FOR ANALYSIS ⚠️\n\nEnter a SMILES and click Analyze\nto see molecular visualization",
            bg='lightyellow', fg='darkblue', font=('Arial', 14, 'bold'),
            relief='ridge', bd=3
        )
        
        print("✅ Direct tkinter placeholder set up - SHOULD BE VISIBLE!")

    def update_embedded_visualization(self, results):
        """Update the embedded visualization with the analysis results using direct tkinter approach"""
        print(f"� Updating embedded visualization with direct tkinter approach...")
        
        try:
            from PIL import Image, ImageTk
            import os
            
            # Get the PNG file from results
            png_file = results['image_file']
            print(f"🔍 Looking for PNG: {png_file}")
            
            if os.path.exists(png_file):
                print(f"✅ PNG file found! Size: {os.path.getsize(png_file)} bytes")
                
                # Load the image with PIL
                pil_image = Image.open(png_file)
                print(f"📏 Original image size: {pil_image.size}")
                
                # Get panel size more reliably by checking multiple sources
                self.image_display_frame.update_idletasks()
                self.image_label.update_idletasks()
                self.root.update_idletasks()
                
                # Try to get the actual display frame size first
                frame_width = self.image_display_frame.winfo_width()
                frame_height = self.image_display_frame.winfo_height()
                
                # If frame size not available, use label size
                if frame_width <= 1 or frame_height <= 1:
                    frame_width = self.image_label.winfo_width()
                    frame_height = self.image_label.winfo_height()
                
                # If still not available, calculate from window size
                if frame_width <= 1 or frame_height <= 1:
                    root_width = self.root.winfo_width()
                    root_height = self.root.winfo_height()
                    # From the screenshot, the right panel appears to be about 60% width, 70% height
                    frame_width = max(500, int(root_width * 0.6))
                    frame_height = max(400, int(root_height * 0.7))
                
                # Final fallback - use fixed reasonable size
                if frame_width <= 1 or frame_height <= 1:
                    frame_width, frame_height = 700, 500
                
                print(f"� Label area size: {frame_width}x{frame_height}")
                
                # Calculate resize ratio to fit in the label, maintaining aspect ratio
                img_width, img_height = pil_image.size
                
                # Improved scaling strategy - fit to panel with margin
                # Account for padding in the display (30px margins)
                usable_width = max(200, frame_width - 60)
                usable_height = max(150, frame_height - 80)
                
                # Calculate scaling ratios
                width_ratio = usable_width / img_width
                height_ratio = usable_height / img_height
                
                # Use the smaller ratio to ensure it fits completely
                ratio = min(width_ratio, height_ratio)
                
                # Apply safety margin to prevent any overflow
                ratio *= 0.85
                
                # Allow aggressive scaling to fit - remove restrictive minimum
                min_ratio = 0.1   # Allow scaling down to 10% if needed
                max_ratio = 2.0   # Don't scale up too much for quality
                ratio = max(min_ratio, min(ratio, max_ratio))
                
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                print(f"📏 Smart panel-fit resizing: {new_width}x{new_height} (ratio: {ratio:.3f})")
                print(f"   Original image: {img_width}x{img_height}")
                print(f"   Panel area: {frame_width}x{frame_height}")
                print(f"   Usable area: {usable_width}x{usable_height}")
                print(f"   Width ratio: {width_ratio:.3f}, Height ratio: {height_ratio:.3f}")
                print(f"   Final scaling: Will fit in panel with margin")
                
                # Resize with highest quality settings
                resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to tkinter PhotoImage
                tk_image = ImageTk.PhotoImage(resized_image)
                
                # Update the label with the image - ensure proper fitting
                self.image_label.configure(
                    image=tk_image,
                    text="",  # Clear text when showing image
                    bg='white',
                    relief='sunken',
                    bd=2,
                    width=new_width,
                    height=new_height,
                    compound='center'  # Center the image
                )
                
                # Keep a reference to prevent garbage collection
                self.image_label.image = tk_image
                # Also store original image for full-resolution viewing
                self.original_image = pil_image
                self.current_png_file = png_file
                
                # Add double-click handler for full resolution view
                self.image_label.bind("<Double-Button-1>", self.show_full_resolution)
                
                print("✅ Image loaded and displayed in tkinter Label!")
                print("💡 Double-click the image to view at full resolution")
                
            else:
                print(f"❌ PNG file not found: {png_file}")
                self.image_label.configure(
                    text=f"❌ Visualization file not found:\n{png_file}\n\nTry running analysis again",
                    bg='lightcoral', fg='white', font=('Arial', 12),
                    image=""  # Clear any existing image
                )
                # Clear image reference
                if hasattr(self.image_label, 'image'):
                    self.image_label.image = None
                
        except Exception as e:
            print(f"❌ Error loading visualization: {e}")
            import traceback
            traceback.print_exc()
            
            self.image_label.configure(
                text=f"❌ Error loading visualization:\n{str(e)}\n\nTry running analysis again",
                bg='lightcoral', fg='white', font=('Arial', 12),
                image=""  # Clear any existing image
            )
            # Clear image reference
            if hasattr(self.image_label, 'image'):
                self.image_label.image = None
        
        print("✅ Embedded visualization update complete!")

    def show_full_resolution(self, event=None):
        """Show the visualization at full resolution in a separate window"""
        print("🔍 Opening full resolution viewer...")
        
        try:
            if hasattr(self, 'original_image') and self.original_image:
                # Create a new window for full resolution display
                full_res_window = tk.Toplevel(self.root)
                full_res_window.title("🔍 Full Resolution - Molecular Toxicity Visualization")
                
                # Calculate window size based on screen size
                screen_width = full_res_window.winfo_screenwidth()
                screen_height = full_res_window.winfo_screenheight()
                
                # Use 90% of screen size, but maintain aspect ratio
                img_width, img_height = self.original_image.size
                max_width = int(screen_width * 0.9)
                max_height = int(screen_height * 0.9)
                
                ratio = min(max_width / img_width, max_height / img_height)
                display_width = int(img_width * ratio)
                display_height = int(img_height * ratio)
                
                print(f"📺 Full-res display: {display_width}x{display_height} (ratio: {ratio:.3f})")
                
                # Resize for full-screen viewing with highest quality
                full_res_image = self.original_image.resize((display_width, display_height), Image.LANCZOS)
                full_res_tk_image = ImageTk.PhotoImage(full_res_image)
                
                # Create scrollable canvas for very large images
                canvas_frame = tk.Frame(full_res_window)
                canvas_frame.pack(fill=tk.BOTH, expand=True)
                
                canvas = tk.Canvas(canvas_frame, width=display_width, height=display_height)
                h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
                v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
                
                canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
                
                # Pack scrollbars
                h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
                v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                # Add image to canvas
                canvas.create_image(0, 0, anchor=tk.NW, image=full_res_tk_image)
                canvas.configure(scrollregion=canvas.bbox("all"))
                
                # Keep reference to prevent garbage collection
                canvas.image = full_res_tk_image
                
                # Set window size and center it
                full_res_window.geometry(f"{min(display_width + 50, max_width)}x{min(display_height + 50, max_height)}")
                
                # Center the window
                x = (screen_width - display_width) // 2
                y = (screen_height - display_height) // 2
                full_res_window.geometry(f"+{x}+{y}")
                
                # Add info label
                info_frame = tk.Frame(full_res_window, bg='lightgray')
                info_frame.pack(side=tk.BOTTOM, fill=tk.X)
                
                info_text = f"Original: {img_width}x{img_height} px | Displayed: {display_width}x{display_height} px | File: {os.path.basename(self.current_png_file)}"
                tk.Label(info_frame, text=info_text, bg='lightgray', font=('Arial', 10)).pack(pady=5)
                
                print("✅ Full resolution viewer opened!")
                
        except Exception as e:
            print(f"❌ Error opening full resolution viewer: {e}")
            messagebox.showerror("Error", f"Could not open full resolution view:\n{str(e)}")

    def setup_batch_analysis_tab(self, main_frame):
        """Setup the batch analysis tab"""
        # File upload section
        upload_frame = tk.LabelFrame(main_frame, text="CSV File Upload", font=('Arial', 12, 'bold'), 
                                   bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        file_frame = tk.Frame(upload_frame, bg='#f0f0f0')
        file_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Label(file_frame, text="Select CSV File:", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0').pack(anchor='w')
        
        csv_select_frame = tk.Frame(file_frame, bg='#f0f0f0')
        csv_select_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.csv_file_var = tk.StringVar()
        self.csv_file_entry = tk.Entry(csv_select_frame, textvariable=self.csv_file_var, 
                                      font=('Arial', 10), state='readonly')
        self.csv_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_csv_btn = tk.Button(csv_select_frame, text="Browse", command=self.browse_csv_file,
                                  bg='#3498db', fg='black', font=('Arial', 9))
        browse_csv_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Column selection
        column_frame = tk.Frame(upload_frame, bg='#f0f0f0')
        column_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(column_frame, text="Select SMILES Column:", font=('Arial', 10, 'bold'), 
                bg='#f0f0f0').pack(anchor='w')
        
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(column_frame, textvariable=self.column_var, 
                                        state="readonly", width=30)
        self.column_combo.pack(fill=tk.X, pady=(5, 0))
        
        # Batch processing controls
        batch_control_frame = tk.Frame(main_frame, bg='#f0f0f0')
        batch_control_frame.pack(pady=10)
        
        self.batch_analyze_btn = tk.Button(batch_control_frame, text="🚀 Batch Analyze", 
                                          command=self.batch_analyze, font=('Arial', 12, 'bold'),
                                          bg='#e74c3c', fg='black', padx=20, pady=10,
                                          state=tk.DISABLED)
        self.batch_analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.batch_clear_btn = tk.Button(batch_control_frame, text="🗑️ Clear", 
                                        command=self.clear_batch_results, font=('Arial', 10),
                                        bg='#95a5a6', fg='black', padx=15, pady=8)
        self.batch_clear_btn.pack(side=tk.LEFT)
        
        # Batch progress
        self.batch_progress_frame = tk.Frame(main_frame, bg='#f0f0f0')
        
        self.batch_progress_var = tk.StringVar()
        self.batch_progress_label = tk.Label(self.batch_progress_frame, textvariable=self.batch_progress_var, 
                                            font=('Arial', 10), bg='#f0f0f0', fg='#e74c3c')
        self.batch_progress_label.pack()
        
        self.batch_progress_bar = ttk.Progressbar(self.batch_progress_frame, mode='determinate')
        self.batch_progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Batch results
        batch_results_frame = tk.LabelFrame(main_frame, text="Batch Results", font=('Arial', 12, 'bold'), 
                                          bg='#f0f0f0', fg='#2c3e50', padx=10, pady=10)
        batch_results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Results text area for batch
        batch_text_frame = tk.Frame(batch_results_frame, bg='#f0f0f0')
        batch_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.batch_results_text = tk.Text(batch_text_frame, font=('Courier', 10), wrap=tk.WORD, 
                                         bg='white', fg='black', height=10)
        batch_scrollbar = tk.Scrollbar(batch_text_frame, orient="vertical", command=self.batch_results_text.yview)
        self.batch_results_text.configure(yscrollcommand=batch_scrollbar.set)
        
        self.batch_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        batch_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Batch result buttons
        batch_btn_frame = tk.Frame(batch_results_frame, bg='#f0f0f0')
        batch_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.export_batch_btn = tk.Button(batch_btn_frame, text="📄 Export Batch Results", 
                                         command=self.export_batch_results, font=('Arial', 10),
                                         bg='#27ae60', fg='white', state=tk.DISABLED)
        self.export_batch_btn.pack(side=tk.LEFT)
    
    def load_model_in_background(self):
        """Load model components in background thread"""
        def load():
            try:
                # Actually preload the model to cache it
                success = preload_model()
                if success:
                    self.model_components = True
                    self.root.after(0, self.on_model_loaded)
                else:
                    self.root.after(0, lambda: self.on_model_error("Failed to preload model"))
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.analyze_btn.config(state=tk.NORMAL)
        self.status_var.set("Model loaded successfully - Ready to analyze")
        self.results_text.insert(tk.END, "✅ GSAT model loaded successfully!\n")
        self.results_text.insert(tk.END, "📝 Enter a SMILES string and click 'Analyze Toxicity' to begin.\n\n")
    
    def on_model_error(self, error_msg):
        """Called when model loading fails"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.status_var.set("Error loading model")
        messagebox.showerror("Model Loading Error", f"Failed to load GSAT model:\n{error_msg}")
        self.results_text.insert(tk.END, f"❌ Error loading model: {error_msg}\n")
    
    def on_example_selected(self, event):
        """Handle example molecule selection"""
        selection = self.example_var.get()
        if selection:
            smiles = selection.split(": ")[1]
            self.smiles_var.set(smiles)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
    
    def analyze_molecule(self):
        """Analyze the input SMILES molecule"""
        smiles = self.smiles_var.get().strip()
        if not smiles:
            messagebox.showwarning("Input Required", "Please enter a SMILES string")
            return
        
        if self.model_components is None:
            messagebox.showerror("Model Not Ready", "Model components are not loaded yet")
            return
        
        # Disable button and show progress
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_var.set("Analyzing molecule...")
        self.progress_bar.start()
        
        # Run analysis in background thread
        def analyze():
            try:
                output_dir = self.output_dir_var.get()
                original_dir = os.getcwd()
                os.chdir(output_dir)
                
                # Import matplotlib and set non-interactive backend to avoid GUI issues
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                
                # Run the complete analysis using the existing function (auto-creates organized directories)
                pred_lc50, atom_importance = predict_with_actual_model(smiles, show_plot=False, 
                                                                     save_image=True, output_dir=None)
                
                # Get the analysis directory and file information
                analysis_info = get_current_analysis_info()
                
                # Create molecule object for visualization
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                
                # Prepare results
                level_info = get_toxicity_level_info(pred_lc50)
                safe_smiles = smiles.replace('/', '_').replace('\\', '_').replace('[', '').replace(']', '')
                log_lc50 = -np.log10(pred_lc50)
                
                # Build correct file paths based on the organized directory structure
                analysis_dir = analysis_info.get('directory', '.')
                image_filename = f"clean_analysis_{safe_smiles}_{pred_lc50:.2f}.png"
                csv_filename = f"importance_data_{safe_smiles}_{pred_lc50:.2f}.csv"
                report_filename = f"analysis_{safe_smiles}_logLC50_{log_lc50:.2f}.txt"
                
                # Full paths for files in organized structure
                image_path = os.path.join(analysis_dir, "visualizations", image_filename)
                csv_path = os.path.join(analysis_dir, "data", csv_filename)
                report_path = os.path.join(analysis_dir, "reports", report_filename)
                
                results = {
                    'smiles': smiles,
                    'lc50': pred_lc50,
                    'atom_importance': atom_importance,
                    'mol': mol,
                    'level_info': level_info,
                    'image_file': image_path,
                    'csv_file': csv_path,
                    'report_file': report_path,
                    'analysis_directory': analysis_dir,
                    'data_file': analysis_info.get('data_file', None)
                }
                
                os.chdir(original_dir)  # Change back to original directory
                
                self.root.after(0, lambda: self.on_analysis_complete(results))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.on_analysis_error(error_msg))
        
        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()
    
    def on_analysis_complete(self, results):
        """Called when analysis completes successfully"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.analyze_btn.config(state=tk.NORMAL)
        
        # Store file paths and molecular data
        self.last_visualization = results['image_file']
        self.last_csv_file = results['csv_file']
        self.current_mol_data = results
        self.current_image_path = results['image_file']
        self.current_csv_path = results['csv_file']
        self.current_report_path = results['report_file']
        
        # Enable result buttons
        self.view_image_btn.config(state=tk.NORMAL)
        self.view_csv_btn.config(state=tk.NORMAL)
        self.open_image_btn.config(state=tk.NORMAL)
        self.open_csv_btn.config(state=tk.NORMAL)
        self.open_report_btn.config(state=tk.NORMAL)
        
        # Update embedded visualization by loading the PNG file
        # Use after() to ensure GUI is ready
        self.root.after(100, lambda: self.update_embedded_visualization(results))
        
        # Ensure the main window is focused and brought to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(200, lambda: self.root.attributes('-topmost', False))
        
        # Display results with eye-catching format
        self.results_text.delete(1.0, tk.END)
        
        # Clean header
        self.results_text.insert(tk.END, "🧬 ANALYSIS RESULTS\n")
        self.results_text.insert(tk.END, "=" * 25 + "\n\n")
        
        self.results_text.insert(tk.END, f"SMILES: {results['smiles']}\n\n")
        
        # Big toxicity result with enhanced visual impact
        symbol = results['level_info']['symbol']
        level = results['level_info']['level']
        lc50_val = results['lc50']
        log_lc50_val = -np.log10(lc50_val)
        
        self.results_text.insert(tk.END, f"🎯 LC50 PREDICTION: {lc50_val:.6f} [mol/L] = {log_lc50_val:.3f} [-log(mol/L)]\n")
        self.results_text.insert(tk.END, f"� TOXICITY: {symbol} {level}\n\n")
        
        # Files section
        self.results_text.insert(tk.END, "� OUTPUT FILES:\n")
        self.results_text.insert(tk.END, f"  🖼️  {results['image_file']}\n")
        self.results_text.insert(tk.END, f"  📊 {results['csv_file']}\n")
        safe_smiles_report = results['smiles'].replace('/', '_').replace('\\', '_').replace('[', '').replace(']', '')
        self.results_text.insert(tk.END, f"  📄 analysis_{safe_smiles_report}_{results['lc50']:.2f}.txt\n\n")
        
        # Add analysis directory information
        analysis_dir = results.get('analysis_directory', '.')
        if analysis_dir != '.':
            dir_name = os.path.basename(analysis_dir)
            self.results_text.insert(tk.END, f"📂 Created directory: {dir_name}/\n")
            self.results_text.insert(tk.END, f"   └── Files organized in subdirectories\n")
        
        self.results_text.insert(tk.END, "✅ ANALYSIS COMPLETE!\n")
        
        log_lc50_status = -np.log10(results['lc50'])
        self.status_var.set(f"Analysis complete - LC50: {results['lc50']:.6f} mol/L (-log: {log_lc50_status:.3f})")
    
    def on_analysis_error(self, error_msg):
        """Called when analysis fails"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.analyze_btn.config(state=tk.NORMAL)
        
        self.status_var.set("Analysis failed")
        messagebox.showerror("Analysis Error", f"Analysis failed:\n{error_msg}")
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"❌ Analysis Error:\n{error_msg}\n\n")
        self.results_text.insert(tk.END, "Please check your SMILES string and try again.\n")
    
    def view_visualization(self):
        """Open the generated visualization"""
        if self.last_visualization:
            filepath = os.path.join(self.output_dir_var.get(), self.last_visualization)
            if os.path.exists(filepath):
                if sys.platform.startswith('darwin'):  # macOS
                    os.system(f'open "{filepath}"')
                elif sys.platform.startswith('linux'):  # Linux
                    os.system(f'xdg-open "{filepath}"')
                elif sys.platform.startswith('win'):  # Windows
                    os.startfile(filepath)
            else:
                messagebox.showwarning("File Not Found", f"Visualization file not found: {filepath}")
        else:
            messagebox.showwarning("No Analysis", "No analysis has been run yet")
    
    def view_csv_data(self):
        """Open the generated CSV file"""
        print(f"🔍 VIEW CSV: Trying to open CSV file: {self.current_csv_path}")
        if self.current_csv_path and os.path.exists(self.current_csv_path):
            print(f"✅ VIEW CSV: CSV file exists, opening: {self.current_csv_path}")
            if sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{self.current_csv_path}"')
            elif sys.platform.startswith('linux'):  # Linux
                os.system(f'xdg-open "{self.current_csv_path}"')
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(self.current_csv_path)
        else:
            print(f"❌ VIEW CSV: CSV file not found: {self.current_csv_path}")
            messagebox.showwarning("File Not Found", f"CSV file not found: {self.current_csv_path}")
    
    def open_output_folder(self):
        """Open the output directory"""
        output_dir = self.output_dir_var.get()
        if os.path.exists(output_dir):
            if sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{output_dir}"')
            elif sys.platform.startswith('linux'):  # Linux
                os.system(f'xdg-open "{output_dir}"')
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(output_dir)
        else:
            messagebox.showwarning("Directory Not Found", "Output directory not found")
    
    def open_image_file(self):
        """Open the generated image file"""
        if self.current_image_path and os.path.exists(self.current_image_path):
            if sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{self.current_image_path}"')
            elif sys.platform.startswith('linux'):  # Linux
                os.system(f'xdg-open "{self.current_image_path}"')
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(self.current_image_path)
        else:
            messagebox.showwarning("File Not Found", "Image file not found")
    
    def open_csv_file(self):
        """Open the generated CSV file"""
        print(f"🔍 Trying to open CSV file: {self.current_csv_path}")
        if self.current_csv_path and os.path.exists(self.current_csv_path):
            print(f"✅ CSV file exists, opening: {self.current_csv_path}")
            if sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{self.current_csv_path}"')
            elif sys.platform.startswith('linux'):  # Linux
                os.system(f'xdg-open "{self.current_csv_path}"')
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(self.current_csv_path)
        else:
            print(f"❌ CSV file not found: {self.current_csv_path}")
            messagebox.showwarning("File Not Found", f"CSV file not found: {self.current_csv_path}")
    
    def open_report_file(self):
        """Open the generated report file"""
        if self.current_report_path and os.path.exists(self.current_report_path):
            if sys.platform.startswith('darwin'):  # macOS
                os.system(f'open "{self.current_report_path}"')
            elif sys.platform.startswith('linux'):  # Linux
                os.system(f'xdg-open "{self.current_report_path}"')
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(self.current_report_path)
        else:
            messagebox.showwarning("File Not Found", f"Report file not found: {self.current_report_path}")
    
    def clear_results(self):
        """Clear the results area"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Results cleared.\n")
        self.view_image_btn.config(state=tk.DISABLED)
        self.view_csv_btn.config(state=tk.DISABLED)
        self.open_image_btn.config(state=tk.DISABLED)
        self.open_csv_btn.config(state=tk.DISABLED)
        self.open_report_btn.config(state=tk.DISABLED)
        self.last_visualization = None
        self.last_csv_file = None
        self.current_mol_data = None
        self.current_image_path = None
        self.current_csv_path = None
        self.current_report_path = None
        self.show_placeholder_visualization()
        self.status_var.set("Ready")

    def browse_csv_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_file_var.set(filename)
            self.load_csv_columns(filename)

    def load_csv_columns(self, filename):
        """Load column names from CSV file"""
        try:
            import pandas as pd
            df = pd.read_csv(filename, nrows=0)  # Read just the header
            columns = list(df.columns)
            self.column_combo['values'] = columns
            if columns:
                self.column_combo.set(columns[0])  # Set first column as default
                self.batch_analyze_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")

    def batch_analyze(self):
        """Perform batch analysis on CSV file"""
        if not self.csv_file_var.get() or not self.column_var.get():
            messagebox.showwarning("Input Required", "Please select a CSV file and SMILES column")
            return

        self.batch_analyze_btn.config(state=tk.DISABLED)
        self.batch_clear_btn.config(state=tk.DISABLED)
        
        def analyze():
            try:
                import pandas as pd
                
                # Load CSV file
                df = pd.read_csv(self.csv_file_var.get())
                smiles_column = self.column_var.get()
                
                if smiles_column not in df.columns:
                    raise ValueError(f"Column '{smiles_column}' not found in CSV")
                
                # Get SMILES from the selected column
                smiles_list = df[smiles_column].dropna().tolist()
                total_molecules = len(smiles_list)
                
                self.root.after(0, lambda: self.batch_progress_frame.pack(fill=tk.X, pady=(10, 0)))
                self.root.after(0, lambda: self.batch_progress_var.set(f"Processing 0/{total_molecules} molecules..."))
                self.root.after(0, lambda: self.batch_progress_bar.config(maximum=total_molecules))
                
                # Initialize results storage
                batch_results = []
                
                # Change to output directory
                original_dir = os.getcwd()
                output_dir = self.output_dir_var.get()
                os.chdir(output_dir)
                
                # Process each SMILES
                for i, smiles in enumerate(smiles_list):
                    try:
                        # Update progress
                        self.root.after(0, lambda i=i: self.batch_progress_var.set(f"Processing {i+1}/{total_molecules}: {smiles[:30]}..."))
                        self.root.after(0, lambda i=i: self.batch_progress_bar.config(value=i))
                        
                        # Analyze molecule
                        pred_lc50, atom_importance = predict_with_actual_model(smiles, show_plot=False)
                        level_info = get_toxicity_level_info(pred_lc50)
                        
                        # Store results
                        result = {
                            'SMILES': smiles,
                            'LC50_Prediction': round(pred_lc50, 2),
                            'Toxicity_Level': level_info['level'],
                            'Toxicity_Symbol': level_info['symbol'],
                            'Status': 'Success'
                        }
                        batch_results.append(result)
                        
                    except Exception as e:
                        # Store error result
                        result = {
                            'SMILES': smiles,
                            'LC50_Prediction': 'Error',
                            'Toxicity_Level': 'Error',
                            'Toxicity_Symbol': '❌',
                            'Status': f'Failed: {str(e)}'
                        }
                        batch_results.append(result)
                
                os.chdir(original_dir)
                
                # Update progress to complete
                self.root.after(0, lambda: self.batch_progress_var.set(f"Completed {total_molecules} molecules!"))
                self.root.after(0, lambda: self.batch_progress_bar.config(value=total_molecules))
                
                # Pass results to UI update
                self.root.after(0, lambda: self.on_batch_complete(batch_results))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.on_batch_error(error_msg))
        
        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()

    def on_batch_complete(self, results):
        """Called when batch analysis completes"""
        self.batch_analyze_btn.config(state=tk.NORMAL)
        self.batch_clear_btn.config(state=tk.NORMAL)
        self.export_batch_btn.config(state=tk.NORMAL)
        
        # Store results for export
        self.batch_results_data = results
        
        # Display results summary
        self.batch_results_text.delete(1.0, tk.END)
        
        success_count = sum(1 for r in results if r['Status'] == 'Success')
        error_count = len(results) - success_count
        
        self.batch_results_text.insert(tk.END, "🚀" + "=" * 50 + "🚀\n")
        self.batch_results_text.insert(tk.END, "          BATCH ANALYSIS RESULTS\n")
        self.batch_results_text.insert(tk.END, "🚀" + "=" * 50 + "🚀\n\n")
        
        self.batch_results_text.insert(tk.END, f"📊 TOTAL PROCESSED: {len(results)} molecules\n")
        self.batch_results_text.insert(tk.END, f"✅ SUCCESSFUL: {success_count}\n")
        self.batch_results_text.insert(tk.END, f"❌ FAILED: {error_count}\n\n")
        
        # Show sample results
        self.batch_results_text.insert(tk.END, "📋 SAMPLE RESULTS (First 10):\n")
        self.batch_results_text.insert(tk.END, "-" * 60 + "\n")
        
        for i, result in enumerate(results[:10]):
            if result['Status'] == 'Success':
                self.batch_results_text.insert(tk.END, f"{i+1:2d}. {result['SMILES'][:20]:<20} | LC50: {result['LC50_Prediction']:>6}\n")
            else:
                self.batch_results_text.insert(tk.END, f"{i+1:2d}. {result['SMILES'][:20]:<20} | {result['Status']}\n")
        
        if len(results) > 10:
            self.batch_results_text.insert(tk.END, f"\n... and {len(results)-10} more results\n")
        
        self.batch_results_text.insert(tk.END, "\n✅ BATCH ANALYSIS COMPLETE!\n")
        self.batch_results_text.insert(tk.END, "📄 Click 'Export Batch Results' to save full results to CSV\n")

    def on_batch_error(self, error_msg):
        """Called when batch analysis fails"""
        self.batch_analyze_btn.config(state=tk.NORMAL)
        self.batch_clear_btn.config(state=tk.NORMAL)
        self.batch_progress_frame.pack_forget()
        
        self.batch_results_text.delete(1.0, tk.END)
        self.batch_results_text.insert(tk.END, f"❌ BATCH ANALYSIS FAILED\n\nError: {error_msg}\n")

    def clear_batch_results(self):
        """Clear batch results"""
        self.batch_results_text.delete(1.0, tk.END)
        self.batch_progress_frame.pack_forget()
        self.export_batch_btn.config(state=tk.DISABLED)
        self.batch_results_data = None

    def export_batch_results(self):
        """Export batch results to CSV"""
        if not hasattr(self, 'batch_results_data') or not self.batch_results_data:
            messagebox.showwarning("No Data", "No batch results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                import pandas as pd
                df = pd.DataFrame(self.batch_results_data)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Batch results exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")

    def setup_model_explanation_tab(self, parent):
        """Setup the comprehensive model explanation tab"""
        # Create scrollable frame for the explanation content
        canvas = tk.Canvas(parent, bg=self.bg_primary, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.bg_primary)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Main content frame with padding
        content_frame = tk.Frame(scrollable_frame, bg=self.bg_primary)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title section
        title_frame = tk.Frame(content_frame, bg=self.bg_secondary, relief='flat', bd=1)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(title_frame, text="🧠 GSAT Model: Scientific Workflow & Methodology", 
                font=('Segoe UI', 18, 'bold'), bg=self.bg_secondary, fg=self.accent_bright).pack(pady=15)
        
        tk.Label(title_frame, text="Graph Spatial-Temporal Attention Networks: A Multi-Modal Deep Learning Approach for Quantitative Structure-Activity Relationship (QSAR) Modeling in Toxicology", 
                font=('Segoe UI', 11), bg=self.bg_secondary, fg=self.text_secondary).pack(pady=(0, 15))
        
        # Architecture diagram display
        diagram_frame = tk.Frame(content_frame, bg=self.bg_secondary, relief='flat', bd=1)
        diagram_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(diagram_frame, text="📊 Architecture Diagram", 
                font=('Segoe UI', 14, 'bold'), bg=self.bg_secondary, fg=self.accent_bright).pack(pady=(15, 5))
        
        # Try to load and display the architecture diagram
        try:
            from PIL import Image, ImageTk
            import os
            
            diagram_path = "gsat_architecture_diagram.png"
            if os.path.exists(diagram_path):
                # Load and resize the diagram
                pil_image = Image.open(diagram_path)
                # Resize to fit in the explanation panel
                display_width = 800
                aspect_ratio = pil_image.height / pil_image.width
                display_height = int(display_width * aspect_ratio)
                
                resized_image = pil_image.resize((display_width, display_height), Image.LANCZOS)
                tk_image = ImageTk.PhotoImage(resized_image)
                
                diagram_label = tk.Label(diagram_frame, image=tk_image, bg=self.bg_secondary)
                diagram_label.image = tk_image  # Keep reference
                diagram_label.pack(pady=10)
                
                tk.Label(diagram_frame, text="Click to view architecture diagram in detail", 
                        font=('Segoe UI', 9), bg=self.bg_secondary, fg=self.text_secondary).pack(pady=(0, 15))
            else:
                # Create diagram button if image doesn't exist
                create_btn = tk.Button(diagram_frame, text="📊 Generate Architecture Diagram", 
                                     command=self.create_architecture_diagram,
                                     font=('Segoe UI', 10, 'bold'), bg=self.accent_bright, fg='black',
                                     relief='flat', bd=0, padx=20, pady=8, cursor='hand2')
                create_btn.pack(pady=(10, 15))
                
        except Exception as e:
            tk.Label(diagram_frame, text=f"Architecture diagram unavailable: {str(e)}", 
                    font=('Segoe UI', 9), bg=self.bg_secondary, fg=self.text_secondary).pack(pady=(0, 15))
        
        # Abstract & Methodology Section
        self.create_explanation_section(content_frame, "📋 Abstract & Research Methodology", 
            """OBJECTIVE: Development of a multi-modal graph neural network for quantitative prediction of acute aquatic toxicity (LC50) in emerging pollutants.

HYPOTHESIS: Integration of three complementary molecular representations through attention mechanisms will improve toxicity prediction accuracy compared to single-modal approaches.

METHODOLOGY:
• Multi-modal architecture combining 3D conformational, structural scaffold, and sequential molecular representations
• Graph attention networks with spatial-temporal encoding
• Cross-modal attention fusion for inter-representation information flow
• Supervised learning on curated LC50 toxicity datasets

MODEL SPECIFICATIONS:
• Architecture: Graph Spatial-Temporal Attention Network (GSAT)
• Parameters: 2,195,265 trainable parameters
• Performance: R² = 0.9633 (95% CI)
• Input: SMILES notation chemical structures
• Output: -log(LC50) with uncertainty quantification""")
        
        # Computational Workflow Section
        self.create_explanation_section(content_frame, "⚙️ Computational Workflow", 
            """STEP 1: MOLECULAR PREPROCESSING & FEATURE EXTRACTION
Input Processing Pipeline:
   1.1 SMILES Validation: Chemical structure parsing and normalization
   1.2 3D Conformer Generation: MMFF94 force field optimization
   1.3 Murcko Scaffold Extraction: Core framework identification
   1.4 Chemical Tokenization: Vocabulary-based sequence encoding

STEP 2: MULTI-MODAL REPRESENTATION LEARNING
Parallel Encoding Pathways:

   2.1 CONFORMER GRAPH ENCODER (G_conf)
       • Input: 3D atomic coordinates with distance matrix
       • Architecture: 3-layer Graph Attention Network
       • Attention mechanism: Distance-biased spatial attention
       • Output: h_conf ∈ R^(N×d) where N=atoms, d=embedding_dim

   2.2 SCAFFOLD GRAPH ENCODER (G_scaffold)  
       • Input: Murcko scaffold topology
       • Architecture: 3-layer Graph Attention Network
       • Focus: Core pharmacophore/toxicophore identification
       • Output: h_scaffold ∈ R^(M×d) where M=scaffold_atoms

   2.3 SEQUENCE ENCODER (S_seq)
       • Input: Tokenized SMILES representation
       • Architecture: 4-layer Transformer
       • Attention: Chemical grammar and syntax learning
       • Output: h_seq ∈ R^(L×d) where L=sequence_length""")
        
        # Mathematical Framework Section
        self.create_explanation_section(content_frame, "🔢 Mathematical Framework & Attention Mechanisms", 
            """STEP 3: CROSS-MODAL ATTENTION FUSION

Multi-Head Attention Function:
   Attention(Q,K,V) = softmax(QK^T/√d_k)V

Cross-Modal Fusion Architecture:
   3.1 Inter-Modal Attention (Layer 1):
       • Q_conf-scaffold = MultiHead(h_conf, h_scaffold, h_scaffold)
       • Q_scaffold-seq = MultiHead(h_scaffold, h_seq, h_seq)
       • Q_conf-seq = MultiHead(h_conf, h_seq, h_seq)

   3.2 Intra-Modal Self-Attention (Layer 2):
       • h_fused = Concat([Q_conf-scaffold, Q_scaffold-seq, Q_conf-seq])
       • h_final = LayerNorm(h_fused + FFN(h_fused))

Distance-Biased Attention (3D Conformer):
   A_ij = softmax((q_i·k_j)/√d_k + b_ij)
   where b_ij = -log(d_ij + ε) for atomic distance d_ij

STEP 4: TOXICITY PREDICTION LAYER
Final Prediction:
   ŷ = W_out · Pool(h_final) + b_out
   where Pool() = global attention pooling
   Output: -log(LC50) ∈ R with uncertainty σ²""")
        
        # Training & Optimization Section
        self.create_explanation_section(content_frame, "🎯 Training Protocol & Optimization", 
            """TRAINING METHODOLOGY:

Loss Function:
   L = MSE(ŷ, y) + λ₁·L_attention + λ₂·L_regularization
   where:
   • MSE: Mean Squared Error for LC50 prediction
   • L_attention: Attention consistency regularization
   • L_regularization: L2 weight decay (λ₂ = 1e-4)

Optimization Strategy:
   • Optimizer: AdamW with weight decay
   • Learning Rate: 1e-4 with cosine annealing
   • Batch Size: 32 molecular structures
   • Training Epochs: Early stopping with patience=10

Model Selection & Validation:
   • Stochastic Weight Averaging (SWA) for final model
   • Cross-validation on independent test sets
   • Performance Metrics: R², MAE, RMSE
   • Uncertainty Quantification: Monte Carlo Dropout

HYPERPARAMETER OPTIMIZATION:
   • d_model: 256 (embedding dimension)
   • n_heads: 8 (multi-head attention)
   • n_layers: 3 (graph), 4 (sequence), 2 (cross-modal)
   • dropout: 0.1 (regularization)
   • Hidden dimensions: [512, 256, 128] (prediction layer)""")
        
        # Interpretability & Analysis Section
        self.create_explanation_section(content_frame, "🔍 Model Interpretability & Mechanistic Analysis", 
            """ATTENTION-BASED MECHANISTIC INTERPRETATION:

Gradient-weighted Class Activation Mapping (Grad-CAM):
   • Atomic Importance: I_atom,i = Σ_k α_k · A_k,i
   • Bond Importance: I_bond,ij = (I_atom,i + I_atom,j) / 2
   where α_k = gradients of attention weights A_k

MOLECULAR FEATURE ATTRIBUTION:
   1. Atom-level Toxicophore Identification:
      • Attention weight aggregation across all heads
      • Statistical significance testing (p < 0.05)
      • Chemical interpretation of high-attention regions

   2. Bond-level Interaction Analysis:
      • Edge attention extraction from graph networks
      • Conjugation and aromaticity effect quantification
      • Structure-activity relationship elucidation

   3. Functional Group Impact Assessment:
      • Automatic substructure detection and scoring
      • SMARTS pattern matching for known toxicophores
      • Quantitative group contribution analysis

UNCERTAINTY QUANTIFICATION:
   • Epistemic Uncertainty: Model parameter uncertainty via dropout
   • Aleatoric Uncertainty: Data-inherent noise estimation
   • Prediction Intervals: 95% confidence bounds on LC50 predictions""")
        

        

        
        # Technical Implementation Section
        self.create_explanation_section(content_frame, "⚙️ Technical Implementation & Reproducibility", 
            """SOFTWARE ENVIRONMENT & DEPENDENCIES:

Core Framework:
   • PyTorch: 2.0.1 (automatic differentiation, GPU acceleration)
   • PyTorch Geometric: 2.3.1 (graph neural network primitives)
   • RDKit: 2023.03.2 (cheminformatics toolkit)
   • NumPy: 1.24.3, SciPy: 1.10.1 (numerical computing)

Molecular Processing Pipeline:
   • SMILES Standardization: RDKit canonicalization
   • 3D Conformer Generation: ETKDG algorithm with MMFF94 optimization
   • Graph Construction: Adjacency matrices with edge attributes
   • Chemical Tokenization: Custom vocabulary (2,048 tokens)

REPRODUCIBILITY PROTOCOLS:
   • Random Seed Control: Fixed seeds for all stochastic processes
   • Deterministic Operations: CuDNN deterministic mode enabled
   • Version Control: Git tracking for all code and hyperparameters
   • Environment Management: Conda environment specifications

COMPUTATIONAL INFRASTRUCTURE:
   • Hardware: NVIDIA Tesla V100 32GB GPU
   • CUDA Version: 11.8 with cuDNN 8.7
   • Memory Management: Gradient checkpointing for large molecules
   • Parallel Processing: DataLoader multiprocessing (8 workers)

CODE AVAILABILITY:
   • Open Source: Available on GitHub with MIT license
   • Documentation: Comprehensive API documentation
   • Testing Suite: Unit tests with 95% code coverage
   • Continuous Integration: Automated testing on multiple platforms""")

    def create_explanation_section(self, parent, title, content):
        """Create a formatted explanation section"""
        # Section frame
        section_frame = tk.Frame(parent, bg=self.bg_secondary, relief='flat', bd=1)
        section_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Section title
        title_frame = tk.Frame(section_frame, bg=self.bg_accent)
        title_frame.pack(fill=tk.X)
        
        tk.Label(title_frame, text=title, font=('Segoe UI', 14, 'bold'), 
                bg=self.bg_accent, fg='black', padx=15, pady=10).pack(anchor='w')
        
        # Section content
        content_frame = tk.Frame(section_frame, bg=self.bg_secondary)
        content_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Use Text widget for better formatting
        text_widget = tk.Text(content_frame, 
                             font=('Consolas', 10), 
                             bg=self.bg_secondary, 
                             fg=self.text_primary,
                             wrap=tk.WORD,
                             height=len(content.split('\n')) + 2,
                             state=tk.NORMAL,
                             relief='flat',
                             bd=0,
                             padx=10,
                             pady=10)
        
        # Insert content with formatting
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        text_widget.pack(fill=tk.X)

    def create_architecture_diagram(self):
        """Create and display the GSAT architecture diagram"""
        try:
            import subprocess
            import os
            
            # Run the diagram creation script
            subprocess.run([sys.executable, "create_architecture_diagram.py"], check=True)
            
            # Refresh the explanation tab to show the new diagram
            messagebox.showinfo("Success", "Architecture diagram created successfully!\nRefresh the Model Explanation tab to view it.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create architecture diagram:\n{str(e)}")

    def setup_similarity_analysis_tab(self, parent):
        """Setup the similarity analysis tab for chemical representation extraction"""
        # Create main container with modern styling
        main_container = tk.Frame(parent, bg=self.bg_primary)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        header_frame = tk.Frame(main_container, bg=self.bg_secondary, relief='flat', bd=0)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        border_accent = tk.Frame(header_frame, bg=self.bg_accent, height=3)
        border_accent.pack(fill=tk.X)
        
        header_content = tk.Frame(header_frame, bg=self.bg_secondary, padx=20, pady=15)
        header_content.pack(fill=tk.X)
        
        title_label = tk.Label(header_content, text="🔬 CHEMICAL REPRESENTATION EXTRACTION", 
                              font=('Segoe UI', 16, 'bold'), 
                              bg=self.bg_secondary, fg=self.accent_bright)
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(header_content, 
                                 text="Tanimoto Similarity Analysis • Representative Selection • Dataset Reduction", 
                                 font=('Segoe UI', 11), 
                                 bg=self.bg_secondary, fg=self.text_secondary)
        subtitle_label.pack(anchor='w', pady=(5, 0))
        
        # Input section
        input_frame = tk.Frame(main_container, bg=self.bg_secondary, relief='flat', bd=0)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        input_border = tk.Frame(input_frame, bg='#f9e2af', height=3)  # Yellow accent
        input_border.pack(fill=tk.X)
        
        input_content = tk.Frame(input_frame, bg=self.bg_secondary, padx=20, pady=15)
        input_content.pack(fill=tk.X)
        
        # File input
        tk.Label(input_content, text="📁 Input SMILES Dataset:", 
                font=('Segoe UI', 12, 'bold'), 
                bg=self.bg_secondary, fg=self.text_primary).pack(anchor='w', pady=(0, 8))
        
        file_frame = tk.Frame(input_content, bg=self.bg_secondary)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.similarity_file_var = tk.StringVar()
        file_entry = tk.Entry(file_frame, textvariable=self.similarity_file_var, 
                             font=('Consolas', 11), width=60,
                             bg=self.bg_primary, fg=self.text_primary,
                             relief='flat', bd=5)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = tk.Button(file_frame, text="📂 Browse", 
                              command=self.browse_similarity_file,
                              font=('Segoe UI', 10, 'bold'),
                              bg=self.bg_accent, fg='black',
                              relief='flat', padx=15, pady=8)
        browse_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Add info label about expected format
        info_label = tk.Label(input_content, 
                             text="💡 Expected format: CSV with columns 'SMILES', 'Name' (optional), 'LC50' (optional)", 
                             font=('Segoe UI', 9), 
                             bg=self.bg_secondary, fg=self.text_secondary)
        info_label.pack(anchor='w', pady=(5, 10))
        
        # Parameters section
        params_frame = tk.Frame(input_content, bg=self.bg_secondary)
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Similarity threshold
        tk.Label(params_frame, text="🎯 Similarity Threshold:", 
                font=('Segoe UI', 11, 'bold'), 
                bg=self.bg_secondary, fg=self.text_primary).pack(anchor='w')
        
        threshold_frame = tk.Frame(params_frame, bg=self.bg_secondary)
        threshold_frame.pack(anchor='w', pady=(5, 10))
        
        self.similarity_threshold_var = tk.DoubleVar(value=0.7)
        threshold_scale = tk.Scale(threshold_frame, from_=0.1, to=0.9, resolution=0.1,
                                  orient=tk.HORIZONTAL, variable=self.similarity_threshold_var,
                                  bg=self.bg_secondary, fg=self.text_primary,
                                  font=('Segoe UI', 10), length=300)
        threshold_scale.pack(side=tk.LEFT)
        
        threshold_label = tk.Label(threshold_frame, textvariable=self.similarity_threshold_var,
                                  font=('Segoe UI', 11, 'bold'),
                                  bg=self.bg_secondary, fg=self.accent_bright)
        threshold_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Analysis button
        analyze_similarity_btn = tk.Button(input_content, text="🧪 ANALYZE SIMILARITY", 
                                          command=self.run_similarity_analysis,
                                          font=('Segoe UI', 12, 'bold'),
                                          bg=self.success_color, fg='black',
                                          relief='flat', padx=20, pady=12)
        analyze_similarity_btn.pack(pady=(10, 0))
        
        # Results section
        results_frame = tk.Frame(main_container, bg=self.bg_secondary, relief='flat', bd=0)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        results_border = tk.Frame(results_frame, bg='#a6e3a1', height=3)  # Green accent
        results_border.pack(fill=tk.X)
        
        results_content = tk.Frame(results_frame, bg=self.bg_secondary, padx=20, pady=15)
        results_content.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(results_content, text="📊 SIMILARITY ANALYSIS RESULTS", 
                font=('Segoe UI', 14, 'bold'), 
                bg=self.bg_secondary, fg=self.accent_bright).pack(anchor='w', pady=(0, 15))
        
        # Create notebook for results visualization
        results_notebook = ttk.Notebook(results_content)
        results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Statistics tab
        stats_frame = tk.Frame(results_notebook, bg=self.bg_secondary)
        results_notebook.add(stats_frame, text="📈 Statistics")
        
        self.similarity_stats_text = tk.Text(stats_frame, font=('Consolas', 10), 
                                            bg=self.bg_primary, fg=self.text_primary,
                                            wrap=tk.WORD, height=15, state=tk.DISABLED)
        stats_scrollbar = tk.Scrollbar(stats_frame, orient="vertical", 
                                      command=self.similarity_stats_text.yview)
        self.similarity_stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.similarity_stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Visualization tab
        viz_frame = tk.Frame(results_notebook, bg=self.bg_secondary)
        results_notebook.add(viz_frame, text="🎨 Visualization")
        
        # Create container for matplotlib
        viz_container = tk.Frame(viz_frame, bg=self.bg_secondary)
        viz_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure for similarity visualization
        self.similarity_fig = Figure(figsize=(10, 6), facecolor='#313244', dpi=80)
        self.similarity_canvas = FigureCanvasTk(self.similarity_fig, viz_container)
        
        # Pack the canvas widget
        canvas_widget = self.similarity_canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas_widget.configure(bg='#313244')
        
        # Add navigation toolbar
        toolbar_frame = tk.Frame(viz_container, bg=self.bg_secondary)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
        try:
            toolbar = NavigationToolbar2Tk(self.similarity_canvas, toolbar_frame)
            toolbar.update()
            # Style the toolbar
            toolbar.configure(bg=self.bg_secondary)
        except Exception as e:
            print(f"Warning: Could not create navigation toolbar: {e}")
        
        # Initialize the plot with placeholder
        self.show_similarity_placeholder()
        
        # Force initial draw
        self.similarity_canvas.draw()
        self.root.update_idletasks()
        
        # Representatives tab
        rep_frame = tk.Frame(results_notebook, bg=self.bg_secondary)
        results_notebook.add(rep_frame, text="🎪 Representatives")
        
        self.representatives_text = tk.Text(rep_frame, font=('Consolas', 10), 
                                           bg=self.bg_primary, fg=self.text_primary,
                                           wrap=tk.WORD, height=15, state=tk.DISABLED)
        rep_scrollbar = tk.Scrollbar(rep_frame, orient="vertical", 
                                    command=self.representatives_text.yview)
        self.representatives_text.configure(yscrollcommand=rep_scrollbar.set)
        
        self.representatives_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rep_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export buttons
        export_frame = tk.Frame(results_content, bg=self.bg_secondary)
        export_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.export_similarity_btn = tk.Button(export_frame, text="💾 Export Results", 
                                              command=self.export_similarity_results,
                                              font=('Segoe UI', 10, 'bold'),
                                              bg='#f9e2af', fg='black',
                                              relief='flat', padx=15, pady=8,
                                              state=tk.DISABLED)
        self.export_similarity_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Network visualization button
        self.network_viz_btn = tk.Button(export_frame, text="🕸️ Network View", 
                                        command=self.create_network_visualization,
                                        font=('Segoe UI', 10, 'bold'),
                                        bg='#94e2d5', fg='black',
                                        relief='flat', padx=15, pady=8,
                                        state=tk.DISABLED)
        self.network_viz_btn.pack(side=tk.LEFT)
        
        self.similarity_analyzer = None
        self.similarity_results = None
    
    def show_similarity_placeholder(self):
        """Show placeholder in similarity visualization"""
        try:
            self.similarity_fig.clear()
            ax = self.similarity_fig.add_subplot(111)
            
            # Create placeholder text
            placeholder_text = ('🔬 SIMILARITY ANALYSIS VISUALIZATION\n\n'
                              'Load a SMILES dataset and click "ANALYZE SIMILARITY"\n'
                              'to view Tanimoto similarity matrices and cluster distributions')
            
            ax.text(0.5, 0.5, placeholder_text, 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14, color='#cdd6f4',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#1e1e2e', 
                             edgecolor='#89b4fa', linewidth=2))
            
            # Style the axes
            ax.set_facecolor('#1e1e2e')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Style the figure
            self.similarity_fig.patch.set_facecolor('#313244')
            self.similarity_fig.tight_layout()
            
            # Force draw and update
            self.similarity_canvas.draw()
            self.similarity_canvas.flush_events()
            
            print("✅ Similarity placeholder displayed")
            
        except Exception as e:
            print(f"❌ Error showing similarity placeholder: {e}")
    
    def browse_similarity_file(self):
        """Browse for SMILES dataset file"""
        filename = filedialog.askopenfilename(
            title="Select SMILES Dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.similarity_file_var.set(filename)
    
    def run_similarity_analysis(self):
        """Run similarity analysis on the input dataset"""
        file_path = self.similarity_file_var.get().strip()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid SMILES dataset file.")
            return
        
        try:
            # Read SMILES from file
            import pandas as pd
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                
                # Find SMILES column
                smiles_columns = [col for col in df.columns if 'smiles' in col.lower()]
                if smiles_columns:
                    smiles_list = df[smiles_columns[0]].dropna().tolist()
                else:
                    # Use first column if no SMILES column found
                    smiles_list = df.iloc[:, 0].dropna().tolist()
                
                # Find compound names
                name_columns = [col for col in df.columns if any(x in col.lower() for x in ['name', 'compound', 'id'])]
                compound_names = df[name_columns[0]].tolist() if name_columns else None
                
                # Find LC50 values
                lc50_columns = [col for col in df.columns if 'lc50' in col.lower() or 'toxicity' in col.lower()]
                lc50_values = None
                if lc50_columns:
                    try:
                        lc50_values = pd.to_numeric(df[lc50_columns[0]], errors='coerce').tolist()
                        print(f"📊 Found LC50 data in column: {lc50_columns[0]}")
                    except Exception as e:
                        print(f"⚠️ Could not parse LC50 values: {e}")
                        lc50_values = None
                
            else:
                # Assume text file with one SMILES per line
                with open(file_path, 'r') as f:
                    smiles_list = [line.strip() for line in f if line.strip()]
                compound_names = None
                lc50_values = None
            
            if len(smiles_list) < 2:
                messagebox.showerror("Error", "Dataset must contain at least 2 valid SMILES.")
                return
            
            # Initialize analyzer
            threshold = self.similarity_threshold_var.get()
            self.similarity_analyzer = SimilarityRepresentativeAnalyzer(
                similarity_threshold=threshold
            )
            
            # Show progress
            self.similarity_stats_text.config(state=tk.NORMAL)
            self.similarity_stats_text.delete(1.0, tk.END)
            self.similarity_stats_text.insert(tk.END, "🔄 Running similarity analysis...\n")
            self.similarity_stats_text.config(state=tk.DISABLED)
            self.similarity_stats_text.update()
            
            # Run analysis in thread to prevent GUI freezing
            def run_analysis():
                try:
                    results = self.similarity_analyzer.analyze_dataset(
                        smiles_list, 
                        compound_names=compound_names, 
                        lc50_values=lc50_values
                    )
                    self.root.after(0, lambda: self.on_similarity_analysis_complete(results))
                except Exception as e:
                    self.root.after(0, lambda: self.on_similarity_analysis_error(str(e)))
            
            threading.Thread(target=run_analysis, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read dataset: {str(e)}")
    
    def on_similarity_analysis_complete(self, results):
        """Handle completion of similarity analysis"""
        try:
            self.similarity_results = results
            
            print(f"🎯 Analysis completed: {results['statistics']['n_representatives']} representatives")
            
            # Update statistics
            self.display_similarity_statistics(results)
            
            # Create visualizations - force update
            print("📊 Creating similarity visualizations...")
            self.create_similarity_visualizations()
            
            # Display representatives
            self.display_representatives(results)
            
            # Enable export button
            self.export_similarity_btn.config(state=tk.NORMAL)
            self.network_viz_btn.config(state=tk.NORMAL)
            
            # Force GUI update
            self.root.update_idletasks()
            
            messagebox.showinfo("Success", 
                              f"Similarity analysis completed!\n"
                              f"Representatives: {results['statistics']['n_representatives']}\n"
                              f"Redundant compounds: {results['statistics']['n_redundant']}\n"
                              f"Dataset reduction: {(1-results['statistics']['reduction_ratio'])*100:.1f}%")
            
        except Exception as e:
            print(f"❌ Error in analysis completion: {e}")
            self.on_similarity_analysis_error(str(e))
    
    def on_similarity_analysis_error(self, error_msg):
        """Handle similarity analysis error"""
        self.similarity_stats_text.config(state=tk.NORMAL)
        self.similarity_stats_text.delete(1.0, tk.END)
        self.similarity_stats_text.insert(tk.END, f"❌ Error: {error_msg}\n")
        self.similarity_stats_text.config(state=tk.DISABLED)
        messagebox.showerror("Analysis Error", f"Similarity analysis failed: {error_msg}")
    
    def display_similarity_statistics(self, results):
        """Display similarity analysis statistics"""
        stats = results['statistics']
        
        stats_text = f"""🧪 CHEMICAL REPRESENTATION EXTRACTION RESULTS
{'='*60}

📊 DATASET STATISTICS:
   • Total Input Compounds: {stats['total_input_compounds']:,}
   • Valid Compounds: {stats['valid_compounds']:,}
   • Chemical Clusters: {stats['n_clusters']:,}
   • Representative Compounds: {stats['n_representatives']:,}
   • Redundant Compounds: {stats['n_redundant']:,}

📈 SIMILARITY ANALYSIS:
   • Tanimoto Threshold: {stats['similarity_threshold']:.2f}
   • Average Cluster Size: {stats['avg_cluster_size']:.2f}
   • Dataset Reduction: {(1-stats['reduction_ratio'])*100:.1f}%
   • Compression Ratio: {1/stats['reduction_ratio']:.2f}:1

🔬 TANIMOTO COEFFICIENT FORMULA:
   T(A,B) = |A ∩ B| / (|A| + |B| - |A ∩ B|)
   
   Where:
   • A, B = Binary molecular fingerprints
   • T = Similarity coefficient (0-1)
   • 0 = No similarity
   • 1 = Identical structures

🎯 CLUSTERING METHODOLOGY:
   • Fingerprint Type: Morgan fingerprints (radius=2)
   • Fingerprint Size: 2048 bits
   • Clustering Method: Agglomerative clustering
   • Distance Metric: 1 - Tanimoto similarity
   • Representative Selection: Highest average intra-cluster similarity

💡 APPLICATIONS:
   • Reduces computational costs
   • Maintains chemical diversity
   • Enables efficient dataset splitting
   • Supports model training optimization
"""
        
        self.similarity_stats_text.config(state=tk.NORMAL)
        self.similarity_stats_text.delete(1.0, tk.END)
        self.similarity_stats_text.insert(tk.END, stats_text)
        self.similarity_stats_text.config(state=tk.DISABLED)
    
    def create_similarity_visualizations(self):
        """Create similarity analysis visualizations"""
        if not self.similarity_analyzer or not self.similarity_results:
            return
        
        try:
            print("🎨 Starting enhanced visualization creation...")
            self.similarity_fig.clear()
            
            cluster_sizes = [rep['cluster_size'] for rep in self.similarity_results['representatives']]
            n_compounds = len(self.similarity_analyzer.smiles_list)
            
            print(f"📊 Creating visualization for {n_compounds} compounds, {len(cluster_sizes)} clusters")
            
            # Get LC50 values if available
            lc50_values = self.similarity_results.get('lc50_values', None)
            compound_names = self.similarity_results.get('compound_names', None)
            
            # Create enhanced heatmap with network visualization
            n_display = min(25, n_compounds)
            sim_subset = self.similarity_analyzer.similarity_matrix[:n_display, :n_display]
            
            # Check if we have similarity matrix
            has_similarity_matrix = (hasattr(self.similarity_analyzer, 'similarity_matrix') and 
                                   self.similarity_analyzer.similarity_matrix is not None)
            
            if has_similarity_matrix:
                print("📈 Creating enhanced similarity heatmap with LC50 integration")
                
                # Enhanced similarity heatmap
                ax1 = self.similarity_fig.add_subplot(111)
                
                # Create the heatmap
                im = ax1.imshow(sim_subset, cmap='viridis', vmin=0, vmax=1, aspect='equal')
                
                # Add similarity values as text for high similarities
                for i in range(n_display):
                    for j in range(n_display):
                        if sim_subset[i, j] > 0.8 and i != j:
                            ax1.text(j, i, f'{sim_subset[i, j]:.2f}', 
                                    ha='center', va='center', color='white', 
                                    fontsize=8, fontweight='bold')
                
                # Styling
                ax1.set_xlabel('Compound Index', color='#cdd6f4', fontsize=12)
                ax1.set_ylabel('Compound Index', color='#cdd6f4', fontsize=12)
                
                # Title with LC50 information
                title = f'Tanimoto Similarity Heatmap ({n_display} compounds)'
                if lc50_values:
                    title += f'\nWith LC50 Toxicity Data'
                ax1.set_title(title, color='#cdd6f4', fontsize=14, fontweight='bold')
                ax1.set_facecolor('#1e1e2e')
                ax1.tick_params(colors='#cdd6f4')
                
                # Add colorbar
                try:
                    cbar = self.similarity_fig.colorbar(im, ax=ax1, shrink=0.8, aspect=30)
                    cbar.set_label('Tanimoto Similarity Coefficient', color='#cdd6f4', fontsize=11)
                    cbar.ax.yaxis.set_tick_params(color='#cdd6f4')
                    cbar.ax.yaxis.label.set_color('#cdd6f4')
                    
                    # Style colorbar ticks
                    for label in cbar.ax.yaxis.get_ticklabels():
                        label.set_color('#cdd6f4')
                except Exception as e:
                    print(f"Warning: Could not create colorbar: {e}")
                
                # Add information box
                info_text = f"Compounds: {n_display}/{n_compounds}\n"
                info_text += f"Threshold: {self.similarity_analyzer.similarity_threshold}\n"
                info_text += f"Clusters: {len(cluster_sizes)}\n"
                if lc50_values:
                    valid_lc50 = [v for v in lc50_values[:n_display] if v and v > 0]
                    if valid_lc50:
                        info_text += f"LC50 range: {min(valid_lc50):.3f} - {max(valid_lc50):.3f}"
                
                ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=10, color='#cdd6f4',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='#1e1e2e', 
                                 edgecolor='#89b4fa', alpha=0.9, linewidth=2))
                
                # Add grid for better readability
                ax1.set_xticks(range(0, n_display, max(1, n_display//10)))
                ax1.set_yticks(range(0, n_display, max(1, n_display//10)))
                ax1.grid(True, alpha=0.2, color='#cdd6f4', linewidth=0.5)
                
            else:
                print("📊 Creating cluster distribution plot")
                # Fallback to cluster distribution
                ax = self.similarity_fig.add_subplot(111)
                
                if len(cluster_sizes) > 0:
                    max_size = max(cluster_sizes)
                    bins = list(range(1, max_size + 2)) if max_size > 1 else [0.5, 1.5, 2.5]
                    
                    n, bins_used, patches = ax.hist(cluster_sizes, bins=bins, alpha=0.8, 
                                                   color='#89b4fa', edgecolor='#1e1e2e', linewidth=1)
                    ax.set_xlabel('Cluster Size', color='#cdd6f4', fontsize=12)
                    ax.set_ylabel('Number of Clusters', color='#cdd6f4', fontsize=12)
                    ax.set_title(f'Chemical Cluster Distribution\n({n_compounds} compounds → {len(cluster_sizes)} clusters)', 
                                color='#cdd6f4', fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3, color='#cdd6f4')
                    ax.set_facecolor('#1e1e2e')
                    ax.tick_params(colors='#cdd6f4')
                    
                    # Add value labels on bars
                    for i, v in enumerate(n):
                        if v > 0:
                            ax.text(bins_used[i] + (bins_used[i+1] - bins_used[i])/2, v + 0.05, 
                                   int(v), ha='center', va='bottom', color='#cdd6f4')
                else:
                    ax.text(0.5, 0.5, 'No cluster data available', 
                           ha='center', va='center', transform=ax.transAxes,
                           color='#f38ba8', fontsize=14)
            
            # Style the figure
            self.similarity_fig.patch.set_facecolor('#313244')
            
            # Adjust layout and refresh
            self.similarity_fig.tight_layout(pad=2.0)
            
            # Force canvas update
            print("🖼️ Drawing enhanced visualization...")
            self.similarity_canvas.draw()
            self.similarity_canvas.flush_events()
            
            # Force GUI update
            self.root.update_idletasks()
            self.root.update()
            
            print(f"✅ Enhanced similarity visualization completed successfully!")
            
        except Exception as e:
            print(f"❌ Error creating similarity visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def create_network_visualization(self):
        """Create network-style similarity visualization"""
        if not self.similarity_analyzer or not self.similarity_results:
            tk.messagebox.showwarning("No Data", "Please run similarity analysis first.")
            return
        
        try:
            print("🕸️ Creating network similarity visualization...")
            
            # Get LC50 values if available
            lc50_values = self.similarity_results.get('lc50_values', None)
            
            # Clear current figure and create network visualization
            self.similarity_fig.clear()
            
            # Use the analyzer's network visualization method
            network_fig = self.similarity_analyzer.create_network_similarity_map(
                results=self.similarity_results,
                lc50_values=lc50_values
            )
            
            if network_fig:
                # Copy the network plot to our GUI figure
                network_ax = network_fig.gca()
                
                # Create new axes in our figure
                ax = self.similarity_fig.add_subplot(111)
                
                # Copy the network plot content
                for line in network_ax.lines:
                    ax.plot(line.get_xdata(), line.get_ydata(), 
                           color=line.get_color(), linewidth=line.get_linewidth(), 
                           alpha=line.get_alpha())
                
                # Copy scatter plots (nodes)
                for collection in network_ax.collections:
                    if hasattr(collection, 'get_offsets'):
                        offsets = collection.get_offsets()
                        if len(offsets) > 0:
                            colors = collection.get_facecolors()
                            sizes = collection.get_sizes()
                            ax.scatter(offsets[:, 0], offsets[:, 1], 
                                     c=colors, s=sizes, alpha=collection.get_alpha())
                
                # Copy styling
                ax.set_xlim(network_ax.get_xlim())
                ax.set_ylim(network_ax.get_ylim())
                ax.axis('off')
                ax.set_facecolor('white')
                
                # Add title
                n_compounds = len(self.similarity_analyzer.smiles_list)
                title = f'Chemical Similarity Network ({n_compounds} compounds)'
                if lc50_values:
                    title += '\nColor: LC50 Toxicity | Edges: Tanimoto Similarity'
                ax.set_title(title, color='#cdd6f4', fontsize=14, fontweight='bold', pad=20)
                
                # Style the figure
                self.similarity_fig.patch.set_facecolor('#313244')
                self.similarity_fig.tight_layout()
                
                # Update canvas
                self.similarity_canvas.draw()
                self.similarity_canvas.flush_events()
                
                # Close the temporary figure
                plt.close(network_fig)
                
                print("✅ Network visualization created successfully!")
                
            else:
                tk.messagebox.showerror("Error", "Failed to create network visualization. NetworkX may be required.")
                
        except Exception as e:
            print(f"❌ Error creating network visualization: {e}")
            tk.messagebox.showerror("Error", f"Failed to create network visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Show error message in plot
            try:
                self.similarity_fig.clear()
                ax = self.similarity_fig.add_subplot(111)
                ax.text(0.5, 0.5, f'❌ Visualization Error:\n{str(e)}\n\nCheck console for details', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12, color='#f38ba8',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='#1e1e2e', 
                                 edgecolor='#f38ba8', linewidth=2))
                ax.set_facecolor('#1e1e2e')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                self.similarity_fig.patch.set_facecolor('#313244')
                self.similarity_canvas.draw()
                self.similarity_canvas.flush_events()
            except Exception as e2:
                print(f"❌ Error showing error message: {e2}")
    
    def display_representatives(self, results):
        """Display representative compounds information"""
        rep_text = f"🎪 REPRESENTATIVE COMPOUNDS\n{'='*50}\n\n"
        
        for i, rep in enumerate(results['representatives'][:20], 1):  # Show first 20
            rep_text += f"Representative #{i}:\n"
            rep_text += f"   • SMILES: {rep['representative_smiles']}\n"
            rep_text += f"   • Cluster ID: {rep['cluster_id']}\n"
            rep_text += f"   • Cluster Size: {rep['cluster_size']}\n"
            rep_text += f"   • Avg Similarity: {rep['avg_similarity']:.3f}\n\n"
        
        if len(results['representatives']) > 20:
            rep_text += f"... and {len(results['representatives']) - 20} more representatives\n\n"
        
        rep_text += f"🗂️ REDUNDANT COMPOUNDS (First 10):\n{'-'*40}\n"
        for i, red in enumerate(results['redundant_compounds'][:10], 1):
            rep_text += f"{i}. {red['smiles']} → Representative: {red['representative_smiles']}\n"
        
        if len(results['redundant_compounds']) > 10:
            rep_text += f"... and {len(results['redundant_compounds']) - 10} more redundant compounds\n"
        
        self.representatives_text.config(state=tk.NORMAL)
        self.representatives_text.delete(1.0, tk.END)
        self.representatives_text.insert(tk.END, rep_text)
        self.representatives_text.config(state=tk.DISABLED)
    
    def export_similarity_results(self):
        """Export similarity analysis results"""
        if not self.similarity_results:
            messagebox.showerror("Error", "No results to export.")
            return
        
        try:
            output_dir = filedialog.askdirectory(title="Select Export Directory")
            if not output_dir:
                return
            
            # Create timestamped directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(output_dir, f"similarity_analysis_{timestamp}")
            os.makedirs(export_path, exist_ok=True)
            
            # Export results using the analyzer's export method
            exported_files = self.similarity_analyzer.export_results(export_path)
            
            # Create summary report
            summary_path = os.path.join(export_path, "analysis_summary.txt")
            with open(summary_path, 'w') as f:
                stats = self.similarity_results['statistics']
                f.write("CHEMICAL REPRESENTATION EXTRACTION SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("DATASET STATISTICS:\n")
                f.write(f"  Total Input Compounds: {stats['total_input_compounds']}\n")
                f.write(f"  Valid Compounds: {stats['valid_compounds']}\n")
                f.write(f"  Chemical Clusters: {stats['n_clusters']}\n")
                f.write(f"  Representative Compounds: {stats['n_representatives']}\n")
                f.write(f"  Redundant Compounds: {stats['n_redundant']}\n")
                f.write(f"  Similarity Threshold: {stats['similarity_threshold']}\n")
                f.write(f"  Dataset Reduction: {(1-stats['reduction_ratio'])*100:.1f}%\n\n")
                f.write("EXPORTED FILES:\n")
                for file_type, file_path in exported_files.items():
                    f.write(f"  {file_type}: {os.path.basename(file_path)}\n")
            
            messagebox.showinfo("Export Complete", 
                              f"Similarity analysis results exported to:\n{export_path}\n\n"
                              f"Files exported: {len(exported_files) + 1}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")

def main():
    """Main function to run the GUI"""
    try:
        # Set environment variable for OpenMP
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        print("🔧 Initializing GUI...")
        root = tk.Tk()
        
        print("🎨 Creating application interface...")
        app = SMILESToxicityGUI(root)
        
        # Center the window
        print("📍 Positioning window...")
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        print("✅ GUI ready - starting main loop...")
        root.mainloop()
        print("👋 GUI closed normally")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all required packages are installed:")
        print("   - tkinter (usually comes with Python)")
        print("   - rdkit")
        print("   - torch, torch_geometric")
        print("   - matplotlib, pandas, numpy")
        input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\n⚠️  Application terminated by user")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print("   Full traceback:")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()