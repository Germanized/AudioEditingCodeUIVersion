import sys
import subprocess
import os
import importlib.util
import time
import re
import traceback
import warnings
from colorama import Fore, Style, init


init(autoreset=True)


warnings.filterwarnings("ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio")


from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLabel, QTextEdit, QFileDialog, QProgressBar,
                             QMessageBox, QComboBox)
from PyQt6.QtGui import (QPalette, QColor, QPainter, QFont, QPainterPath, QPen)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QSize, QObject


try:
    import pathlib
    _gui_script_dir = pathlib.Path(__file__).parent.resolve()
    _processing_script_name = "main_run.py"
    PROCESSING_SCRIPT = str(_gui_script_dir / _processing_script_name)
except ImportError:
    PROCESSING_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_run.py")
OUTPUT_PATH = "output/"
SPLASH_DURATION = 3000
MODEL_CHOICES = {
    "AudioLDM2 (cvssp/audioldm2)": "cvssp/audioldm2",
    "AudioLDM2 - Large": "cvssp/audioldm2-large",
    "AudioLDM2 - Music": "cvssp/audioldm2-music",
    "AudioLDM - Small v2": "cvssp/audioldm-s-full-v2",
    "AudioLDM - Large": "cvssp/audioldm-l-full",
    "Tango - Music Caps": "declare-lab/tango-full-ft-audio-music-caps",
    "Tango - AudioCaps": "declare-lab/tango-full-ft-audiocaps",
    "Stable Audio Open 1.0 (Requires HF_TOKEN)": "stabilityai/stable-audio-open-1.0",
}
DEFAULT_MODEL = "AudioLDM2 (cvssp/audioldm2)"
STABLE_AUDIO_ID = "stabilityai/stable-audio-open-1.0"


class ModernSplashScreen(QWidget):
    
    def __init__(self, text="Loading...", width=450, height=180, corner_radius=15): super().__init__(); self.text=text; self._width=width; self._height=height; self.corner_radius=corner_radius; self.setWindowFlags(Qt.WindowType.SplashScreen|Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint); self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground); self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose); self.setFixedSize(QSize(self._width,self._height)); self.background_color=QColor(40,42,45); self.text_color=QColor(220,220,220); self.font=QFont("Segoe UI",16); self.font.setBold(True)
    def paintEvent(self, event): painter=QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing,True); painter.setRenderHint(QPainter.RenderHint.TextAntialiasing,True); path=QPainterPath(); rect=QRectF(0,0,self.width(),self.height()); path.addRoundedRect(rect,self.corner_radius,self.corner_radius); painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(self.background_color); painter.drawPath(path); painter.setPen(QPen(self.text_color)); painter.setFont(self.font); painter.drawText(rect,Qt.AlignmentFlag.AlignCenter,self.text); painter.end()


class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    PROGRESS_RE = re.compile(r"(\d+)%\|")  

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None
        self._is_running = True

    def run(self):
        stdout_full = []
        stderr_full = []
        last_progress = -1
        try:
            self.status_signal.emit("Status: Processing starting... (NOT FROZEN JUST WAIT)")
            self.progress_signal.emit(0)

            
            output_dir = os.path.dirname(OUTPUT_PATH)
            if not output_dir:
                output_dir = "."
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            print(f"{Fore.CYAN}--- Starting Subprocess ---")
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True,
                env=env,
                bufsize=1,  
            )
            self.progress_signal.emit(10)
            print(f"{Fore.CYAN}--- Subprocess Launched, Reading Output Lines ---")

            
            while self._is_running and self.process.poll() is None:
                stdout_line = self.process.stdout.readline()
                stderr_line = self.process.stderr.readline()

                if stdout_line:
                    line_strip = stdout_line.strip()
                    print(f"{Fore.GREEN}STDOUT: {line_strip}")
                    self.log_signal.emit(f"OUT: {line_strip}")
                    stdout_full.append(line_strip)

                if stderr_line:
                    line_strip = stderr_line.strip()
                    print(f"{Fore.RED}STDERR: {line_strip}")
                    self.log_signal.emit(f"ERR: {line_strip}")
                    stderr_full.append(line_strip)

                    
                    match = self.PROGRESS_RE.search(line_strip)
                    if match:
                        try:
                            progress = int(match.group(1))
                            progress = max(0, min(progress, 100))
                            if progress > last_progress:
                                self.progress_signal.emit(progress)
                                last_progress = progress
                        except ValueError:
                            pass

                
                time.sleep(0.01)

            print(f"{Fore.CYAN}--- Finished Reading Loop, Reading Remaining Output ---")

            
            remaining_stdout = self.process.stdout.read()
            if remaining_stdout:
                for line in remaining_stdout.strip().splitlines():
                    print(f"{Fore.GREEN}STDOUT: {line}")
                    self.log_signal.emit(f"OUT: {line}")
                    stdout_full.append(line)

            remaining_stderr = self.process.stderr.read()
            if remaining_stderr:
                for line in remaining_stderr.strip().splitlines():
                    print(f"{Fore.RED}STDERR: {line}")
                    self.log_signal.emit(f"ERR: {line}")
                    stderr_full.append(line)

            return_code = self.process.wait()
            print(f"{Fore.CYAN}--- Process finished with code: {return_code} ---")
            if return_code == 0:
                self.progress_signal.emit(100)
                final_message = "Status: Processing complete!"
                if stdout_full:
                    final_message += "\n--- Final Output Snippet ---\n"
                    final_message += "\n".join(stdout_full[-5:])
                self.finished_signal.emit(True, final_message)
            else:
                error_message = f"Error during processing (Code: {return_code}).\n"
                stdout_text = "\n".join(stdout_full)
                stderr_text = "\n".join(stderr_full)
                if stdout_text:
                    error_message += f"\n--- Collected Standard Output ---\n{stdout_text[-2500:]}\n"
                if stderr_text:
                    error_message += f"\n--- Collected Standard Error ---\n{stderr_text[-2500:]}\n"
                if not stdout_text and not stderr_text:
                    error_message += "\n(No standard output or error captured)"
                self.progress_signal.emit(0)
                self.finished_signal.emit(False, error_message)
        except Exception as e:
            print(f"{Fore.RED}--- Error in thread setup/Popen: {e} ---")
            traceback.print_exc()
            self.progress_signal.emit(0)
            self.finished_signal.emit(False, f"Error starting processing thread: {e}")
        finally:
            print(f"{Fore.CYAN}--- ProcessingThread finished ---")
            self.process = None
            self._is_running = False

    def stop(self):
        print(f"{Fore.YELLOW}--- Stop requested ---")
        self._is_running = False
        if self.process and self.process.poll() is None:
            try:
                self.status_signal.emit("Status: Attempting to terminate process...")
                self.process.kill()
                self.process.wait(timeout=5)
                self.status_signal.emit("Status: Processing terminated by user.")
            except subprocess.TimeoutExpired:
                self.status_signal.emit("Status: Process termination check timed out.")
            except Exception as e:
                self.status_signal.emit(f"Status: Error stopping process: {e}")
        self.quit()


class AudioEditorUI(QWidget):
    def __init__(self): super().__init__(); self.audioFilePath=None; self.processing_thread=None; self.torch_available=False; self.cuda_available=False; self.device="cpu"; self.initUI()
    def initUI(self):
        
        self.setWindowTitle("AudioEditing UI By Germanized"); self.setGeometry(200,200,700,600)
        palette=QPalette(); palette.setColor(QPalette.ColorRole.Window,QColor(35,35,35)); palette.setColor(QPalette.ColorRole.WindowText,QColor(220,220,220)); palette.setColor(QPalette.ColorRole.Base,QColor(25,25,25)); palette.setColor(QPalette.ColorRole.AlternateBase,QColor(53,53,53)); palette.setColor(QPalette.ColorRole.ToolTipBase,Qt.GlobalColor.white); palette.setColor(QPalette.ColorRole.ToolTipText,Qt.GlobalColor.white); palette.setColor(QPalette.ColorRole.Text,QColor(220,220,220)); palette.setColor(QPalette.ColorRole.PlaceholderText,QColor(120,120,120)); palette.setColor(QPalette.ColorRole.Button,QColor(53,53,53)); palette.setColor(QPalette.ColorRole.ButtonText,QColor(220,220,220)); palette.setColor(QPalette.ColorRole.BrightText,Qt.GlobalColor.red); palette.setColor(QPalette.ColorRole.Link,QColor(42,130,218)); palette.setColor(QPalette.ColorRole.Highlight,QColor(42,130,218)); palette.setColor(QPalette.ColorRole.HighlightedText,Qt.GlobalColor.black); palette.setColor(QPalette.ColorGroup.Disabled,QPalette.ColorRole.ButtonText,QColor(120,120,120)); palette.setColor(QPalette.ColorGroup.Disabled,QPalette.ColorRole.Text,QColor(120,120,120)); palette.setColor(QPalette.ColorGroup.Disabled,QPalette.ColorRole.WindowText,QColor(120,120,120)); self.setPalette(palette)
        layout=QVBoxLayout(); layout.setContentsMargins(15,15,15,15); layout.setSpacing(10)
        self.dependencyStatusLabel=QLabel("Checking dependencies..."); self.dependencyStatusLabel.setStyleSheet("color: yellow; font-style: italic; padding-bottom: 5px;"); layout.addWidget(self.dependencyStatusLabel)
        self.label=QLabel("Select an audio file (.wav or .mp3):"); layout.addWidget(self.label)
        self.uploadButton=QPushButton("Upload Audio"); self.uploadButton.setMinimumHeight(35); self.uploadButton.clicked.connect(self.uploadFile); layout.addWidget(self.uploadButton)
        self.textEdit=QTextEdit(); self.textEdit.setPlaceholderText("Enter text-based editing instructions..."); self.textEdit.setMinimumHeight(100); layout.addWidget(self.textEdit)
        self.lengthLabel=QLabel("Audio Length: N/A"); layout.addWidget(self.lengthLabel)
        self.modelLabel=QLabel("Select Model:"); layout.addWidget(self.modelLabel)
        self.modelComboBox=QComboBox(); self.modelComboBox.setMinimumHeight(30)
        for display_name,model_id in MODEL_CHOICES.items(): self.modelComboBox.addItem(display_name,userData=model_id)
        if DEFAULT_MODEL in MODEL_CHOICES: self.modelComboBox.setCurrentText(DEFAULT_MODEL)
        layout.addWidget(self.modelComboBox)
        self.progressBar=QProgressBar(); self.progressBar.setValue(0); self.progressBar.setTextVisible(True); self.progressBar.setMinimumHeight(25); layout.addWidget(self.progressBar)
        self.deviceLabel=QLabel("Device: Unknown"); layout.addWidget(self.deviceLabel)
        self.processButton=QPushButton("Process Audio"); self.processButton.setMinimumHeight(35); self.processButton.clicked.connect(self.processAudio); self.processButton.setEnabled(False); layout.addWidget(self.processButton)
        self.statusLabel=QLabel("Status: Waiting for input..."); self.statusLabel.setWordWrap(True); self.statusLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse|Qt.TextInteractionFlag.TextSelectableByKeyboard); self.statusLabel.setStyleSheet("padding-top: 5px; border: 1px solid 
        self.setLayout(layout)

    
    def checkDependenciesAndDevice(self):
        self.dependencyStatusLabel.setText("Status: Checking PyTorch..."); QApplication.processEvents()
        torch_spec=importlib.util.find_spec("torch");
        if torch_spec is None: self.handleDependencyError("❌ Error: PyTorch not found.","Status: PyTorch is required.","PyTorch not installed.\nSee: https://pytorch.org/get-started/locally/"); return
        self.torch_available=True
        try: self.dependencyStatusLabel.setText("Status: Importing PyTorch..."); QApplication.processEvents(); import torch; import torchaudio
        except ImportError as e: self.handleDependencyError(f"❌ Error: Import failed ({e}).",f"Status: Import error.",f"Failed to import PyTorch/Torchaudio:\n{e}"); self.torch_available=False; return
        try: 
            self.dependencyStatusLabel.setText("Status: Checking CUDA..."); QApplication.processEvents(); import torch; self.cuda_available=torch.cuda.is_available()
            if self.cuda_available:
                self.device="cuda";
                try: 
                    device_index=torch.cuda.current_device(); device_name=torch.cuda.get_device_name(device_index)
                    try: props=torch.cuda.get_device_properties(device_index); vram_gb=props.total_memory/(1024**3); device_name+=f" ({vram_gb:.1f} GB VRAM)"
                    except Exception: pass
                    self.deviceLabel.setText(f"✅ Device: {device_name} (CUDA:{device_index})"); self.dependencyStatusLabel.setText("✅ Dependencies OK (CUDA Found)"); self.dependencyStatusLabel.setStyleSheet("color: lightgreen;"); self.processButton.setEnabled(True); self.statusLabel.setText("Status: Ready.")
                except Exception as e: self.deviceLabel.setText(f"⚠️ Device: CUDA detected but error (CUDA:{torch.cuda.current_device()})!"); self.dependencyStatusLabel.setText("⚠️ Warning: CUDA query issue."); self.dependencyStatusLabel.setStyleSheet("color: orange;"); QMessageBox.warning(self,"CUDA Warning",f"CUDA ok, but failed get name:\n{e}"); self.processButton.setEnabled(True); self.statusLabel.setText("Status: Ready (CUDA warn).")
            else: 
                self.device="cpu"; self.deviceLabel.setText("️ Device: CPU (Processing Disabled)"); self.processButton.setEnabled(False)
                nvidia_likely = False

                
                
                try:
                    
                    self.dependencyStatusLabel.setText("Status: Checking for NVIDIA driver...")
                    QApplication.processEvents()
                    cmd = ["nvidia-smi"]
                    flags = 0
                    
                    if sys.platform == "win32":
                        
                        flags = subprocess.CREATE_NO_WINDOW
                    
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
                    nvidia_likely = True 
                
                except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
                    
                    print(f"nvidia-smi check failed: {e}")
                    nvidia_likely = False 
                

                
                if nvidia_likely:
                    self.dependencyStatusLabel.setText("❌ Error: PyTorch CUDA support missing!"); self.dependencyStatusLabel.setStyleSheet("color: red; font-weight: bold;"); self.statusLabel.setText("Status: GPU requires PyTorch w/ CUDA."); QMessageBox.warning(self,"CUDA Config Issue","NVIDIA GPU present, but PyTorch lacks CUDA.\nThis tool requires CUDA.\nPlease reinstall PyTorch with CUDA.")
                else:
                    self.dependencyStatusLabel.setText("❌ Error: CUDA device not found."); self.dependencyStatusLabel.setStyleSheet("color: red; font-weight: bold;"); self.statusLabel.setText("Status: No CUDA GPU detected. CUDA required."); QMessageBox.warning(self,"CUDA Required","No CUDA GPU detected.\nThis tool requires CUDA.")
            
        
        except Exception as e:
            self.handleDependencyError("❌ Error checking PyTorch/CUDA.",f"Status: Error: {e}",f"Error checking PyTorch/CUDA:\n{e}"); self.cuda_available=False; self.processButton.setEnabled(False)
    

    def handleDependencyError(self,dep_status_text,status_text,msg_box_text): 
        self.dependencyStatusLabel.setText(dep_status_text); self.dependencyStatusLabel.setStyleSheet("color: red; font-weight: bold;"); self.statusLabel.setText(status_text); self.deviceLabel.setText("Device: Unknown (Error)"); self.processButton.setEnabled(False); self.torch_available=False; QMessageBox.critical(self,"Dependency Check Error",msg_box_text)
    def uploadFile(self): 
        filePath,_=QFileDialog.getOpenFileName(self,"Open Audio File","","Audio Files (*.wav *.mp3)");
        if filePath: self.audioFilePath=filePath; base_name=os.path.basename(filePath); max_len=50; display_name=base_name if len(base_name)<=max_len else base_name[:max_len-3]+"..."; self.label.setText(f"Selected: {display_name}"); self.label.setToolTip(filePath); self.getAudioLength()
        if self.processButton.isEnabled(): self.statusLabel.setText("Status: Ready for editing instructions.")
        self.progressBar.setValue(0)
    def getAudioLength(self): 
        if not self.audioFilePath: self.lengthLabel.setText("Audio Length: N/A"); return
        if not self.torch_available: self.lengthLabel.setText("Audio Length: Cannot determine"); return
        try: import torchaudio; info=torchaudio.info(self.audioFilePath); length_sec=info.num_frames/info.sample_rate; self.lengthLabel.setText(f"Audio Length: {length_sec:.2f} sec")
        except FileNotFoundError: self.lengthLabel.setText("Audio Length: Error (Not Found)"); self.statusLabel.setText("Status: Error loading audio file."); QMessageBox.warning(self,"File Error",f"Audio file not found:\n{self.audioFilePath}"); self.audioFilePath=None; self.label.setText("Select an audio file...")
        except Exception as e: self.lengthLabel.setText("Audio Length: Error"); self.statusLabel.setText(f"Status: Error reading audio: {e}"); QMessageBox.warning(self,"Audio Load Error",f"Could not read audio file:\n{e}")
    def processAudio(self): 
        if not self.cuda_available: QMessageBox.critical(self,"CUDA Required","Cannot start: CUDA GPU required."); self.statusLabel.setText("Status: Error - CUDA required."); return
        if not self.audioFilePath: QMessageBox.warning(self,"Input Missing","Please upload an audio file."); return
        editPrompt=self.textEdit.toPlainText().strip();
        if not editPrompt: QMessageBox.warning(self,"Input Missing","Please enter editing instructions."); return
        if self.processing_thread and self.processing_thread.isRunning(): QMessageBox.warning(self,"Busy","Processing already in progress."); return
        if not os.path.exists(PROCESSING_SCRIPT): QMessageBox.critical(self,"Script Error",f"Processing script not found at:\n{PROCESSING_SCRIPT}"); self.statusLabel.setText(f"Status: Error - Script not found!"); return
        selected_model_id=self.modelComboBox.currentData()
        if not selected_model_id: QMessageBox.warning(self,"Input Missing","Please select a model."); return
        if selected_model_id == STABLE_AUDIO_ID:
            reply=QMessageBox.warning(self,"Hugging Face Token Required",f"Ensure `HF_TOKEN` in '{os.path.basename(PROCESSING_SCRIPT)}' is set.",QMessageBox.StandardButton.Ok|QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Ok)
            if reply == QMessageBox.StandardButton.Cancel: self.statusLabel.setText("Status: Processing cancelled."); return
        self.statusLabel.setText("Status: Preparing to process..."); self.progressBar.setValue(5); self.processButton.setEnabled(False); self.uploadButton.setEnabled(False); self.modelComboBox.setEnabled(False)
        command=[sys.executable,PROCESSING_SCRIPT,"--cfg_tar","3.0","--cfg_src","1.0","--init_aud",self.audioFilePath,"--target_prompt",editPrompt,"--tstart","100","--model_id",selected_model_id,"--results_path",OUTPUT_PATH]
        if self.device=="cuda":
            try: import torch; command.extend(["--device_num",str(torch.cuda.current_device())])
            except ImportError: QMessageBox.critical(self,"Error","Torch import failed."); self.onProcessingFinished(False,"Status: Error - Failed CUDA setup."); return
            except Exception as e: QMessageBox.critical(self,"Error",f"CUDA device error: {e}"); self.onProcessingFinished(False,f"Status: Error - CUDA device num: {e}"); return
        else: QMessageBox.critical(self,"Error","Internal state error: CUDA required."); self.onProcessingFinished(False,"Status: Internal Error - CUDA state"); return
        print(f"Running command list: {command}")
        print(f"Equivalent manual command (add quotes in terminal!): {' '.join(command)}")
        self.processing_thread=ProcessingThread(command); self.processing_thread.progress_signal.connect(self.progressBar.setValue); self.processing_thread.status_signal.connect(self.statusLabel.setText); self.processing_thread.finished_signal.connect(self.onProcessingFinished);
        
        self.processing_thread.start()
    def onProcessingFinished(self,success,message): 
        self.statusLabel.setText(message.strip())
        if success:
            self.progressBar.setValue(100)
            try:
                output_abs_path = os.path.abspath(OUTPUT_PATH)
                print(f"Attempting to open output folder: {output_abs_path}")
                if not os.path.exists(output_abs_path):
                    print(f"Output path does not exist: {output_abs_path}")
                    self.statusLabel.setText(f"{self.statusLabel.text()}\n(Output folder path does not exist)")
                elif sys.platform == "win32":
                    print("Platform is Windows, using os.startfile()")
                    os.startfile(output_abs_path)
                elif sys.platform == "darwin":
                    print("Platform is macOS, using 'open'")
                    subprocess.run(['open', output_abs_path], check=True)
                else:
                    print("Platform is Linux/other, using 'xdg-open'")
                    subprocess.run(['xdg-open', output_abs_path], check=True)
            except FileNotFoundError:
                print(f"FileNotFoundError when trying to open output folder.")
                self.statusLabel.setText(f"{self.statusLabel.text()}\n(Could not open output folder - opener not found or path invalid?)")
            except subprocess.CalledProcessError as cpe:
                print(f"CalledProcessError when opening output folder: {cpe}")
                self.statusLabel.setText(f"{self.statusLabel.text()}\n(Error running folder opener command: {cpe})")
            except Exception as e:
                print(f"Generic Exception when opening output folder: {e}")
                self.statusLabel.setText(f"{self.statusLabel.text()}\n(Could not open output folder: {e})")
        else: self.progressBar.setValue(0); QMessageBox.critical(self,"Processing Error","An error occurred. See status message for details.")
        if self.cuda_available: self.processButton.setEnabled(True)
        else: self.processButton.setEnabled(False)
        self.uploadButton.setEnabled(True); self.modelComboBox.setEnabled(True); self.processing_thread=None
    def closeEvent(self, event):  
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                'Confirm Exit',
                "Processing ongoing. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.processing_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  
    if hasattr(Qt,'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    if hasattr(Qt,'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,True)
    app=QApplication(sys.argv);
    splash=ModernSplashScreen(text="Made by Germanized",width=450,height=180,corner_radius=20)
    screen_geometry=app.primaryScreen().availableGeometry(); splash.move(screen_geometry.center()-splash.rect().center()); splash.show(); app.processEvents()
    editor=AudioEditorUI()
    def show_main_window(): screen_geometry=app.primaryScreen().availableGeometry(); editor.move(screen_geometry.center()-editor.rect().center()); editor.show(); splash.close(); editor.checkDependenciesAndDevice()
    QTimer.singleShot(SPLASH_DURATION,show_main_window)
    sys.exit(app.exec())