import customtkinter as ctk
import subprocess
from tkinter import messagebox

SCRIPT_PATH = "C:\\Users\\jcotrim\\OneDrive - inegi.up.pt\\Desktop\\zed-sdk\\body tracking\\body tracking\\python\\body_tracking.py"

def run_script():
    try:
        subprocess.Popen(["python", SCRIPT_PATH])
        app.destroy()  
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro: {e}")


ctk.set_appearance_mode("dark")  # ou "light"
ctk.set_default_color_theme("green")  # "blue", "dark-blue"


app = ctk.CTk()
app.title("ZED 2i Body Tracking")
app.geometry("500x250")

label = ctk.CTkLabel(app, text="Bem-vindo à aplicação de Body Tracking com ZED 2i", font=ctk.CTkFont(size=16), wraplength=480, justify="center")
label.pack(pady=30)

botao = ctk.CTkButton(app, text="Iniciar Body Tracking", command=run_script, font=ctk.CTkFont(size=14))
botao.pack(pady=10)

# Inicia a interface
app.mainloop()
