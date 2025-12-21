import customtkinter as ctk
from datetime import datetime
from tkinter import messagebox

# Configurazione Globale dell'Aspetto
ctk.set_appearance_mode("Dark")  # Modalità scura fissa
ctk.set_default_color_theme("blue")  # Tema colori (blu, verde, o dark-blue)

class ModernConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configurazione Finestra Principale
        self.title("Currency Converter 2025")
        self.geometry("400x720")
        self.resizable(False, False)
        
        # Grid layout configuration (1 colonna centrale)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Dati (Tassi basati su 1 EUR) ---
        self.rates = {
            "EUR": 1.0,
            "USD": 1.09,    # Dollaro USA
            "GBP": 0.86,    # Sterlina
            "JPY": 161.50,  # Yen
            "CHF": 0.95,    # Franco Svizzero
            "CAD": 1.48     # Dollaro Canadese
        }
        self.currencies = list(self.rates.keys())

        # --- Creazione UI ---
        self.create_ui()

    def create_ui(self):
        # 1. Header Title
        self.lbl_title = ctk.CTkLabel(
            self, 
            text="Converter Pro", 
            font=("Roboto Medium", 24),
            text_color="#FFFFFF"
        )
        self.lbl_title.pack(pady=(30, 20))

        # --- CARD 1: INPUT & AZIONI ---
        self.input_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#2B2B2B")
        self.input_frame.pack(fill="x", padx=20, pady=10)

        # Etichetta Importo
        self.lbl_amount = ctk.CTkLabel(
            self.input_frame, 
            text="IMPORTO", 
            font=("Roboto", 12, "bold"), 
            text_color="#A0A0A0"
        )
        self.lbl_amount.pack(anchor="w", padx=20, pady=(20, 5))

        # Campo Input Importo
        self.entry_amount = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="0.00",
            height=50,
            font=("Roboto", 20),
            corner_radius=10,
            fg_color="#3A3A3A",      # Sfondo scuro input
            text_color="#FFFFFF",    # Testo bianco brillante
            border_color="#565B5E",
            border_width=1
        )
        self.entry_amount.pack(fill="x", padx=20, pady=(0, 20))

        # Container per i menu a tendina
        self.combo_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.combo_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Dropdown DA
        self.combo_from = ctk.CTkComboBox(
            self.combo_frame,
            values=self.currencies,
            font=("Roboto", 14),
            width=130,
            height=40,
            corner_radius=10,
            fg_color="#3A3A3A",
            border_color="#565B5E",
            dropdown_fg_color="#2B2B2B"
        )
        self.combo_from.set("EUR")
        self.combo_from.pack(side="left")

        # Freccia Icona (Testuale)
        self.lbl_arrow = ctk.CTkLabel(
            self.combo_frame, 
            text="➔", 
            font=("Roboto", 20), 
            text_color="#A0A0A0"
        )
        self.lbl_arrow.pack(side="left", expand=True)

        # Dropdown A
        self.combo_to = ctk.CTkComboBox(
            self.combo_frame,
            values=self.currencies,
            font=("Roboto", 14),
            width=130,
            height=40,
            corner_radius=10,
            fg_color="#3A3A3A",
            border_color="#565B5E",
            dropdown_fg_color="#2B2B2B"
        )
        self.combo_to.set("USD")
        self.combo_to.pack(side="right")

        # Bottone Converti (Grande e Moderno)
        self.btn_convert = ctk.CTkButton(
            self.input_frame,
            text="CONVERTI ORA",
            font=("Roboto Medium", 15),
            height=55,
            corner_radius=15,
            fg_color="#1F6AA5",      # Colore Primario (Blu CustomTkinter)
            hover_color="#144870",   # Colore Hover
            command=self.perform_conversion
        )
        self.btn_convert.pack(fill="x", padx=20, pady=(0, 25))

        # --- CARD 2: RISULTATO ---
        # Non uso un frame, ma lascio il risultato fluttuante al centro per impatto
        self.lbl_result = ctk.CTkLabel(
            self,
            text="---",
            font=("Roboto Medium", 36),
            text_color="#2CC985" # Verde Neon Moderno
        )
        self.lbl_result.pack(pady=(20, 5))

        self.lbl_rate_info = ctk.CTkLabel(
            self,
            text="",
            font=("Roboto", 12),
            text_color="#808080"
        )
        self.lbl_rate_info.pack(pady=(0, 20))

        # --- CARD 3: STORICO (Chat Style) ---
        self.history_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#212121")
        self.history_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.lbl_hist_title = ctk.CTkLabel(
            self.history_frame,
            text="Storico Recente",
            font=("Roboto", 12, "bold"),
            text_color="#A0A0A0"
        )
        self.lbl_hist_title.pack(anchor="w", padx=15, pady=(15, 5))

        # Textbox Console Style
        self.history_box = ctk.CTkTextbox(
            self.history_frame,
            font=("Consolas", 12),
            text_color="#E0E0E0",    # Testo chiarissimo
            fg_color="#1A1A1A",      # Sfondo molto scuro (console)
            corner_radius=10,
            activate_scrollbars=True
        )
        self.history_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.history_box.configure(state="disabled") # Read-only all'avvio

    def perform_conversion(self):
        try:
            # 1. Validazione Input
            amount_str = self.entry_amount.get()
            if not amount_str:
                return # Ignora click vuoti
            
            # Sostituisci virgola con punto per compatibilità italiana
            amount_str = amount_str.replace(",", ".")
            amount = float(amount_str)

            if amount < 0:
                self.show_error("L'importo deve essere positivo.")
                return

            # 2. Logica Conversione
            from_curr = self.combo_from.get()
            to_curr = self.combo_to.get()

            rate_from = self.rates[from_curr]
            rate_to = self.rates[to_curr]

            # Formula Cross-Rate: (Importo / TassoPartenza) * TassoArrivo
            converted_amount = (amount / rate_from) * rate_to
            
            # Calcolo del tasso singolo per display (1 From = X To)
            single_rate = rate_to / rate_from

            # 3. Aggiornamento UI Risultati
            result_formatted = f"{converted_amount:,.2f}"
            self.lbl_result.configure(text=f"{result_formatted} {to_curr}")
            
            self.lbl_rate_info.configure(
                text=f"Tasso di cambio: 1 {from_curr} ≈ {single_rate:.4f} {to_curr}"
            )

            # 4. Aggiornamento Storico
            self.update_history_log(amount, from_curr, result_formatted, to_curr)

        except ValueError:
            self.show_error("Inserisci un numero valido (es. 10.50)")

    def update_history_log(self, amount, f_c, res_s, t_c):
        self.history_box.configure(state="normal")
        
        timestamp = datetime.now().strftime("%H:%M")
        # Formattazione tipo "Chat log"
        log_line = f"[{timestamp}]  {amount:.2f} {f_c}  ➔  {res_s} {t_c}\n\n"
        
        # Inserisci in cima (0.0) invece che in fondo
        self.history_box.insert("0.0", log_line)
        self.history_box.configure(state="disabled")

    def show_error(self, message):
        # Usa messagebox standard di tkinter perché CTk non ha popup nativi stabili ancora
        messagebox.showerror("Errore", message)

if __name__ == "__main__":
    app = ModernConverterApp()
    app.mainloop()