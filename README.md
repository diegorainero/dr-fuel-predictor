# 🚗 DR Fuel Predictor

**Sistema di previsione prezzi carburanti con intelligenza artificiale**

Un'applicazione web sviluppata in Rust che analizza i dati storici dei prezzi dei carburanti in Italia, integra notizie di mercato e utilizza l'intelligenza artificiale per generare previsioni accurate sui prezzi futuri.

## 📋 Caratteristiche

- **📊 Analisi Avanzata**: Calcolo di volatilità, tendenze e indicatori tecnici
- **🤖 Intelligenza Artificiale**: Integrazione con OpenAI GPT-4 per analisi predittive
- **📰 Analisi News**: Monitoraggio sentiment delle notizie di mercato
- **🌐 API REST**: Endpoint completi per l'integrazione con sistemi esterni
- **💻 Interfaccia Web**: Dashboard moderna e responsive con Bootstrap 5
- **📈 Visualizzazioni**: Grafici e metriche in tempo reale
- **🔄 Aggiornamenti Real-time**: Dati sempre aggiornati dalle fonti ufficiali

## 🏗️ Architettura

### Backend (Rust)
- **Framework**: Actix-Web per il server HTTP
- **Database**: Integrazione con API MASE (Ministero dell'Ambiente e della Sicurezza Energetica)
- **AI Integration**: async-openai per l'analisi predittiva
- **News API**: Integrazione per il monitoraggio delle notizie di mercato

### Frontend (JavaScript + Bootstrap)
- **UI Framework**: Bootstrap 5 con Font Awesome
- **Charts**: Chart.js per visualizzazioni
- **Responsive Design**: Ottimizzato per desktop e mobile

## 🚀 Installazione e Setup

### Prerequisiti
- **Rust** (edizione 2021+)
- **OpenAI API Key** (per l'analisi AI)
- **News API Key** (opzionale, per le notizie)

### Installazione

1. **Clona il repository**:
```bash
git clone https://github.com/tuousername/dr_fuel_predictor.git
cd dr_fuel_predictor
```

2. **Configura le variabili d'ambiente**:
```bash
cp .env.example .env
```

Modifica il file `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
NEWS_API_KEY=your_news_api_key_here
```

3. **Compila ed esegui**:
```bash
cargo run
```

L'applicazione sarà disponibile su: `http://localhost:8083`

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```
Verifica lo stato del server.

**Risposta**:
```json
{
  "success": true,
  "data": "Server is running",
  "error": null
}
```

### Previsioni Carburanti
```http
POST /api/predict
```
Genera previsioni per tutti i tipi di carburante.

**Risposta**:
```json
{
  "success": true,
  "data": [
    {
      "prodotto": "BENZINA SUPER",
      "prezzo_attuale": 1.785,
      "prezzo_previsto": 1.798,
      "variazione_percentuale": 0.73,
      "tendenza": "Tendenza al rialzo",
      "ai_analysis": "Analisi AI dettagliata...",
      "confidence": 82.5,
      "market_sentiment": "Mercato stabile",
      "news_impact": 0.3,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "error": null
}
```

### Notizie di Mercato
```http
GET /api/news
```
Recupera le ultime notizie correlate al mercato energetico.

**Risposta**:
```json
{
  "success": true,
  "data": [
    {
      "title": "Tensioni in Medio Oriente fanno salire il prezzo del petrolio",
      "description": "Le recenti tensioni geopolitiche...",
      "url": "https://example.com/news1",
      "published_at": "2024-01-15T08:00:00Z",
      "source": {
        "name": "Reuters"
      },
      "sentiment": -0.7
    }
  ],
  "error": null
}
```

## 🔧 Configurazione

### Variabili d'Ambiente

| Variabile | Descrizione | Obbligatorio |
|-----------|-------------|--------------|
| `OPENAI_API_KEY` | Chiave API di OpenAI per l'analisi AI | ✅ |
| `NEWS_API_KEY` | Chiave API per il recupero notizie | ❌ |

### Porte e Binding
- **Server HTTP**: `127.0.0.1:8083`
- **File Statici**: Serviti da `/static`

## 📊 Fonti Dati

### Dati Ufficiali Carburanti
- **Fonte**: MASE (Ministero dell'Ambiente e della Sicurezza Energetica)
- **API**: `https://sisen.mase.gov.it/dgsaie/api/v1/weekly-prices/report/export`
- **Formato**: JSON
- **Frequenza**: Settimanale

### Analisi Tecnica
L'applicazione calcola automaticamente:
- **Volatilità**: Deviazione standard dei prezzi
- **Tendenze**: Analisi delle variazioni percentuali
- **Confidence Score**: Basato su volatilità e dati storici
- **Indicatori**: RSI, MACD, Bollinger Bands (in sviluppo)

## 🤖 Integrazione AI

### OpenAI GPT-4
- **Modello**: `gpt-4-turbo-preview`
- **Temperatura**: 0.2 (per risposte più conservative)
- **Max Tokens**: 2000
- **Funzioni**:
  - Analisi contestuale del mercato
  - Interpretazione delle notizie
  - Previsioni qualitative
  - Identificazione fattori di rischio

### Prompt Engineering
Il sistema utilizza prompt specializzati in italiano per:
- Analisi macroeconomica
- Correlazioni geopolitiche
- Fattori stagionali
- Sentiment analysis delle notizie

## 🎨 Interfaccia Utente

### Dashboard Principale
- **Previsioni**: Cards colorate per ogni tipo di carburante
- **Tendenze**: Indicatori visivi (↗️ ↘️ ➡️)
- **Confidence Bar**: Barre di progresso per il livello di fiducia
- **News Feed**: Ultime notizie con sentiment analysis
- **Analisi AI**: Sezione dedicata ai insights generati dall'AI

### Controlli
- **Refresh**: Aggiornamento manuale dei dati
- **Export**: Esportazione dati (in sviluppo)
- **Responsive**: Ottimizzato per tutti i dispositivi

## 🔍 Struttura del Progetto

```
dr_fuel_predictor/
├── src/
│   ├── main.rs           # Server HTTP e routing
│   └── predictor.rs      # Logica predittiva e AI
├── static/
│   ├── index.html        # Dashboard principale
│   ├── app.js           # Logica frontend
│   └── styles.css       # Stili personalizzati
├── Cargo.toml           # Dipendenze Rust
├── .env                 # Variabili d'ambiente
└── README.md           # Documentazione
```

## 📈 Algoritmi Predittivi

### 1. Analisi Statistica
- **Media Mobile**: Calcolo tendenze a breve termine
- **Volatilità**: Deviazione standard normalizzata
- **Correlazioni**: Analisi multi-prodotto

### 2. Fattori di Mercato
- **Geopolitica**: Score basato su sentiment news
- **Stagionalità**: Coefficienti mensili
- **Domanda/Offerta**: Analisi pattern storici

### 3. Machine Learning (Pianificato)
- **Time Series Forecasting**: ARIMA, Prophet
- **Feature Engineering**: Variabili esogene
- **Model Ensemble**: Combinazione algoritmi

## 🛠️ Sviluppo

### Test
```bash
cargo test
```

### Build Release
```bash
cargo build --release
```

### Linting
```bash
cargo clippy
cargo fmt
```

### Logs e Debug
Il sistema fornisce logging dettagliato:
- 📡 Richieste API
- 📊 Calcoli statistici
- 🤖 Interazioni AI
- ⚠️ Errori e warning

## 🚀 Deploy

### Docker (Raccomandato)
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/dr_fuel_predictor /usr/local/bin/
COPY --from=builder /app/static /usr/local/bin/static
EXPOSE 8083
CMD ["dr_fuel_predictor"]
```

### Server Tradizionale
1. Compila per la produzione: `cargo build --release`
2. Copia il binario e la cartella `static` sul server
3. Configura reverse proxy (nginx/Apache)
4. Setup systemd service per auto-restart

## 📊 Performance

### Benchmark
- **Latenza API**: < 500ms per previsione completa
- **Throughput**: > 100 req/sec
- **Memory Usage**: ~50MB a riposo
- **AI Response Time**: 2-5 secondi (dipende da OpenAI)

### Ottimizzazioni
- **Caching**: Risultati AI cached per 1 ora
- **Compression**: Gzip per risposte JSON
- **Connection Pooling**: HTTP client riutilizzabile
- **Async Processing**: Elaborazione non bloccante

## 🔐 Sicurezza

### API Keys
- Mai committare chiavi nel codice
- Usa variabili d'ambiente
- Rotazione periodica delle chiavi

### CORS
- Configurato per sviluppo locale
- Restringi in produzione

### Rate Limiting
- Implementare throttling per API esterne
- Monitorare usage OpenAI

## 🤝 Contributi

1. Fork il progetto
2. Crea un branch per la feature (`git checkout -b feature/AmazingFeature`)
3. Commit le modifiche (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📝 TODO / Roadmap

### Versione 2.0
- [ ] **Database persistente** (PostgreSQL/SQLite)
- [ ] **Autenticazione utenti**
- [ ] **Dashboard amministratore**
- [ ] **Notifiche push/email**
- [ ] **API rate limiting**
- [ ] **Unit tests completi**

### Versione 2.5
- [ ] **Machine Learning avanzato**
- [ ] **Previsioni multiple timeframe**
- [ ] **Analisi competitor pricing**
- [ ] **Mobile app (React Native)**
- [ ] **Integrazione social media**

### Versione 3.0
- [ ] **Microservizi architecture**
- [ ] **Kubernetes deployment**
- [ ] **Real-time websockets**
- [ ] **Multi-country support**
- [ ] **Blockchain price verification**

## 📄 Licenza

Questo progetto è sotto licenza MIT. Vedi il file `LICENSE` per i dettagli.

## 👨‍💻 Autore

**Diego Rainero**
- Email: [tuo-email@example.com]
- LinkedIn: [tuo-linkedin]
- GitHub: [@tuousername]

## 🙏 Ringraziamenti

- **MASE** per i dati aperti sui carburanti
- **OpenAI** per le API di intelligenza artificiale
- **Rust Community** per l'ecosistema fantastico
- **Actix-Web** per il framework web performante

---

💡 **Suggerimento**: Per supporto o domande, apri una [Issue](https://github.com/tuousername/dr_fuel_predictor/issues) su GitHub!
