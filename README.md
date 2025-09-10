# 🚗 DR Fuel Predictor

AI-powered fuel price forecasting system

A Rust-based web application that analyzes historical fuel price data in Italy, integrates market news, and uses artificial intelligence to generate accurate forecasts of future prices.

## 📋 Features

- 📊 Advanced Analytics: volatility, trends, and technical indicators
- 🤖 Artificial Intelligence: OpenAI GPT-4 integration for predictive analysis
- 📰 News Analysis: market news sentiment monitoring
- 🌐 REST API: complete endpoints for integration with external systems
- 💻 Web Interface: modern, responsive dashboard with Bootstrap 5
- 📈 Visualizations: real-time charts and metrics
- 🔄 Real-time Updates: continuously refreshed data from official sources

## 🏗️ Architecture

### Backend (Rust)
- Framework: Actix-Web HTTP server
- Data Source: integration with MASE API (Italian Ministry of Environment and Energy Security)
- AI Integration: async-openai for predictive analysis
- News API: integration for monitoring market news

### Frontend (JavaScript + Bootstrap)
- UI Framework: Bootstrap 5 with Font Awesome
- Charts: Chart.js visualizations
- Responsive Design: optimized for desktop and mobile

## 🚀 Installation & Setup

### Prerequisites
- Rust (edition 2021+)
- OpenAI API Key (for AI analysis)
- News API Key (optional, for news)

### Installation

1) Clone the repository:
```bash
git clone https://github.com/tuousername/dr_fuel_predictor.git
cd dr_fuel_predictor
```

2) Configure environment variables:
```bash
cp .env.example .env
```

Edit the `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
NEWS_API_KEY=your_news_api_key_here
```

3) Build and run:
```bash
cargo run
```

The application will be available at: `http://localhost:8083`

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```
Checks the server status.

Response:
```json
{
  "success": true,
  "data": "Server is running",
  "error": null
}
```

### Fuel Predictions
```http
POST /api/predict
```
Generates forecasts for all fuel types.

Response:
```json
{
  "success": true,
  "data": [
    {
      "product": "BENZINA SUPER",
      "current_price": 1.785,
      "predicted_price": 1.798,
      "percent_change": 0.73,
      "trend": "Upward trend",
      "ai_analysis": "Detailed AI analysis...",
      "confidence": 82.5,
      "market_sentiment": "Stable market",
      "news_impact": 0.3,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "error": null
}
```

### Market News
```http
GET /api/news
```
Retrieves the latest news related to the energy market.

Response:
```json
{
  "success": true,
  "data": [
    {
      "title": "Tensions in the Middle East push oil prices higher",
      "description": "Recent geopolitical tensions...",
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

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI analysis | ✅ |
| `NEWS_API_KEY` | API key to fetch news | ❌ |

### Ports and Binding
- HTTP Server: `127.0.0.1:8083`
- Static Files: served from `/static`

## 📊 Data Sources

### Official Fuel Data
- Source: MASE (Italian Ministry of Environment and Energy Security)
- API: `https://sisen.mase.gov.it/dgsaie/api/v1/weekly-prices/report/export`
- Format: JSON
- Frequency: Weekly

### Technical Analysis
The application automatically calculates:
- Volatility: standard deviation of prices
- Trends: percentage change analysis
- Confidence Score: based on volatility and historical data
- Indicators: RSI, MACD, Bollinger Bands (in progress)

## 🤖 AI Integration

### OpenAI GPT-4
- Model: `gpt-4-turbo-preview`
- Temperature: 0.2 (for more conservative responses)
- Max Tokens: 2000
- Capabilities:
  - Contextual market analysis
  - News interpretation
  - Qualitative forecasts
  - Risk factor identification

### Prompt Engineering
The system uses specialized prompts (in Italian) for:
- Macroeconomic analysis
- Geopolitical correlations
- Seasonal factors
- News sentiment analysis

## 🎨 User Interface

### Main Dashboard
- Predictions: colored cards for each fuel type
- Trends: visual indicators (↗️ ↘️ ➡️)
- Confidence Bar: progress bars indicating confidence level
- News Feed: latest news with sentiment analysis
- AI Analysis: dedicated section with AI-generated insights

### Controls
- Refresh: manual data refresh
- Export: data export (in progress)
- Responsive: optimized for all devices

## 🔍 Project Structure

```
dr_fuel_predictor/
├── src/
│   ├── main.rs           # HTTP server and routing
│   └── predictor.rs      # Predictive logic and AI
├── static/
│   ├── index.html        # Main dashboard
│   ├── app.js            # Frontend logic
│   └── styles.css        # Custom styles
├── Cargo.toml            # Rust dependencies
├── .env                  # Environment variables
└── README.md             # Documentation
```

## 📈 Predictive Algorithms

### 1. Statistical Analysis
- Moving Average: short-term trend calculation
- Volatility: normalized standard deviation
- Correlations: multi-product analysis

### 2. Market Factors
- Geopolitics: score based on news sentiment
- Seasonality: monthly coefficients
- Supply/Demand: historical pattern analysis

### 3. Machine Learning (Planned)
- Time Series Forecasting: ARIMA, Prophet
- Feature Engineering: exogenous variables
- Model Ensemble: algorithm combinations

## 🛠️ Development

### Tests
```bash
cargo test
```

### Release Build
```bash
cargo build --release
```

### Linting
```bash
cargo clippy
cargo fmt
```

### Logging & Debugging
The system provides detailed logging:
- 📡 API requests
- 📊 Statistical calculations
- 🤖 AI interactions
- ⚠️ Errors and warnings

## 🚀 Deployment

### Docker (Recommended)
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

### Traditional Server
1. Build for production: `cargo build --release`
2. Copy the binary and the `static` folder to the server
3. Configure a reverse proxy (nginx/Apache)
4. Set up a systemd service for auto-restart

## 📊 Performance

### Benchmarks
- API Latency: < 500ms per full forecast
- Throughput: > 100 req/sec
- Memory Usage: ~50MB idle
- AI Response Time: 2-5 seconds (depends on OpenAI)

### Optimizations
- Caching: AI results cached for 1 hour
- Compression: Gzip for JSON responses
- Connection Pooling: reusable HTTP client
- Async Processing: non-blocking workloads

## 🔐 Security

### API Keys
- Never commit keys in code
- Use environment variables
- Rotate keys periodically

### CORS
- Configured for local development
- Restrict in production

### Rate Limiting
- Implement throttling for external APIs
- Monitor OpenAI usage

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 TODO / Roadmap

### Version 2.0
- [ ] Persistent database (PostgreSQL/SQLite)
- [ ] User authentication
- [ ] Admin dashboard
- [ ] Push/email notifications
- [ ] API rate limiting
- [ ] Comprehensive unit tests

### Version 2.5
- [ ] Advanced Machine Learning
- [ ] Multiple timeframe forecasts
- [ ] Competitor pricing analysis
- [ ] Mobile app (React Native)
- [ ] Social media integration

### Version 3.0
- [ ] Microservices architecture
- [ ] Kubernetes deployment
- [ ] Real-time websockets
- [ ] Multi-country support
- [ ] Blockchain price verification

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 👨‍💻 Author

**Diego Rainero**
- Email: [your-email@example.com]
- LinkedIn: [your-linkedin]
- GitHub: [@tuousername]

## 🙏 Acknowledgements

- MASE for open fuel data
- OpenAI for AI APIs
- The Rust Community for the fantastic ecosystem
- Actix-Web for the high-performance web framework

---

💡 Tip: For support or questions, open an [Issue](https://github.com/tuousername/dr_fuel_predictor/issues) on GitHub!
