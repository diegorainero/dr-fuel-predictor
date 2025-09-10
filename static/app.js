class FuelPredictorApp {
  constructor() {
    this.baseUrl = "http://localhost:8083/api";
    this.init();
  }

  init() {
    this.loadPredictions();
    this.loadNews();

    document.getElementById("refresh-btn").addEventListener("click", () => {
      this.loadPredictions();
      this.loadNews();
    });

    document.getElementById("export-btn").addEventListener("click", () => {
      this.exportData();
    });
  }

  async loadPredictions() {
    this.showLoading(true);

    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const result = await response.json();

      if (result.success) {
        this.displayPredictions(result.data);
      } else {
        this.showError(result.error || "Errore sconosciuto");
      }
    } catch (error) {
      this.showError("Errore di connessione al server");
    } finally {
      this.showLoading(false);
    }
  }

  async loadNews() {
    try {
      const response = await fetch(`${this.baseUrl}/news`);
      const result = await response.json();

      if (result.success) {
        this.displayNews(result.data);
      }
    } catch (error) {
      console.error("Errore nel caricamento news:", error);
    }
  }

  displayPredictions(predictions) {
    const container = document.getElementById("predictions-container");
    container.innerHTML = "";
    container.classList.remove("d-none");

    predictions.forEach((prediction) => {
      const card = this.createPredictionCard(prediction);
      container.appendChild(card);
    });
  }

  createPredictionCard(prediction) {
    const card = document.createElement("div");
    card.className = "card fuel-card mb-3";

    const trendClass = this.getTrendClass(prediction.tendenza);
    const arrowIcon = this.getTrendIcon(prediction.tendenza);

    card.innerHTML = `
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h6 class="card-title mb-1">${prediction.prodotto}</h6>
                        <p class="card-text mb-1">
                            <span class="${trendClass}">
                                ${arrowIcon} ${prediction.tendenza}
                            </span>
                        </p>
                    </div>
                    <div class="text-end">
                        <h5 class="${trendClass} mb-1">${prediction.variazione_percentuale.toFixed(1)}%</h5>
                        <small class="text-muted">${prediction.prezzo_previsto.toFixed(2)} €/1000L</small>
                    </div>
                </div>

                <div class="mt-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <small>Confidenza:</small>
                        <small>${prediction.confidence.toFixed(1)}%</small>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill bg-success"
                             style="width: ${prediction.confidence}%"></div>
                    </div>
                </div>

                ${
                  prediction.ai_analysis
                    ? `
                    <div class="mt-3">
                        <small class="text-muted">🤖 ${prediction.ai_analysis}</small>
                    </div>
                `
                    : ""
                }
            </div>
        `;

    return card;
  }

  displayNews(newsArticles) {
    const container = document.getElementById("news-container");
    container.innerHTML = "";

    if (!newsArticles || newsArticles.length === 0) {
      container.innerHTML =
        '<p class="text-muted">Nessuna news disponibile</p>';
      return;
    }

    newsArticles.slice(0, 5).forEach((article) => {
      const sentimentClass = this.getSentimentClass(article.sentiment);
      const sentimentIcon = this.getSentimentIcon(article.sentiment);

      const newsItem = document.createElement("div");
      newsItem.className = `news-item ${sentimentClass}`;
      newsItem.innerHTML = `
                <h6 class="mb-1">${article.title}</h6>
                <p class="mb-1 small text-muted">${article.source.name}</p>
                <div class="d-flex justify-content-between align-items-center">
                    <small class="${sentimentClass}">${sentimentIcon} ${this.getSentimentText(article.sentiment)}</small>
                    <small class="text-muted">${new Date(article.published_at).toLocaleDateString()}</small>
                </div>
            `;
      container.appendChild(newsItem);
    });
  }

  getTrendClass(tendenza) {
    if (tendenza.includes("rialzo")) return "trend-up";
    if (tendenza.includes("calo")) return "trend-down";
    return "trend-stable";
  }

  getTrendIcon(tendenza) {
    if (tendenza.includes("rialzo")) return "↗️";
    if (tendenza.includes("calo")) return "↘️";
    return "➡️";
  }

  getSentimentClass(sentiment) {
    if (sentiment > 0.3) return "positive-sentiment";
    if (sentiment < -0.3) return "negative-sentiment";
    return "";
  }

  getSentimentIcon(sentiment) {
    if (sentiment > 0.3) return "😊";
    if (sentiment < -0.3) return "😟";
    return "😐";
  }

  getSentimentText(sentiment) {
    if (sentiment > 0.3) return "Positivo";
    if (sentiment < -0.3) return "Negativo";
    return "Neutro";
  }

  showLoading(show) {
    document.getElementById("loading").classList.toggle("d-none", !show);
    document
      .getElementById("predictions-container")
      .classList.toggle("d-none", show);
  }

  showError(message) {
    const errorContainer = document.getElementById("error-container");
    errorContainer.textContent = message;
    errorContainer.classList.remove("d-none");

    setTimeout(() => {
      errorContainer.classList.add("d-none");
    }, 5000);
  }

  exportData() {
    // Implementa l'esportazione in CSV o PDF
    alert("Funzione di esportazione in sviluppo!");
  }
}

// Avvia l'applicazione
document.addEventListener("DOMContentLoaded", () => {
  new FuelPredictorApp();
});
