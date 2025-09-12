use async_openai::{
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest, Role,
    },
    Client,
};
use chrono::{Datelike, NaiveDate, Utc};
use dotenv::dotenv;
use google_generative_ai_rs::v1::{
    api::Client as GeminiClient,
    gemini::{Content, Part},
};
use reqwest::Client as ReqwestClient; // Per DeepSeek
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::env;
use std::error::Error;

#[derive(Debug, Serialize, Clone)]
struct SentimentSummary {
    positive: usize,
    negative: usize,
    neutral: usize,
    overall_score: f64,
}

// Aggiungi queste struct per l'analisi avanzata
#[derive(Debug, Serialize, Clone)]
struct AdvancedMarketContext {
    market_context: MarketContext,
    sentiment_summary: SentimentSummary,
    product_volatilities: HashMap<String, f64>,
    seasonal_adjustment: f64,
    geopolitical_risk_score: f64,
}

#[derive(Debug, Serialize, Clone)]
struct ProductAnalysis {
    name: String,
    current_price: f64,
    predicted_price: f64,
    change_percentage: f64,
    volatility: f64,
    confidence: f64,
    trend_strength: f64,
}

#[derive(Debug, Serialize)]
pub struct FuelPrediction {
    pub prodotto: String,
    pub prezzo_attuale: f64,
    pub prezzo_previsto: f64,
    pub variazione_percentuale: f64,
    pub tendenza: String,
    pub ai_analysis: Option<String>,
    pub confidence: f64,
    pub market_sentiment: Option<String>,
    pub news_impact: Option<f64>,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FuelData {
    #[serde(rename = "DATA_RILEVAZIONE")]
    data_rilevazione: String,
    #[serde(rename = "CODICE_PRODOTTO")]
    codice_prodotto: u32,
    #[serde(rename = "NOME_PRODOTTO")]
    nome_prodotto: String,
    #[serde(rename = "PREZZO")]
    prezzo: String,
    #[serde(rename = "ACCISA")]
    accisa: String,
    #[serde(rename = "IVA")]
    iva: String,
    #[serde(rename = "NETTO")]
    netto: String,
    #[serde(rename = "VARIAZIONE")]
    variazione: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct NewsArticle {
    title: String,
    description: Option<String>,
    url: String,
    published_at: String,
    source: NewsSource,
    sentiment: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct NewsSource {
    name: String,
}

#[derive(Debug, Clone)]
enum AIProvider {
    OpenAI,
    DeepSeek,
    Gemini,
    Statistical, // Fallback
}

// Funzione per determinare il provider in base alle API key disponibili
fn determine_ai_provider() -> AIProvider {
    let openai_key = env::var("OPENAI_API_KEY").unwrap_or_default();
    let deepseek_key = env::var("DEEPSEEK_API_KEY").unwrap_or_default();
    let gemini_key = env::var("GEMINI_API_KEY").unwrap_or_default();

    // Priorità: Gemini (free) > DeepSeek > OpenAI
    if !gemini_key.is_empty() {
        println!("🎯 Selected Gemini provider (free tier)");
        AIProvider::Gemini
    } else if !deepseek_key.is_empty() {
        println!("🎯 Selected DeepSeek provider");
        AIProvider::DeepSeek
    } else if !openai_key.is_empty() {
        println!("🎯 Selected OpenAI provider");
        AIProvider::OpenAI
    } else {
        println!("🎯 No API keys found, using statistical analysis");
        AIProvider::Statistical
    }
}

// Funzioni di supporto
fn parse_date(date_str: &str) -> Result<NaiveDate, Box<dyn Error>> {
    Ok(NaiveDate::parse_from_str(date_str, "%Y-%m-%d")?)
}

fn calculate_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let variance = prices.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;

    variance.sqrt() / mean * 100.0
}

// Funzioni principali
pub async fn fetch_fuel_data() -> Result<Vec<FuelData>, Box<dyn Error>> {
    let url =
        "https://sisen.mase.gov.it/dgsaie/api/v1/weekly-prices/report/export?type=ALL&format=JSON&lang=it";
    println!("📡 Fetching data from: {}", url);

    let response = reqwest::get(url).await?;
    let status = response.status();
    println!("📊 Response status: {}", status);

    if !status.is_success() {
        return Err(format!("API returned error status: {}", status).into());
    }

    // Prima proviamo a leggere come testo per debug
    let text = response.text().await?;
    println!(
        "📄 Raw response (first 500 chars): {}",
        &text[..500.min(text.len())]
    );

    // Proviamo a parsare come JSON generico per vedere la struttura
    // Importa Value da serde_json
    let json_value: serde_json::Value = serde_json::from_str(&text)?;
    println!(
        "🔍 JSON structure: {}",
        serde_json::to_string_pretty(&json_value).unwrap_or("Invalid JSON".to_string())
    );

    // Ora parsiamo come Vec<FuelData>
    let fuel_data: Vec<FuelData> = serde_json::from_str(&text)?;
    Ok(fuel_data)
}

pub async fn fetch_news_articles() -> Result<Vec<NewsArticle>, Box<dyn Error>> {
    dotenv().ok();
    let news_api_key = env::var("NEWS_API_KEY").unwrap_or_default();

    if news_api_key.is_empty() {
        println!("⚠️  News API key non configurata. Utilizzo dati mock.");
        return Ok(get_mock_news().await);
    }

    Ok(get_mock_news().await) // Per ora usiamo mock
}

async fn get_mock_news() -> Vec<NewsArticle> {
    vec![
        NewsArticle {
            title: "Tensioni in Medio Oriente fanno salire il prezzo del petrolio".to_string(),
            description: Some(
                "Le recenti tensioni geopolitiche stanno influenzando i mercati energetici"
                    .to_string(),
            ),
            url: "https://example.com/news1".to_string(),
            published_at: Utc::now().to_rfc3339(),
            source: NewsSource {
                name: "Reuters".to_string(),
            },
            sentiment: Some(-0.7),
        },
        NewsArticle {
            title: "OPEC annuncia taglio della produzione".to_string(),
            description: Some(
                "L'organizzazione dei paesi esportatori di petrolio riduce l'offerta".to_string(),
            ),
            url: "https://example.com/news2".to_string(),
            published_at: Utc::now().to_rfc3339(),
            source: NewsSource {
                name: "Bloomberg".to_string(),
            },
            sentiment: Some(0.8),
        },
    ]
}

fn calculate_sentiment_summary(articles: &[NewsArticle]) -> SentimentSummary {
    let mut positive = 0;
    let mut negative = 0;
    let mut neutral = 0;
    let mut total_score = 0.0;

    for article in articles {
        if let Some(score) = article.sentiment {
            total_score += score;
            if score > 0.3 {
                positive += 1;
            } else if score < -0.3 {
                negative += 1;
            } else {
                neutral += 1;
            }
        }
    }

    let overall_score = if !articles.is_empty() {
        total_score / articles.len() as f64
    } else {
        0.0
    };

    SentimentSummary {
        positive,
        negative,
        neutral,
        overall_score,
    }
}

fn calculate_statistical_predictions(data: &[FuelData]) -> Vec<FuelPrediction> {
    let mut predictions = Vec::new();
    let mut grouped_data: HashMap<String, Vec<(NaiveDate, f64)>> = HashMap::new();

    for entry in data {
        if let Ok(date) = parse_date(&entry.data_rilevazione) {
            if let Ok(price) = entry.prezzo.parse::<f64>() {
                grouped_data
                    .entry(entry.nome_prodotto.clone())
                    .or_insert_with(Vec::new)
                    .push((date, price));
            }
        }
    }

    for (prodotto, mut prices) in grouped_data {
        prices.sort_by(|a, b| a.0.cmp(&b.0));

        if prices.len() < 4 {
            continue;
        }

        let recent_prices: Vec<f64> = prices
            .iter()
            .rev()
            .take(12)
            .map(|(_, price)| *price)
            .collect();

        let current_price = *recent_prices.first().unwrap_or(&0.0);

        // Media mobile
        let window_size = 4;
        let mut moving_avg = 0.0;
        let count = recent_prices.len().min(window_size);

        for i in 0..count {
            moving_avg += recent_prices[i];
        }
        moving_avg /= count as f64;

        // Regressione lineare
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let n = recent_prices.len() as f64;

        for (i, &price) in recent_prices.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += price;
            sum_xy += x * price;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let predicted_change = slope * 4.0;

        let predicted_price = moving_avg + predicted_change;
        let percent_change = ((predicted_price - current_price) / current_price) * 100.0;

        let tendenza = if slope > 0.5 {
            "↗️ In rialzo"
        } else if slope < -0.5 {
            "↘️ In calo"
        } else {
            "➡️ Stabile"
        }
        .to_string();

        // Calcola confidenza basata sulla volatilità
        let volatility = calculate_volatility(&recent_prices);
        let confidence = (100.0 - (volatility * 10.0)).max(0.0).min(100.0);

        predictions.push(FuelPrediction {
            prodotto,
            prezzo_attuale: current_price,
            prezzo_previsto: predicted_price,
            variazione_percentuale: percent_change,
            tendenza,
            ai_analysis: Some("Analisi di esempio basata su dati storici".to_string()),
            confidence,
            market_sentiment: Some("Mercato stabile con tendenza al rialzo".to_string()),
            news_impact: Some(0.3),
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    predictions
}

fn calculate_overall_volatility(predictions: &[FuelPrediction]) -> f64 {
    let volatilities: Vec<f64> = predictions
        .iter()
        .map(|p| {
            let recent_prices = vec![p.prezzo_attuale, p.prezzo_previsto];
            calculate_volatility(&recent_prices)
        })
        .collect();

    volatilities.iter().sum::<f64>() / volatilities.len() as f64
}

fn create_market_context(sentiment: &SentimentSummary, volatility: f64) -> MarketContext {
    MarketContext {
        geopolitical_tensions: calculate_geopolitical_tension(sentiment),
        economic_indicators: 0.6,
        seasonal_factors: calculate_seasonal_factor(),
        news_sentiment: sentiment.overall_score,
        overall_volatility: volatility,
    }
}

fn calculate_geopolitical_tension(sentiment: &SentimentSummary) -> f64 {
    let tension_level = sentiment.negative as f64
        / (sentiment.positive + sentiment.negative + sentiment.neutral) as f64;
    tension_level.max(0.0).min(1.0)
}

fn calculate_seasonal_factor() -> f64 {
    let month = Utc::now().month();
    match month {
        12 | 1 | 2 => 0.8,
        6 | 7 | 8 => 0.4,
        _ => 0.6,
    }
}

fn calculate_news_impact(sentiment: &SentimentSummary, articles: &[NewsArticle]) -> f64 {
    let sentiment_score = sentiment.overall_score;
    let article_count = articles.len() as f64;

    let impact = (article_count / 20.0).min(1.0) * sentiment_score.abs();
    impact.max(0.0).min(1.0)
}

#[derive(Debug, Serialize, Clone)]
struct MarketContext {
    geopolitical_tensions: f64,
    economic_indicators: f64,
    seasonal_factors: f64,
    news_sentiment: f64,
    overall_volatility: f64,
}

async fn get_ai_analysis(
    _fuel_data: &[FuelData],
    _predictions: &[FuelPrediction],
    _market_context: MarketContext,
    _news_articles: Vec<NewsArticle>,
    _sentiment: SentimentSummary,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut analysis_map = HashMap::new();
    analysis_map.insert(
        "GLOBAL".to_string(),
        "Analisi AI simulata per demo".to_string(),
    );
    Ok(analysis_map)
}

async fn get_deepseek_analysis(
    fuel_data: &[FuelData],
    predictions: &[FuelPrediction],
    market_context: MarketContext,
    news_articles: &[NewsArticle],
    sentiment: SentimentSummary,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let deepseek_api_key = env::var("DEEPSEEK_API_KEY")?;
    let client = ReqwestClient::new();

    let enhanced_context = prepare_enhanced_context(
        fuel_data,
        predictions,
        market_context,
        news_articles,
        sentiment,
    );

    let system_prompt = r#"SEI UN ANALISTA ESPERTO DEI MERCATI ENERGETICI ITALIANI ED INTERNAZIONALI.

COMPITI:
1. Analisi globale del mercato carburanti
2. Previsioni specifiche per prodotto
3. Identificazione fattori di rischio
4. Raccomandazioni basate su dati

LINEE GUIDA:
- Usa dati concreti e metriche quantitative
- Considera: geopolitica, stagionalità, trend storici
- Fornisci percentuali e intervalli di confidenza
- Sii realistico e conservativo nelle previsioni
- Analizza in italiano chiaro e professionale

FORMATO:
## ANALISI GLOBALE
[analisi generale]

## PRODOTTO SPECIFICO
[analisi dettagliata per prodotto]

## RISCHI E OPPORTUNITÀ
[lista punti]"#;

    let product_analyses: Vec<ProductAnalysis> = predictions
        .iter()
        .map(|p| ProductAnalysis {
            name: p.prodotto.clone(),
            current_price: p.prezzo_attuale,
            predicted_price: p.prezzo_previsto,
            change_percentage: p.variazione_percentuale,
            volatility: calculate_product_volatility(fuel_data, &p.prodotto),
            confidence: p.confidence,
            trend_strength: p.variazione_percentuale.abs() / 100.0,
        })
        .collect();

    let user_content = format!(
        "Analizza questo scenario mercato carburanti:\n\n\
        CONTESTO MERCATO:\n\
        - Tensione geopolitica: {:.1}%\n\
        - Sentiment news: {:.2} ({} positivo, {} negativo)\n\
        - Fattore stagionale: {:.1}%\n\
        - Volatilità generale: {:.1}%\n\n\
        PREVISIONI ATTUALI:\n{}\n\n\
        ULTIME NOTIZIE:\n{}\n\n\
        Fornisci analisi dettagliata in italiano.",
        enhanced_context.market_context.geopolitical_tensions * 100.0,
        enhanced_context.sentiment_summary.overall_score,
        enhanced_context.sentiment_summary.positive,
        enhanced_context.sentiment_summary.negative,
        enhanced_context.market_context.seasonal_factors * 100.0,
        enhanced_context.market_context.overall_volatility,
        serde_json::to_string_pretty(&product_analyses)?,
        format_news_for_ai(news_articles)
    );

    // Payload per DeepSeek API (adatta in base alla documentazione ufficiale)
    let payload = json!({
        "model": "deepseek-chat", // o il modello appropriato
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "max_tokens": 2000
    });

    let response = client
        .post("https://api.deepseek.com/v1/chat/completions") // URL corretto dell'API
        .header("Authorization", format!("Bearer {}", deepseek_api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let error_text = response.text().await?;
        return Err(format!("DeepSeek API error: {}", error_text).into());
    }

    let response_json: serde_json::Value = response.json().await?;

    let mut analysis_map = HashMap::new();

    if let Some(choices) = response_json["choices"].as_array() {
        if let Some(first_choice) = choices.first() {
            if let Some(content) = first_choice["message"]["content"].as_str() {
                let (global, specific) = parse_ai_response(content, predictions);
                analysis_map.insert("GLOBAL".to_string(), global);

                for (product, analysis) in specific {
                    analysis_map.insert(product, analysis);
                }
            }
        }
    }

    Ok(analysis_map)
}

// 1. FUNZIONE OPENAI AVANZATA
async fn get_openai_analysis(
    fuel_data: &[FuelData],
    predictions: &[FuelPrediction],
    market_context: MarketContext,
    news_articles: &[NewsArticle],
    sentiment: SentimentSummary,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let client = Client::new();

    let enhanced_context = prepare_enhanced_context(
        fuel_data,
        predictions,
        market_context,
        &news_articles,
        sentiment,
    );

    let system_prompt = r#"SEI UN ANALISTA ESPERTO DEI MERCATI ENERGETICI ITALIANI ED INTERNAZIONALI.

COMPITI:
1. Analisi globale del mercato carburanti
2. Previsioni specifiche per prodotto
3. Identificazione fattori di rischio
4. Raccomandazioni basate su dati

LINEE GUIDA:
- Usa dati concreti e metriche quantitative
- Considera: geopolitica, stagionalità, trend storici
- Fornisci percentuali e intervalli di confidenza
- Sii realistico e conservativo nelle previsioni
- Analizza in italiano chiaro e professionale

FORMATO:
## ANALISI GLOBALE
[analisi generale]

## PRODOTTO SPECIFICO
[analisi dettagliata per prodotto]

## RISCHI E OPPORTUNITÀ
[lista punti]"#;

    // Prepare product analyses for AI (since AdvancedMarketContext does not have product_analyses)
    let product_analyses: Vec<ProductAnalysis> = predictions
        .iter()
        .map(|p| ProductAnalysis {
            name: p.prodotto.clone(),
            current_price: p.prezzo_attuale,
            predicted_price: p.prezzo_previsto,
            change_percentage: p.variazione_percentuale,
            volatility: calculate_product_volatility(fuel_data, &p.prodotto),
            confidence: p.confidence,
            trend_strength: p.variazione_percentuale.abs() / 100.0,
        })
        .collect();

    let user_content = format!(
        "Analizza questo scenario mercato carburanti:\n\n\
        CONTESTO MERCATO:\n\
        - Tensione geopolitica: {:.1}%\n\
        - Sentiment news: {:.2} ({} positivo, {} negativo)\n\
        - Fattore stagionale: {:.1}%\n\
        - Volatilità generale: {:.1}%\n\n\
        PREVISIONI ATTUALI:\n{}\n\n\
        ULTIME NOTIZIE:\n{}\n\n\
        Fornisci analisi dettagliata in italiano.",
        enhanced_context.market_context.geopolitical_tensions * 100.0,
        enhanced_context.sentiment_summary.overall_score,
        enhanced_context.sentiment_summary.positive,
        enhanced_context.sentiment_summary.negative,
        enhanced_context.market_context.seasonal_factors * 100.0,
        enhanced_context.market_context.overall_volatility,
        serde_json::to_string_pretty(&product_analyses)?,
        format_news_for_ai(&news_articles)
    );

    let request = CreateChatCompletionRequest {
        model: "gpt-4.1".to_string(),
        messages: vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt.to_string())
                    .build()?,
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_content)
                    .build()?,
            ),
        ],
        // Remove deprecated field `max_tokens`
        temperature: Some(0.2),
        top_p: Some(0.9),
        ..Default::default()
    };

    let response = client.chat().create(request).await?;
    println!("🚀 Using OpenAI advanced analysis: {:?}", response);
    let mut analysis_map = HashMap::new();

    if let Some(choice) = response.choices.first() {
        let content = choice.message.content.clone().unwrap_or_default();

        // Estrai analisi globale e specifiche
        let (global, specific) = parse_ai_response(&content, predictions);

        analysis_map.insert("GLOBAL".to_string(), global);

        for (product, analysis) in specific {
            analysis_map.insert(product, analysis);
        }
    }

    Ok(analysis_map)
}

// 2. FUNZIONE STATISTICA AVANZATA (FALLBACK)
fn get_advanced_statistical_analysis(
    fuel_data: &[FuelData],
    predictions: &[FuelPrediction],
    market_context: MarketContext,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut analysis_map = HashMap::new();

    println!("📊 Using advanced statistical analysis");

    let timestamp = Utc::now().format("%d/%m/%Y %H:%M").to_string();

    // Analisi globale
    let global_analysis = format!(
        "📊 ANALISI STATISTICA AVANZATA\n\
        ⏰ Data: {}\n\
        🎯 Metodo: Modelli statistici avanzati\n\n\
        🌍 CONTESTO MACRO:\n\
        • Tensione geopolitica: {:.1}%\n\
        • Volatilità complessiva: {:.1}%\n\
        • Fattore stagionale: {:.1}%\n\n\
        📈 TENDENZA GENERALE: {}\n\
        🔍 BASATO SU: Analisi di {} dati storici",
        timestamp,
        market_context.geopolitical_tensions * 100.0,
        market_context.overall_volatility,
        market_context.seasonal_factors * 100.0,
        get_overall_trend(predictions),
        fuel_data.len()
    );

    analysis_map.insert("GLOBAL".to_string(), global_analysis);

    // Analisi per prodotto
    for prediction in predictions {
        let product_analysis = format!(
            "⛽ ANALISI STATISTICA: {}\n\
            💰 Prezzo attuale: {:.3}€\n\
            🔮 Prezzo previsto: {:.3}€\n\
            📊 Variazione: {:+.2}%\n\
            🎯 Confidence: {:.1}%\n\
            📈 Trend: {}\n\
            🌊 Volatilità: {:.1}%\n\
            ⚠️  Rischio: {}\n\n\
            💡 METODOLOGIA: Regressione lineare + Media mobile",
            prediction.prodotto,
            prediction.prezzo_attuale,
            prediction.prezzo_previsto,
            prediction.variazione_percentuale,
            prediction.confidence,
            prediction.tendenza,
            calculate_product_volatility(fuel_data, &prediction.prodotto),
            assess_risk_level(prediction)
        );

        analysis_map.insert(prediction.prodotto.clone(), product_analysis);
    }

    Ok(analysis_map)
}

// FUNZIONI DI SUPPORTO PER ANALISI STATISTICA AVANZATA
fn calculate_weighted_confidence(predictions: &[FuelPrediction]) -> f64 {
    let total_weight: f64 = predictions.iter().map(|p| p.prezzo_attuale).sum();
    predictions
        .iter()
        .map(|p| p.confidence * p.prezzo_attuale)
        .sum::<f64>()
        / total_weight
}

fn calculate_average_rsi(fuel_data: &[FuelData]) -> f64 {
    // Implementazione semplificata RSI
    let prices: Vec<f64> = fuel_data
        .iter()
        .filter_map(|d| d.prezzo.parse::<f64>().ok())
        .collect();

    if prices.len() < 14 {
        return 50.0;
    }

    let mut gains = 0.0;
    let mut losses = 0.0;

    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains += change;
        } else {
            losses += change.abs();
        }
    }

    let avg_gain = gains / 14.0;
    let avg_loss = losses / 14.0;

    if avg_loss == 0.0 {
        return 100.0;
    }

    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

fn calculate_market_momentum(predictions: &[FuelPrediction]) -> String {
    let avg_change: f64 = predictions
        .iter()
        .map(|p| p.variazione_percentuale)
        .sum::<f64>()
        / predictions.len() as f64;

    match avg_change {
        x if x > 2.0 => "FORTE RIALZISTA",
        x if x > 0.5 => "LEGGERMENTE RIALZISTA",
        x if x > -0.5 => "NEUTRA",
        x if x > -2.0 => "LEGGERMENTE RIBASSISTA",
        _ => "FORTE RIBASSISTA",
    }
    .to_string()
}

fn identify_support_resistance(fuel_data: &[FuelData]) -> String {
    let prices: Vec<f64> = fuel_data
        .iter()
        .filter_map(|d| d.prezzo.parse::<f64>().ok())
        .collect();

    if prices.len() < 20 {
        return "Dati insufficienti".to_string();
    }

    let recent_prices = &prices[prices.len() - 20..];
    let support = recent_prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let resistance = recent_prices
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    format!("Supporto: {:.3}€, Resistenza: {:.3}€", support, resistance)
}

fn calculate_product_volatility(fuel_data: &[FuelData], product: &str) -> f64 {
    let product_prices: Vec<f64> = fuel_data
        .iter()
        .filter(|d| d.nome_prodotto == product)
        .filter_map(|d| d.prezzo.parse::<f64>().ok())
        .collect();

    calculate_volatility(&product_prices)
}

fn assess_risk_level(prediction: &FuelPrediction) -> String {
    match prediction.confidence {
        x if x > 80.0 => "BASSO",
        x if x > 60.0 => "MODERATO",
        x if x > 40.0 => "MEDIO",
        x if x > 20.0 => "ALTO",
        _ => "MOLTO ALTO",
    }
    .to_string()
}

fn calculate_bollinger_signal(_fuel_data: &[FuelData], _product: &str) -> String {
    // Implementazione semplificata
    "Neutro".to_string()
}

fn calculate_macd_signal(_fuel_data: &[FuelData], _product: &str) -> String {
    // Implementazione semplificata
    "Neutro".to_string()
}

fn calculate_volume_ratio(_fuel_data: &[FuelData]) -> f64 {
    // Placeholder per analisi volume
    1.0
}

fn get_overall_trend(predictions: &[FuelPrediction]) -> String {
    let bullish = predictions
        .iter()
        .filter(|p| p.variazione_percentuale > 0.0)
        .count();
    let bearish = predictions
        .iter()
        .filter(|p| p.variazione_percentuale < 0.0)
        .count();

    match (bullish, bearish) {
        (b, e) if b > e * 2 => "FORTEMENTE RIALZISTA",
        (b, e) if b > e => "RIALZISTA",
        (b, e) if e > b * 2 => "FORTEMENTE RIBASSISTA",
        (b, e) if e > b => "RIBASSISTA",
        _ => "NEUTRA",
    }
    .to_string()
}

// FUNZIONI DI PREPARAZIONE CONTESTO
fn prepare_enhanced_context(
    fuel_data: &[FuelData],
    predictions: &[FuelPrediction],
    market_context: MarketContext,
    news_articles: &[NewsArticle],
    sentiment: SentimentSummary,
) -> AdvancedMarketContext {
    let _product_analyses: Vec<ProductAnalysis> = predictions
        .iter()
        .map(|p| ProductAnalysis {
            name: p.prodotto.clone(),
            current_price: p.prezzo_attuale,
            predicted_price: p.prezzo_previsto,
            change_percentage: p.variazione_percentuale,
            volatility: calculate_product_volatility(fuel_data, &p.prodotto),
            confidence: p.confidence,
            trend_strength: p.variazione_percentuale.abs() / 100.0,
        })
        .collect();

    AdvancedMarketContext {
        market_context,
        sentiment_summary: sentiment,
        product_volatilities: calculate_all_volatilities(fuel_data, predictions),
        seasonal_adjustment: calculate_seasonal_adjustment(),
        geopolitical_risk_score: calculate_geopolitical_risk(&news_articles),
    }
}

fn calculate_all_volatilities(
    fuel_data: &[FuelData],
    predictions: &[FuelPrediction],
) -> HashMap<String, f64> {
    predictions
        .iter()
        .map(|p| {
            (
                p.prodotto.clone(),
                calculate_product_volatility(fuel_data, &p.prodotto),
            )
        })
        .collect()
}

fn calculate_seasonal_adjustment() -> f64 {
    let month = Utc::now().month();
    match month {
        12 | 1 | 2 => 1.2, // Inverno: +20%
        6 | 7 | 8 => 0.8,  // Estate: -20%
        _ => 1.0,          // Normale
    }
}

fn calculate_geopolitical_risk(news_articles: &[NewsArticle]) -> f64 {
    let risk_keywords = ["war", "tension", "sanction", "opec", "embargo", "crisis"];
    let risk_count = news_articles
        .iter()
        .filter(|article| {
            risk_keywords.iter().any(|kw| {
                article.title.to_lowercase().contains(kw)
                    || article
                        .description
                        .as_ref()
                        .map_or(false, |d| d.to_lowercase().contains(kw))
            })
        })
        .count();

    (risk_count as f64 / news_articles.len().max(1) as f64).min(1.0)
}

fn format_news_for_ai(news_articles: &[NewsArticle]) -> String {
    news_articles
        .iter()
        .take(5)
        .map(|article| {
            format!(
                "• {} (Sentiment: {:.2}) - {}",
                article.title,
                article.sentiment.unwrap_or(0.0),
                article.source.name
            )
        })
        .collect::<Vec<String>>()
        .join("\n")
}

fn parse_ai_response(
    response: &str,
    predictions: &[FuelPrediction],
) -> (String, HashMap<String, String>) {
    let mut specific_analyses = HashMap::new();
    let mut global_analysis = String::new();

    // Estrazione semplice (implementazione base)
    let lines: Vec<&str> = response.lines().collect();
    let mut current_section = String::new();

    for line in lines {
        if line.starts_with("## ") {
            current_section = line.to_string();
        } else if current_section.contains("GLOBALE") {
            global_analysis.push_str(line);
            global_analysis.push('\n');
        } else {
            for prediction in predictions {
                if line.contains(&prediction.prodotto) {
                    specific_analyses.insert(prediction.prodotto.clone(), line.to_string());
                }
            }
        }
    }

    (global_analysis, specific_analyses)
}
pub async fn analyze_fuel_prices() -> Result<Vec<FuelPrediction>, Box<dyn Error>> {
    let fuel_data = fetch_fuel_data().await?;
    let news_articles = fetch_news_articles().await?;
    let sentiment = calculate_sentiment_summary(&news_articles);

    let mut predictions = calculate_statistical_predictions(&fuel_data);
    let volatility = calculate_overall_volatility(&predictions);
    let market_context = create_market_context(&sentiment, volatility);

    // Determina automaticamente il provider
    let provider = determine_ai_provider();

    let ai_analysis = match provider {
        AIProvider::Gemini => {
            println!("🚀 Using Gemini detailed analysis");
            match get_gemini_analysis(
                &fuel_data,
                &predictions,
                market_context.clone(),
                &news_articles,
                &sentiment,
            )
            .await
            {
                Ok(analysis) => {
                    println!("✅ Gemini detailed analysis successful");
                    analysis
                }
                Err(e) => {
                    println!("⚠️ Gemini analysis failed: {}. Falling back", e);
                    get_advanced_statistical_analysis(
                        &fuel_data,
                        &predictions,
                        market_context.clone(),
                    )?
                }
            }
        }
        AIProvider::DeepSeek => {
            println!("🚀 Using DeepSeek analysis");
            match get_deepseek_analysis(
                &fuel_data,
                &predictions,
                market_context.clone(),
                &news_articles,
                sentiment,
            )
            .await
            {
                Ok(analysis) => {
                    println!("✅ DeepSeek analysis successful");
                    analysis
                }
                Err(e) => {
                    println!(
                        "⚠️ DeepSeek analysis failed: {}. Falling back to statistical",
                        e
                    );
                    get_advanced_statistical_analysis(
                        &fuel_data,
                        &predictions,
                        market_context.clone(),
                    )?
                }
            }
        }
        AIProvider::OpenAI => {
            println!("🚀 Using OpenAI analysis");
            match get_openai_analysis(
                &fuel_data,
                &predictions,
                market_context.clone(),
                &news_articles,
                sentiment,
            )
            .await
            {
                Ok(analysis) => {
                    println!("✅ OpenAI analysis successful");
                    analysis
                }
                Err(e) => {
                    println!(
                        "⚠️ OpenAI analysis failed: {}. Falling back to statistical",
                        e
                    );
                    get_advanced_statistical_analysis(
                        &fuel_data,
                        &predictions,
                        market_context.clone(),
                    )?
                }
            }
        }
        AIProvider::Statistical => {
            println!("📊 Using advanced statistical analysis (no API keys)");
            get_advanced_statistical_analysis(&fuel_data, &predictions, market_context.clone())?
        }
    };

    // Aggiorna le previsioni
    for prediction in &mut predictions {
        if let Some(analysis) = ai_analysis.get(&prediction.prodotto) {
            prediction.ai_analysis = Some(analysis.clone());
        }

        // Aggiungi analisi globale e info provider
        prediction.market_sentiment = ai_analysis.get("GLOBAL").cloned();
        prediction.ai_analysis = prediction.ai_analysis.as_ref().map(|analysis| {
            format!(
                "{} [Provider: {} | News: {} articoli]",
                analysis,
                match provider {
                    AIProvider::Gemini => "Gemini (Free)",
                    AIProvider::DeepSeek => "DeepSeek",
                    AIProvider::OpenAI => "OpenAI",
                    AIProvider::Statistical => "Statistical",
                },
                news_articles.len()
            )
        });

        // Aggiungi impatto news calcolato
        prediction.news_impact = Some(calculate_news_impact_for_product(
            &prediction.prodotto,
            news_articles.as_slice(),
        ));
    }

    Ok(predictions)
}

async fn get_gemini_analysis(
    fuel_data: &[FuelData],
    predictions: &[FuelPrediction],
    market_context: MarketContext,
    news_articles: &[NewsArticle],
    sentiment: &SentimentSummary,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let gemini_api_key = env::var("GEMINI_API_KEY")?;
    let client = ReqwestClient::new();

    // Prepara analisi dettagliate per ogni prodotto
    let product_analyses: Vec<ProductAnalysis> = predictions
        .iter()
        .map(|p| ProductAnalysis {
            name: p.prodotto.clone(),
            current_price: p.prezzo_attuale,
            predicted_price: p.prezzo_previsto,
            change_percentage: p.variazione_percentuale,
            volatility: calculate_product_volatility(fuel_data, &p.prodotto),
            confidence: p.confidence,
            trend_strength: p.variazione_percentuale.abs() / 100.0,
        })
        .collect();

    // Prepara dati storici per ogni prodotto
    let historical_data = prepare_historical_analysis(fuel_data, predictions);

    // Formatta le notizie con sentiment
    let formatted_news = format_news_with_sentiment(news_articles);

    let user_content = format!(
        "ANALISI COMPLETA MERCATO CARBURANTI ITALIANO\n\n\
        📊 CONTESTO MACROECONOMICO:\n\
        • Tensione geopolitica: {:.1}%\n\
        • Sentiment notizie: {:.2} ({} positivo, {} negativo, {} neutro)\n\
        • Fattore stagionale: {:.1}%\n\
        • Volatilità generale: {:.1}%\n\n\
        📈 DATI PREVISIONALI DETTAGLIATI:\n{}\n\n\
        📜 STORICO RECENTE:\n{}\n\n\
        📰 ULTIME NOTIZIE RILEVANTI:\n{}\n\n\
        🔍 ANALISI RICHIESTA:\n\
        Per OGNI tipo di carburante, fornire:\n\
        1. Analisi specifica del prodotto\n\
        2. Impatto delle notizie su quel prodotto\n\
        3. Fattori di rischio specifici\n\
        4. Previsione a breve termine (7-15 giorni)\n\
        5. Livello di confidenza\n\n\
        Formato italiano, massimo 300 caratteri per prodotto.",
        market_context.geopolitical_tensions * 100.0,
        sentiment.overall_score,
        sentiment.positive,
        sentiment.negative,
        sentiment.neutral,
        market_context.seasonal_factors * 100.0,
        market_context.overall_volatility,
        serde_json::to_string_pretty(&product_analyses)?,
        historical_data,
        formatted_news
    );

    let system_prompt = r#"SEI UN ANALISTA ESPERTO DEI MERCATI ENERGETICI ITALIANI ED INTERNAZIONALI.

COMPITI SPECIFICI:
1. Analisi DETTAGLIATA per OGNI tipo di carburante
2. Collegamento diretto tra notizie e impatto sui prezzi
3. Valutazione rischio specifica per prodotto
4. Previsioni a breve termine (7-15 giorni)

FORMATO OBBLIGATORIO PER OGNI PRODOTTO:
## [NOME PRODOTTO]
• Analisi: [breve analisi specifica]
• Notizie: [impatto notizie su questo prodotto]
• Rischio: [livello rischio]
• Previsione: [tendenza 7-15 giorni]
• Confidence: [livello confidenza]

Mantenere analisi concise ma complete (max 300 caratteri per prodotto)."#;

    let full_prompt = format!("{}\n\n{}", system_prompt, user_content);

    let url = format!(
        "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={}",
        gemini_api_key
    );

    let payload = json!({
        "contents": [{
            "parts": [{
                "text": full_prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,  // Più basso per analisi più precise
            "topP": 0.8,
            "maxOutputTokens": 4000,  // Aumentato per analisi multiple
            "topK": 32
        }
    });

    println!("🌐 Invio richiesta analisi dettagliata a Gemini...");

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !response.status().is_success() {
        let error_text = response.text().await?;
        return Err(format!("Gemini API error: {}", error_text).into());
    }

    let response_json: serde_json::Value = response.json().await?;
    println!("✅ Gemini response received");

    let mut analysis_map = HashMap::new();

    if let Some(text) = extract_gemini_text(&response_json) {
        println!("📝 Gemini analysis received: {} characters", text.len());

        // Parsing avanzato per estrarre analisi per ogni prodotto
        let analyses = parse_detailed_ai_response(&text, predictions);

        for (product, analysis) in analyses {
            analysis_map.insert(product, analysis);
        }

        // Aggiungi analisi globale se presente
        if let Some(global) = extract_global_analysis(&text) {
            analysis_map.insert("GLOBAL".to_string(), global);
        }
    }

    Ok(analysis_map)
}

// Nuove funzioni di supporto per l'analisi dettagliata
fn prepare_historical_analysis(fuel_data: &[FuelData], predictions: &[FuelPrediction]) -> String {
    let mut result = String::new();

    for prediction in predictions {
        let product_data: Vec<&FuelData> = fuel_data
            .iter()
            .filter(|d| d.nome_prodotto == prediction.prodotto)
            .collect();

        if product_data.len() >= 4 {
            let recent_prices: Vec<f64> = product_data
                .iter()
                .rev()
                .take(8)
                .filter_map(|d| d.prezzo.parse::<f64>().ok())
                .collect();

            if !recent_prices.is_empty() {
                let avg_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
                let min_price = recent_prices.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_price = recent_prices
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);

                result.push_str(&format!(
                    "• {}: {:.3}€ (min {:.3}€, max {:.3}€, {} dati)\n",
                    prediction.prodotto,
                    avg_price,
                    min_price,
                    max_price,
                    recent_prices.len()
                ));
            }
        }
    }

    result
}

fn format_news_with_sentiment(news_articles: &[NewsArticle]) -> String {
    news_articles
        .iter()
        .take(8) // Limita a 8 notizie più rilevanti
        .map(|article| {
            let sentiment_emoji = match article.sentiment {
                Some(s) if s > 0.3 => "✅",
                Some(s) if s < -0.3 => "❌",
                _ => "➡️",
            };

            format!(
                "{} {}: {}",
                sentiment_emoji, article.source.name, article.title
            )
        })
        .collect::<Vec<String>>()
        .join("\n")
}

fn extract_gemini_text(response_json: &serde_json::Value) -> Option<String> {
    response_json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .map(|s| s.to_string())
}

fn parse_detailed_ai_response(
    response: &str,
    predictions: &[FuelPrediction],
) -> HashMap<String, String> {
    let mut analyses = HashMap::new();
    let lines: Vec<&str> = response.lines().collect();

    let mut current_product = String::new();
    let mut current_analysis = String::new();

    for line in lines {
        if line.starts_with("## ") {
            // Salva l'analisi del prodotto precedente
            if !current_product.is_empty() && !current_analysis.is_empty() {
                analyses.insert(current_product.clone(), current_analysis.trim().to_string());
            }

            // Nuovo prodotto
            current_product = line.replace("##", "").trim().to_string();
            current_analysis = String::new();
        } else if !current_product.is_empty() {
            current_analysis.push_str(line);
            current_analysis.push('\n');
        }
    }

    // Aggiungi l'ultima analisi
    if !current_product.is_empty() && !current_analysis.is_empty() {
        analyses.insert(current_product, current_analysis.trim().to_string());
    }

    // Assicurati che ogni prediction abbia un'analisi
    for prediction in predictions {
        if !analyses.contains_key(&prediction.prodotto) {
            analyses.insert(
                prediction.prodotto.clone(),
                "Analisi non disponibile".to_string(),
            );
        }
    }

    analyses
}

fn extract_global_analysis(response: &str) -> Option<String> {
    let lines: Vec<&str> = response.lines().collect();
    let mut global_analysis = String::new();
    let mut in_global_section = false;

    for line in lines {
        if line.contains("ANALISI GLOBALE") || line.contains("CONTESTO GENERALE") {
            in_global_section = true;
            continue;
        }

        if in_global_section {
            if line.starts_with("## ") {
                break; // Inizia nuova sezione
            }
            global_analysis.push_str(line);
            global_analysis.push('\n');
        }
    }

    if !global_analysis.trim().is_empty() {
        Some(global_analysis.trim().to_string())
    } else {
        None
    }
}

// Funzione per provare modelli alternativi
async fn try_alternative_gemini_models(
    fuel_data: &[FuelData],
    predictions: &[FuelPrediction],
    market_context: MarketContext,
    news_articles: &[NewsArticle],
    sentiment: &SentimentSummary,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let gemini_api_key = env::var("GEMINI_API_KEY")?;
    let client = ReqwestClient::new();

    let models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
        "models/gemini-pro",
    ];

    for model in models_to_try.iter() {
        println!("🔄 Trying model: {}", model);

        let url = format!(
            "https://generativelanguage.googleapis.com/v1/models/{}:generateContent?key={}",
            model, gemini_api_key
        );

        let simple_prompt = "Analizza brevemente il mercato carburanti italiano. Rispondi in italiano in massimo 200 caratteri.";

        let payload = json!({
            "contents": [{
                "parts": [{
                    "text": simple_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 300
            }
        });

        let response = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                println!("✅ Model {} works!", model);
                // Se funziona, usa questo modello per la richiesta completa
                return get_gemini_analysis_with_model(
                    fuel_data,
                    predictions,
                    market_context,
                    news_articles,
                    sentiment,
                    model,
                )
                .await;
            }
            Ok(resp) => {
                println!("❌ Model {} failed: {}", model, resp.status());
            }
            Err(e) => {
                println!("❌ Model {} error: {}", model, e);
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    Err("All Gemini models failed".into())
}

async fn get_gemini_analysis_with_model(
    fuel_data: &[FuelData],
    predictions: &[FuelPrediction],
    market_context: MarketContext,
    news_articles: &[NewsArticle],
    sentiment: &SentimentSummary,
    model: &str,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let gemini_api_key = env::var("GEMINI_API_KEY")?;
    let client = ReqwestClient::new();

    let url = format!(
        "https://generativelanguage.googleapis.com/v1/models/{}:generateContent?key={}",
        model, gemini_api_key
    );

    let product_analyses: Vec<ProductAnalysis> = predictions
        .iter()
        .map(|p| ProductAnalysis {
            name: p.prodotto.clone(),
            current_price: p.prezzo_attuale,
            predicted_price: p.prezzo_previsto,
            change_percentage: p.variazione_percentuale,
            volatility: calculate_product_volatility(fuel_data, &p.prodotto),
            confidence: p.confidence,
            trend_strength: p.variazione_percentuale.abs() / 100.0,
        })
        .collect();

    let user_content = format!(
        "Analizza questo scenario mercato carburanti:\n\n\
        CONTESTO MERCATO:\n\
        - Tensione geopolitica: {:.1}%\n\
        - Sentiment news: {:.2} ({} positivo, {} negativo)\n\
        - Fattore stagionale: {:.1}%\n\
        - Volatilità generale: {:.1}%\n\n\
        PREVISIONI ATTUALI:\n{}\n\n\
        Fornisci analisi dettagliata in italiano in massimo 500 caratteri.",
        market_context.geopolitical_tensions * 100.0,
        sentiment.overall_score,
        sentiment.positive,
        sentiment.negative,
        market_context.seasonal_factors * 100.0,
        market_context.overall_volatility,
        serde_json::to_string_pretty(&product_analyses)?
    );

    let system_prompt =
        r#"SEI UN ANALISTA ESPERTO DEI MERCATI ENERGETICI. Analizza brevemente in italiano."#;

    let full_prompt = format!("{}\n\n{}", system_prompt, user_content);

    let payload = json!({
        "contents": [{
            "parts": [{
                "text": full_prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 1000,  // Ridotto per efficienza
            "topK": 40
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    });

    println!("🌐 Invio richiesta a Gemini con modello {}...", model);

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    let status = response.status();
    println!("📊 Status response: {}", status);

    if !status.is_success() {
        let error_text = response.text().await?;
        println!("❌ Gemini API error: {}", error_text);

        // Prova con un modello alternativo se il primo fallisce
        if error_text.contains("not found") {
            return Box::pin(try_alternative_gemini_models(
                fuel_data,
                predictions,
                market_context,
                news_articles,
                sentiment,
            ))
            .await;
        }

        return Err(format!("Gemini API error: {} - {}", status, error_text).into());
    }

    let response_json: serde_json::Value = response.json().await?;
    println!("✅ Gemini response received");

    // Debug: stampa la struttura della response
    println!("📋 Response structure: {:?}", response_json);

    let mut analysis_map = HashMap::new();

    // Estrazione corretta del testo dalla response
    if let Some(candidates) = response_json["candidates"].as_array() {
        if let Some(first_candidate) = candidates.first() {
            if let Some(content_parts) = first_candidate["content"]["parts"].as_array() {
                if let Some(first_part) = content_parts.first() {
                    if let Some(text) = first_part["text"].as_str() {
                        println!("📝 Gemini analysis: {} characters", text.len());

                        let (global, specific) = parse_ai_response(text, predictions);
                        analysis_map.insert("GLOBAL".to_string(), global);

                        for (product, analysis) in specific {
                            analysis_map.insert(product, analysis);
                        }
                    } else {
                        println!("❌ No text in part: {:?}", first_part);
                    }
                }
            } else {
                println!("❌ No parts in content: {:?}", first_candidate["content"]);
            }
        }
    } else {
        println!(
            "❌ No candidates in response, trying different path: {:?}",
            response_json
        );

        // Prova un percorso alternativo per l'estrazione
        if let Some(text) = response_json["candidates"][0]["content"]["parts"][0]["text"].as_str() {
            println!("✅ Found text via alternative path");
            let (global, specific) = parse_ai_response(text, predictions);
            analysis_map.insert("GLOBAL".to_string(), global);
            for (product, analysis) in specific {
                analysis_map.insert(product, analysis);
            }
        } else {
            return Err("Could not extract text from Gemini response".into());
        }
    }

    Ok(analysis_map)
}

// Funzione per listare i modelli disponibili (debug)
async fn list_available_models() -> Result<(), Box<dyn Error>> {
    let gemini_api_key = env::var("GEMINI_API_KEY")?;
    let client = ReqwestClient::new();

    let url = format!(
        "https://generativelanguage.googleapis.com/v1/models?key={}",
        gemini_api_key
    );

    let response = client.get(&url).send().await?;

    if response.status().is_success() {
        let models: serde_json::Value = response.json().await?;
        println!(
            "📋 Available models: {}",
            serde_json::to_string_pretty(&models)?
        );
    } else {
        println!("❌ Failed to list models: {}", response.status());
    }

    Ok(())
}

// Funzione per calcolare l'impatto delle news specifico per prodotto
fn calculate_news_impact_for_product(product: &str, news_articles: &[NewsArticle]) -> f64 {
    let product_keywords = match product.to_lowercase().as_str() {
        s if s.contains("diesel") => vec!["diesel", "gasolio", "transport", "camion"],
        s if s.contains("benzina") => vec!["benzina", "petrol", "gasoline", "auto"],
        s if s.contains("gpl") => vec!["gpl", "lpg", "gas", "auto"],
        s if s.contains("metano") => vec!["metano", "cng", "gas", "auto"],
        _ => vec!["carburante", "fuel", "energy", "petrolio"],
    };

    let relevant_articles = news_articles
        .iter()
        .filter(|article| {
            product_keywords.iter().any(|kw| {
                article.title.to_lowercase().contains(kw)
                    || article
                        .description
                        .as_ref()
                        .map_or(false, |d| d.to_lowercase().contains(kw))
            })
        })
        .count();

    (relevant_articles as f64 / news_articles.len().max(1) as f64).min(1.0)
}
