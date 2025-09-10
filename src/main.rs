use crate::predictor::FuelPrediction;
use actix_cors::Cors;
use actix_files as files;
use actix_web::{get, post, App, HttpResponse, HttpServer};
use serde::Serialize;

mod predictor;
use predictor::{analyze_fuel_prices, fetch_news_articles};

#[derive(Debug, Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[get("/api/health")]
async fn health_check() -> HttpResponse {
    HttpResponse::Ok().json(ApiResponse {
        success: true,
        data: Some("Server is running"),
        error: None,
    })
}

#[post("/api/predict")]
async fn predict_fuel_prices() -> HttpResponse {
    match analyze_fuel_prices().await {
        Ok(predictions) => HttpResponse::Ok().json(ApiResponse {
            success: true,
            data: Some(predictions),
            error: None,
        }),
        Err(e) => HttpResponse::InternalServerError().json(ApiResponse::<Vec<FuelPrediction>> {
            success: false,
            data: None,
            error: Some(e.to_string()),
        }),
    }
}

#[get("/api/news")]
async fn get_news() -> HttpResponse {
    match fetch_news_articles().await {
        Ok(news) => HttpResponse::Ok().json(ApiResponse {
            success: true,
            data: Some(news),
            error: None,
        }),
        Err(e) => {
            HttpResponse::InternalServerError().json(ApiResponse::<Vec<predictor::NewsArticle>> {
                success: false,
                data: None,
                error: Some(e.to_string()),
            })
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("🚀 Starting Fuel Predictor Web Server on http://localhost:8083");

    HttpServer::new(|| {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(cors)
            .service(health_check)
            .service(predict_fuel_prices)
            .service(get_news)
            .service(files::Files::new("/", "./static").index_file("index.html"))
    })
    .bind("127.0.0.1:8083")?
    .run()
    .await
}
