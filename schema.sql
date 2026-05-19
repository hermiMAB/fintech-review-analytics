-- PostgreSQL Data Warehouse Schema Blueprint
-- Database Target Name: bank_reviews

CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) UNIQUE NOT NULL,
    app_name VARCHAR(100)
);

CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INT REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT NOT NULL,
    rating INT CHECK (rating BETWEEN 1 AND 5),
    review_date DATE,
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    identified_theme TEXT,
    source VARCHAR(50)
);

-- =================================================================
-- INTEGRITY QUALITY ASSURANCE TESTING QUERIES
-- Run these in your pgAdmin Query Tool to audit database ingestion
-- =================================================================

-- Query 1: Total review volume records inserted grouped per bank
SELECT b.bank_name, COUNT(r.review_id) as total_reviews
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
GROUP BY b.bank_name;

-- Query 2: Mean user rating profile calculations per competitor
SELECT b.bank_name, ROUND(AVG(r.rating), 2) as average_star_rating
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
GROUP BY b.bank_name;

-- Query 3: Audit check verifying zero null inputs on critical telemetry paths
SELECT 
    COUNT(*) FILTER (WHERE review_text IS NULL) as null_texts,
    COUNT(*) FILTER (WHERE sentiment_label IS NULL) as null_sentiments,
    COUNT(*) FILTER (WHERE rating IS NULL) as null_ratings
FROM reviews;