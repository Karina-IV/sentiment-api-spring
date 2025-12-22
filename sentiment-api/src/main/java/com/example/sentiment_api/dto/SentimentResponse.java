package com.example.sentiment_api.dto;

public record SentimentResponse(
        String previsao,
        Double probabilidade
) {}
