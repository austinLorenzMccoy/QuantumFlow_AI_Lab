import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.sentiment_analysis_service import SentimentAnalysisService


@pytest.fixture
def sentiment_service():
    """Create a sentiment analysis service instance for testing."""
    return SentimentAnalysisService(websocket_channel="test_sentiment")


@pytest.mark.asyncio
async def test_sentiment_service_initialization(sentiment_service):
    """Test sentiment service initialization."""
    with patch.object(sentiment_service, '_collect_news_sentiment', new_callable=AsyncMock):
        with patch.object(sentiment_service, '_collect_social_sentiment', new_callable=AsyncMock):
            await sentiment_service.initialize()
            assert sentiment_service.is_initialized


@pytest.mark.asyncio
async def test_get_sentiment_data(sentiment_service):
    """Test getting sentiment data for a cryptocurrency."""
    # Mock sentiment data
    mock_sentiment = {
        "symbol": "BTC",
        "sentiment_score": 0.75,
        "sentiment_label": "bullish",
        "volume": 1500,
        "trend": 0.12,
        "sources": {
            "news": {"count": 50, "avg_sentiment": 0.8},
            "social": {"count": 1450, "avg_sentiment": 0.74}
        }
    }
    
    sentiment_service.sentiment_data["BTC"] = mock_sentiment
    
    result = await sentiment_service.get_sentiment("BTC")
    
    assert result is not None
    assert result.symbol == "BTC"
    assert result.sentiment_score == 0.75
    assert result.sentiment_label == "bullish"


@pytest.mark.asyncio
async def test_get_all_sentiment_data(sentiment_service):
    """Test getting all sentiment data."""
    # Mock multiple cryptocurrency sentiment data
    mock_data = {
        "BTC": {
            "symbol": "BTC",
            "sentiment_score": 0.75,
            "sentiment_label": "bullish",
            "volume": 1500,
            "trend": 0.12
        },
        "ETH": {
            "symbol": "ETH",
            "sentiment_score": 0.65,
            "sentiment_label": "bullish",
            "volume": 1200,
            "trend": 0.08
        }
    }
    
    sentiment_service.sentiment_data = mock_data
    
    result = await sentiment_service.get_all_sentiment()
    
    assert len(result) == 2
    assert any(s.symbol == "BTC" for s in result)
    assert any(s.symbol == "ETH" for s in result)


@pytest.mark.asyncio
async def test_generate_trading_signals(sentiment_service):
    """Test generating trading signals based on sentiment."""
    # Mock sentiment data
    mock_data = {
        "BTC": {
            "sentiment_score": 0.85,  # Very bullish
            "sentiment_label": "very_bullish",
            "volume": 2000,
            "trend": 0.15
        },
        "ETH": {
            "sentiment_score": 0.25,  # Bearish
            "sentiment_label": "bearish",
            "volume": 1500,
            "trend": -0.10
        }
    }
    
    sentiment_service.sentiment_data = mock_data
    
    signals = await sentiment_service.get_trading_signals()
    
    assert len(signals) > 0
    
    # Check for BTC buy signal (very bullish sentiment)
    btc_signals = [s for s in signals if s["symbol"] == "BTC"]
    assert len(btc_signals) > 0
    assert btc_signals[0]["signal"] == "BUY"


@pytest.mark.asyncio
async def test_sentiment_classification(sentiment_service):
    """Test sentiment score classification."""
    # Test various sentiment scores
    test_cases = [
        (0.9, "very_bullish"),
        (0.7, "bullish"),
        (0.55, "neutral"),
        (0.3, "bearish"),
        (0.1, "very_bearish")
    ]
    
    for score, expected_label in test_cases:
        label = sentiment_service._classify_sentiment(score)
        assert label == expected_label


if __name__ == "__main__":
    pytest.main([__file__])
