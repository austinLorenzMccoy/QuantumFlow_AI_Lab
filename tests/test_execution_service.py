import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.execution_service import ExecutionService
from app.models.strategy import Strategy, StrategyType, RiskTolerance, StrategyPerformance


@pytest.fixture
def sample_strategy():
    """Create a sample strategy for testing."""
    return Strategy(
        id="test-strategy-1",
        name="Test RSI Strategy",
        description="Test strategy for RSI",
        type=StrategyType.MEAN_REVERSION,
        trading_pairs=["ETH-USDT"],
        risk_tolerance=RiskTolerance.MEDIUM,
        indicators=[
            {
                "name": "RSI",
                "parameters": {"period": 14, "overbought": 70, "oversold": 30}
            }
        ],
        rules=[
            {
                "name": "Buy when oversold",
                "conditions": [
                    {
                        "indicator": "RSI",
                        "operator": "<",
                        "value": 30
                    }
                ],
                "actions": [
                    {
                        "type": "BUY",
                        "asset": "ETH",
                        "amount_type": "percentage",
                        "amount": 50.0
                    }
                ]
            },
            {
                "name": "Sell when overbought",
                "conditions": [
                    {
                        "indicator": "RSI",
                        "operator": ">",
                        "value": 70
                    }
                ],
                "actions": [
                    {
                        "type": "SELL",
                        "asset": "ETH",
                        "amount_type": "percentage",
                        "amount": 100.0
                    }
                ]
            }
        ],
        parameters={
            "stop_loss": 5.0,
            "take_profit": 10.0,
            "position_size": 0.1
        }
    )


@pytest.fixture
def execution_service():
    """Create an execution service instance for testing."""
    return ExecutionService()


@pytest.mark.asyncio
async def test_deploy_strategy(execution_service, sample_strategy):
    """Test deploying a strategy."""
    result = await execution_service.deploy_strategy(sample_strategy)
    
    assert result["status"] == "success"
    assert result["strategy_id"] == sample_strategy.id
    assert sample_strategy.id in execution_service.active_strategies
    assert execution_service.active_strategies[sample_strategy.id]["status"] == "running"


@pytest.mark.asyncio
async def test_stop_strategy(execution_service, sample_strategy):
    """Test stopping a strategy."""
    # First deploy
    await execution_service.deploy_strategy(sample_strategy)
    
    # Then stop
    result = await execution_service.stop_strategy(sample_strategy.id)
    
    assert result["status"] == "success"
    assert execution_service.active_strategies[sample_strategy.id]["status"] == "stopped"


@pytest.mark.asyncio
async def test_get_strategy_performance(execution_service, sample_strategy):
    """Test getting strategy performance."""
    # First deploy
    await execution_service.deploy_strategy(sample_strategy)
    
    # Add some mock trades
    execution_service.active_strategies[sample_strategy.id]["trades"] = [
        {
            "type": "buy",
            "trading_pair": "ETH-USDT",
            "price": 1000,
            "amount": 0.1,
            "timestamp": time.time() - 3600
        },
        {
            "type": "sell",
            "trading_pair": "ETH-USDT",
            "price": 1100,
            "amount": 0.1,
            "timestamp": time.time()
        }
    ]
    
    performance = await execution_service.get_strategy_performance(sample_strategy.id)
    
    assert performance["strategy_id"] == sample_strategy.id
    assert "total_return" in performance
    assert "total_trades" in performance
    assert performance["total_trades"] == 2


@pytest.mark.asyncio
async def test_get_all_strategies(execution_service, sample_strategy):
    """Test getting all active strategies."""
    # Initially empty
    strategies = await execution_service.get_all_strategies()
    assert len(strategies) == 0
    
    # Deploy a strategy
    await execution_service.deploy_strategy(sample_strategy)
    
    # Should now have one strategy
    strategies = await execution_service.get_all_strategies()
    assert len(strategies) == 1
    assert strategies[0]["strategy_id"] == sample_strategy.id


@pytest.mark.asyncio
async def test_evaluate_strategy_rules(execution_service, sample_strategy):
    """Test evaluating strategy rules."""
    # Mock market data
    market_data = {
        "ETH-USDT": {
            "close": 1000,
            "volume": 100,
            "rsi": 25  # Below oversold threshold
        }
    }
    
    signals = await execution_service._evaluate_strategy_rules(sample_strategy, market_data)
    
    # Should generate a buy signal since RSI < 30
    assert "ETH-USDT" in signals
    assert signals["ETH-USDT"] == 1  # Buy signal


@pytest.mark.asyncio
async def test_run_strategy_simulation(execution_service, sample_strategy):
    """Test running strategy with simulated data."""
    # Mock dependencies - using simulated market data instead of external APIs
    with patch('app.services.execution_service.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        # Mock _evaluate_strategy_rules
        execution_service._evaluate_strategy_rules = AsyncMock()
        execution_service._evaluate_strategy_rules.return_value = {"ETH-USDT": 1}  # Buy signal
        
        # Mock _execute_signal
        execution_service._execute_signal = AsyncMock()
        
        # First deploy
        await execution_service.deploy_strategy(sample_strategy)
        
        # Run the strategy (this would normally be called by create_task)
        # We'll run it directly for testing
        task = asyncio.create_task(execution_service._run_strategy(sample_strategy.id))
        
        # Wait a bit for the task to run
        await asyncio.sleep(0.1)
        
        # Cancel the task to stop the loop
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Check that methods were called
        assert execution_service._evaluate_strategy_rules.called
        assert execution_service._execute_signal.called
        
        # Check strategy status
        assert execution_service.active_strategies[sample_strategy.id]["status"] in ["running", "stopped"]


@pytest.mark.asyncio
async def test_execute_signal_simulation(execution_service, sample_strategy):
    """Test executing signals with simulation."""
    # Test with simulated order execution (no external exchange needed)
    await execution_service.deploy_strategy(sample_strategy)
    
    # Create a buy signal
    signal = {
        "action": "BUY",
        "trading_pair": "ETH-USDT",
        "amount": 0.1,
        "price": 1000.0,
        "order_type": "market"
    }
    
    # Execute the signal (this will use simulated execution)
    result = await execution_service._execute_signal(signal, sample_strategy.id)
    
    # Check that the trade was recorded
    trades = execution_service.active_strategies[sample_strategy.id]["trades"]
    assert len(trades) > 0
    assert trades[-1]["type"] == "buy"
    assert trades[-1]["trading_pair"] == "ETH-USDT"


@pytest.mark.asyncio
async def test_calculate_performance_metrics(execution_service, sample_strategy):
    """Test calculating performance metrics."""
    # Deploy strategy and add mock trades
    await execution_service.deploy_strategy(sample_strategy)
    
    trades = [
        {"type": "buy", "price": 1000, "amount": 0.1, "timestamp": time.time() - 7200},
        {"type": "sell", "price": 1100, "amount": 0.1, "timestamp": time.time() - 3600},
        {"type": "buy", "price": 1050, "amount": 0.1, "timestamp": time.time() - 1800},
        {"type": "sell", "price": 1150, "amount": 0.1, "timestamp": time.time()}
    ]
    
    execution_service.active_strategies[sample_strategy.id]["trades"] = trades
    
    metrics = execution_service._calculate_performance_metrics(trades)
    
    assert "total_return" in metrics
    assert "win_rate" in metrics
    assert "total_trades" in metrics
    assert metrics["total_trades"] == 4
    assert metrics["total_return"] > 0  # Should be profitable


@pytest.mark.asyncio
async def test_risk_management(execution_service, sample_strategy):
    """Test risk management features."""
    await execution_service.deploy_strategy(sample_strategy)
    
    # Test position sizing
    position_size = execution_service._calculate_position_size(
        sample_strategy, "ETH-USDT", 1000.0
    )
    
    assert position_size > 0
    assert position_size <= sample_strategy.parameters.get("position_size", 0.1)


@pytest.mark.asyncio
async def test_error_handling(execution_service, sample_strategy):
    """Test error handling in execution service."""
    # Test with invalid strategy ID
    result = await execution_service.stop_strategy("invalid-id")
    assert result["status"] == "error"
    
    # Test getting performance for non-existent strategy
    with pytest.raises(Exception):
        await execution_service.get_strategy_performance("invalid-id")


if __name__ == "__main__":
    pytest.main([__file__])
