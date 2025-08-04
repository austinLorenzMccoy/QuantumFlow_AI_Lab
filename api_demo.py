#!/usr/bin/env python3
"""
QuantumFlow AI Lab - API Demo

This demo showcases the core capabilities of the QuantumFlow AI Lab platform:
- Strategy generation using LLMs and RAG
- Reinforcement learning optimization
- Multi-agent collaboration
- Real-time WebSocket communication
- Sentiment analysis
- Transaction simulation

Run this demo to see the platform in action without requiring external exchange APIs.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.core.marketstack_client import get_stock_data, get_market_overview

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Demo FastAPI app
app = FastAPI(
    title="QuantumFlow AI Lab - API Demo",
    description="Interactive demo of the QuantumFlow AI Lab platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Start background tasks when the app starts."""
    asyncio.create_task(simulate_background_tasks())

# Demo data storage
demo_strategies = {}
demo_market_data = {
    "BTC-USDT": {"price": 50000.0, "volume": 1000000, "change_24h": 2.5},
    "ETH-USDT": {"price": 3000.0, "volume": 800000, "change_24h": 1.8},
    "SOL-USDT": {"price": 100.0, "volume": 500000, "change_24h": -0.5}
}

# WebSocket connections
active_connections: List[WebSocket] = []

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        message_str = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")

manager = ConnectionManager()

@app.get("/")
async def get_demo_page():
    """Serve the demo dashboard."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>QuantumFlow AI Lab - Demo Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .card h3 {
            margin-top: 0;
            color: #ffd700;
        }
        button {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .status {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            font-family: monospace;
            max-height: 300px;
            overflow-y: auto;
        }
        .market-data {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        .price {
            font-size: 1.2em;
            font-weight: bold;
        }
        .positive { color: #2ecc71; }
        .negative { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 QuantumFlow AI Lab - Demo Dashboard</h1>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Market Data</h3>
                <div id="market-data">
                    <div class="market-data">
                        <span>BTC-USDT:</span>
                        <span class="price positive">$50,000.00 (+2.5%)</span>
                    </div>
                    <div class="market-data">
                        <span>ETH-USDT:</span>
                        <span class="price positive">$3,000.00 (+1.8%)</span>
                    </div>
                    <div class="market-data">
                        <span>SOL-USDT:</span>
                        <span class="price negative">$100.00 (-0.5%)</span>
                    </div>
                </div>
                <button onclick="refreshMarketData()">Refresh Data</button>
            </div>
            
            <div class="card">
                <h3>🤖 AI Strategy Generation</h3>
                <p>Generate trading strategies using LLMs and RAG</p>
                <button onclick="generateStrategy()">Generate Strategy</button>
                <button onclick="optimizeStrategy()">Optimize with RL</button>
                <div id="strategy-status" class="status"></div>
            </div>
            
            <div class="card">
                <h3>👥 Multi-Agent System</h3>
                <p>Collaborative AI agents for analysis and execution</p>
                <button onclick="startAgents()">Start Agents</button>
                <button onclick="stopAgents()">Stop Agents</button>
                <div id="agent-status" class="status"></div>
            </div>
            
            <div class="card">
                <h3>💱 Transaction Simulation</h3>
                <p>Test strategies with realistic market simulation</p>
                <button onclick="simulateTransaction()">Simulate Trade</button>
                <button onclick="runBacktest()">Run Backtest</button>
                <div id="simulation-status" class="status"></div>
            </div>
            
            <div class="card">
                <h3>📈 Sentiment Analysis</h3>
                <p>Real-time market sentiment from news and social media</p>
                <button onclick="analyzeSentiment()">Analyze Sentiment</button>
                <div id="sentiment-status" class="status"></div>
            </div>
            
            <div class="card">
                <h3>🔄 Real-time Updates</h3>
                <p>WebSocket connection status and live data</p>
                <button onclick="connectWebSocket()">Connect</button>
                <button onclick="disconnectWebSocket()">Disconnect</button>
                <div id="websocket-status" class="status">Disconnected</div>
            </div>
        </div>
        
        <div class="card">
            <h3>📝 System Logs</h3>
            <div id="system-logs" class="status" style="height: 200px;">
                Welcome to QuantumFlow AI Lab Demo! Click buttons above to explore features.
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        
        function log(message) {
            const logs = document.getElementById('system-logs');
            const timestamp = new Date().toLocaleTimeString();
            logs.innerHTML += `\\n[${timestamp}] ${message}`;
            logs.scrollTop = logs.scrollHeight;
        }
        
        function connectWebSocket() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                log('WebSocket already connected');
                return;
            }
            
            ws = new WebSocket(`ws://localhost:8001/ws/demo`);
            
            ws.onopen = function(event) {
                log('WebSocket connected successfully');
                document.getElementById('websocket-status').innerHTML = 'Connected ✅';
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                log(`Received: ${data.type} - ${data.message || JSON.stringify(data)}`);
                
                // Update relevant sections based on message type
                if (data.type === 'market_data') {
                    updateMarketData(data);
                } else if (data.type === 'strategy_update') {
                    updateStrategyStatus(data);
                } else if (data.type === 'agent_update') {
                    updateAgentStatus(data);
                }
            };
            
            ws.onclose = function(event) {
                log('WebSocket connection closed');
                document.getElementById('websocket-status').innerHTML = 'Disconnected ❌';
            };
            
            ws.onerror = function(error) {
                log(`WebSocket error: ${error}`);
            };
        }
        
        function disconnectWebSocket() {
            if (ws) {
                ws.close();
                ws = null;
            }
        }
        
        async function apiCall(endpoint, method = 'GET', data = null) {
            try {
                const options = {
                    method: method,
                    headers: {
                        'Content-Type': 'application/json',
                    }
                };
                
                if (data) {
                    options.body = JSON.stringify(data);
                }
                
                const response = await fetch(`http://localhost:8001${endpoint}`, options);
                const result = await response.json();
                return result;
            } catch (error) {
                log(`API Error: ${error.message}`);
                return { error: error.message };
            }
        }
        
        async function refreshMarketData() {
            log('Refreshing market data...');
            const data = await apiCall('/api/market/data');
            if (data.error) {
                log(`Error: ${data.error}`);
            } else {
                log('Market data refreshed successfully');
                updateMarketDataDisplay(data);
            }
        }
        
        async function generateStrategy() {
            log('Generating AI trading strategy...');
            document.getElementById('strategy-status').innerHTML = 'Generating strategy...';
            
            const data = await apiCall('/api/strategies/generate', 'POST', {
                goal: 'momentum',
                risk_tolerance: 'medium',
                trading_pairs: ['BTC-USDT', 'ETH-USDT']
            });
            
            if (data.error) {
                document.getElementById('strategy-status').innerHTML = `Error: ${data.error}`;
            } else {
                document.getElementById('strategy-status').innerHTML = `Strategy generated: ${data.name}\\nLogic: ${data.logic}`;
                log(`Strategy generated: ${data.name}`);
            }
        }
        
        async function optimizeStrategy() {
            log('Optimizing strategy with reinforcement learning...');
            document.getElementById('strategy-status').innerHTML = 'Optimizing with RL...';
            
            setTimeout(() => {
                document.getElementById('strategy-status').innerHTML = 'RL Optimization complete!\\nImproved Sharpe ratio: 1.85\\nMax drawdown reduced: 12%';
                log('RL optimization completed successfully');
            }, 3000);
        }
        
        async function startAgents() {
            log('Starting multi-agent system...');
            document.getElementById('agent-status').innerHTML = 'Starting agents...';
            
            const agents = ['Data Analyst', 'Strategy Optimizer', 'Risk Manager', 'Execution Agent'];
            let agentStatus = '';
            
            for (let i = 0; i < agents.length; i++) {
                setTimeout(() => {
                    agentStatus += `${agents[i]}: Active\\n`;
                    document.getElementById('agent-status').innerHTML = agentStatus;
                    log(`${agents[i]} started`);
                }, i * 1000);
            }
        }
        
        async function stopAgents() {
            log('Stopping all agents...');
            document.getElementById('agent-status').innerHTML = 'All agents stopped';
        }
        
        async function simulateTransaction() {
            log('Running transaction simulation...');
            document.getElementById('simulation-status').innerHTML = 'Simulating trade...';
            
            setTimeout(() => {
                const result = `Trade Simulation Results:
Amount: 0.1 BTC
Entry Price: $50,000
Exit Price: $51,200
Profit: $120.00
Slippage: 0.02%
Fees: $5.00`;
                document.getElementById('simulation-status').innerHTML = result;
                log('Transaction simulation completed');
            }, 2000);
        }
        
        async function runBacktest() {
            log('Running strategy backtest...');
            document.getElementById('simulation-status').innerHTML = 'Running backtest...';
            
            setTimeout(() => {
                const result = `Backtest Results (30 days):
Total Return: +15.7%
Sharpe Ratio: 1.42
Max Drawdown: -8.3%
Win Rate: 68%
Total Trades: 47`;
                document.getElementById('simulation-status').innerHTML = result;
                log('Backtest completed successfully');
            }, 4000);
        }
        
        async function analyzeSentiment() {
            log('Analyzing market sentiment...');
            document.getElementById('sentiment-status').innerHTML = 'Analyzing sentiment...';
            
            setTimeout(() => {
                const result = `Sentiment Analysis:
Overall Market: Bullish (72%)
BTC Sentiment: Very Bullish (85%)
ETH Sentiment: Bullish (68%)
Fear & Greed Index: 76 (Greed)
Social Media Buzz: High`;
                document.getElementById('sentiment-status').innerHTML = result;
                log('Sentiment analysis completed');
            }, 2500);
        }
        
        function updateMarketData(data) {
            // Update market data display with real-time data
        }
        
        function updateStrategyStatus(data) {
            document.getElementById('strategy-status').innerHTML += `\\n${data.message}`;
        }
        
        function updateAgentStatus(data) {
            document.getElementById('agent-status').innerHTML += `\\n${data.message}`;
        }
        
        function updateMarketDataDisplay(data) {
            // Update the market data section with new data
        }
        
        // Auto-connect WebSocket on page load
        window.onload = function() {
            log('QuantumFlow AI Lab Demo loaded successfully');
            setTimeout(connectWebSocket, 1000);
        };
        
        // Clean up WebSocket on page unload
        window.onbeforeunload = function() {
            if (ws) {
                ws.close();
            }
        };
    </script>
</body>
</html>
    """)

@app.websocket("/ws/demo")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time demo updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)
            
            # Simulate market data updates
            for pair in demo_market_data:
                demo_market_data[pair]["price"] *= (1 + (hash(str(time.time())) % 100 - 50) / 10000)
                demo_market_data[pair]["change_24h"] = (hash(str(time.time())) % 200 - 100) / 10
            
            await manager.broadcast({
                "type": "market_data",
                "data": demo_market_data,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/market/data")
async def get_market_data():
    """Get current market data from Marketstack API."""
    try:
        # Try to get real market data
        market_overview = await get_market_overview()
        if market_overview['indices']:
            return {
                "status": "success",
                "data": market_overview,
                "source": "marketstack_api",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.warning(f"Failed to get real market data: {e}")
    
    # Fallback to demo data
    return {
        "status": "success",
        "data": demo_market_data,
        "source": "simulation",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stocks/{symbol}")
async def get_stock_data_endpoint(symbol: str):
    """Get real stock data for a specific symbol."""
    try:
        stock_data = await get_stock_data([symbol.upper()], days_back=1)
        if stock_data:
            return {
                "status": "success",
                "symbol": symbol.upper(),
                "data": stock_data[0],
                "source": "marketstack_api",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get stock data for {symbol}: {e}")
    
    return {
        "status": "error",
        "message": f"Could not fetch data for {symbol}",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/strategies/generate")
async def generate_strategy(request: Dict[str, Any]):
    """Generate a new trading strategy using AI."""
    strategy_id = f"strategy_{len(demo_strategies) + 1}"
    
    # Simulate strategy generation
    strategy = {
        "id": strategy_id,
        "name": f"AI {request.get('goal', 'momentum').title()} Strategy",
        "logic": f"Buy when RSI < 30 and volume > avg_volume * 1.5, Sell when RSI > 70",
        "risk_tolerance": request.get("risk_tolerance", "medium"),
        "trading_pairs": request.get("trading_pairs", ["BTC-USDT"]),
        "created_at": datetime.now().isoformat(),
        "status": "generated"
    }
    
    demo_strategies[strategy_id] = strategy
    
    # Broadcast update
    await manager.broadcast({
        "type": "strategy_update",
        "message": f"New strategy generated: {strategy['name']}",
        "strategy": strategy
    })
    
    return {
        "status": "success",
        "strategy": strategy
    }

@app.get("/api/strategies")
async def get_strategies():
    """Get all generated strategies."""
    return {
        "status": "success",
        "strategies": list(demo_strategies.values())
    }

@app.get("/api/demo/info")
async def get_demo_info():
    """Get demo platform information."""
    return {
        "name": "QuantumFlow AI Lab",
        "version": "1.0.0",
        "description": "AI-driven crypto trading strategy platform",
        "features": [
            "AI Strategy Generation",
            "Reinforcement Learning Optimization",
            "Multi-Agent Collaboration",
            "Real-time WebSocket Communication",
            "Sentiment Analysis",
            "Transaction Simulation",
            "Risk Management"
        ],
        "status": "running",
        "active_connections": len(manager.active_connections),
        "demo_strategies": len(demo_strategies)
    }

async def simulate_background_tasks():
    """Simulate background AI processes."""
    while True:
        await asyncio.sleep(10)
        
        # Simulate agent activities
        agents = ["Data Analyst", "Strategy Optimizer", "Risk Manager", "Execution Agent"]
        agent = agents[int(time.time()) % len(agents)]
        
        await manager.broadcast({
            "type": "agent_update",
            "message": f"{agent}: Processing market data and updating models",
            "timestamp": datetime.now().isoformat()
        })

if __name__ == "__main__":
    logger.info("Starting QuantumFlow AI Lab Demo Server...")
    logger.info("Features included:")
    logger.info("  🤖 AI Strategy Generation")
    logger.info("  🧠 Reinforcement Learning")
    logger.info("  👥 Multi-Agent Systems")
    logger.info("  📡 Real-time WebSockets")
    logger.info("  📊 Market Simulation")
    logger.info("  💱 Transaction Testing")
    logger.info("")
    logger.info("Demo will be available at: http://localhost:8001")
    logger.info("WebSocket endpoint: ws://localhost:8001/ws/demo")
    logger.info("")
    logger.info("This demo runs independently without requiring external APIs")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001)
