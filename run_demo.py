#!/usr/bin/env python3
"""
QuantumFlow AI Lab - Demo Runner

This script provides an easy way to run the QuantumFlow AI Lab demo.
It starts the demo server and opens the dashboard in your browser.
"""

import subprocess
import sys
import time
import webbrowser
import threading
from pathlib import Path

def print_banner():
    """Print the QuantumFlow AI Lab banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                🚀 QuantumFlow AI Lab Demo 🚀                 ║
    ║                                                              ║
    ║              Advanced AI-Driven Trading Platform             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Features:
    • 🤖 AI Strategy Generation using LLMs and RAG
    • 🧠 Reinforcement Learning Optimization  
    • 👥 Multi-Agent Collaborative Systems
    • 📡 Real-time WebSocket Communication
    • 📊 Advanced Market Simulation
    • 💱 Transaction Testing & Backtesting
    • 📈 Sentiment Analysis Integration
    • 🛡️ Risk Management Systems
    
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import websockets
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def start_demo_server():
    """Start the demo server."""
    print("🚀 Starting QuantumFlow AI Lab Demo Server...")
    print("📍 Server will be available at: http://localhost:8001")
    print("🔌 WebSocket endpoint: ws://localhost:8001/ws/demo")
    print()
    
    # Run the demo server
    try:
        subprocess.run([sys.executable, "api_demo.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting demo server: {e}")

def open_browser():
    """Open the demo dashboard in the browser."""
    time.sleep(3)  # Wait for server to start
    print("🌐 Opening demo dashboard in browser...")
    webbrowser.open("http://localhost:8001")

def main():
    """Main function to run the demo."""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("api_demo.py").exists():
        print("❌ Error: api_demo.py not found in current directory")
        print("Please run this script from the QuantumFlow AI Lab root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("🔧 Preparing demo environment...")
    
    # Start browser opener in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the demo server
    start_demo_server()

if __name__ == "__main__":
    main()
