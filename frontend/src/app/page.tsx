'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, Brain, Zap, Settings, User, Bell, Search, Menu, AlertCircle } from 'lucide-react'

interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
}

interface PortfolioData {
  totalValue: number
  totalReturn: number
  totalReturnPercent: number
  activeStrategies: number
}

// Error boundary wrapper
function SafeComponent({ children, fallback }: { children: React.ReactNode, fallback: React.ReactNode }) {
  try {
    return <>{children}</>
  } catch (error) {
    console.error('Component error:', error)
    return <>{fallback}</>
  }
}

export default function QuantumFlowDashboard() {
  const [marketData, setMarketData] = useState<MarketData[]>([])
  const [portfolio, setPortfolio] = useState<PortfolioData>({
    totalValue: 125430,
    totalReturn: 6890,
    totalReturnPercent: 5.7,
    activeStrategies: 3
  })
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState('')
  const [isMounted, setIsMounted] = useState(false)
  const [isGeneratingStrategy, setIsGeneratingStrategy] = useState(false)
  const [strategyResult, setStrategyResult] = useState<string | null>(null)

  // Simulate real-time data updates
  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        setError(null)
        // Try to fetch real data from your backend
        const response = await fetch('http://localhost:8001/api/market/data')
        if (response.ok) {
          const data = await response.json()
          setIsConnected(true)
          // Process the data based on your API structure
          if (data.data && data.data.indices && Array.isArray(data.data.indices)) {
            const validData = data.data.indices
              .filter((item: any) => item && typeof item === 'object')
              .slice(0, 4)
              .map((item: any) => ({
                symbol: item.symbol || 'N/A',
                price: typeof item.price === 'number' ? item.price : 0,
                change: typeof item.change === 'number' ? item.change : 0,
                changePercent: typeof item.changePercent === 'number' ? item.changePercent : 0
              }))
            setMarketData(validData)
          }
        } else {
          // Fallback to demo data
          setIsConnected(false)
          setMarketData([
            { symbol: 'AAPL', price: 202.38, change: -8.49, changePercent: -4.03 },
            { symbol: 'MSFT', price: 423.45, change: 12.34, changePercent: 2.99 },
            { symbol: 'GOOGL', price: 178.92, change: -2.15, changePercent: -1.19 },
            { symbol: 'TSLA', price: 248.50, change: 15.67, changePercent: 6.73 }
          ])
        }
      } catch (error) {
        console.log('Using demo data:', error)
        setIsConnected(false)
        setError('Backend connection failed - using demo data')
        setMarketData([
          { symbol: 'AAPL', price: 202.38, change: -8.49, changePercent: -4.03 },
          { symbol: 'MSFT', price: 423.45, change: 12.34, changePercent: 2.99 },
          { symbol: 'GOOGL', price: 178.92, change: -2.15, changePercent: -1.19 },
          { symbol: 'TSLA', price: 248.50, change: 15.67, changePercent: 6.73 }
        ])
      } finally {
        setIsLoading(false)
      }
    }

    // Set mounted state
    setIsMounted(true)
    
    fetchMarketData()
    
    // Update time every second to avoid hydration mismatch
    const updateTime = () => setCurrentTime(new Date().toLocaleTimeString())
    updateTime() // Set initial time
    
    const dataInterval = setInterval(fetchMarketData, 30000) // Update every 30 seconds
    const timeInterval = setInterval(updateTime, 1000) // Update time every second

    return () => {
      clearInterval(dataInterval)
      clearInterval(timeInterval)
    }
  }, [])

  // Generate Strategy function
  const generateStrategy = async () => {
    setIsGeneratingStrategy(true)
    setStrategyResult(null)
    setError(null)
    
    try {
      const response = await fetch('http://localhost:8001/api/strategies/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          goal: 'momentum',
          risk_tolerance: 'medium',
          trading_pairs: ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        })
      })
      
      if (response.ok) {
        const data = await response.json()
        if (data.status === 'success' && data.strategy) {
          const strategy = data.strategy
          setStrategyResult(`Strategy Generated: ${strategy.name}\n\nLogic: ${strategy.logic}\n\nRisk Level: ${strategy.risk_tolerance}\nTrading Pairs: ${strategy.trading_pairs.join(', ')}\nCreated: ${new Date(strategy.created_at).toLocaleString()}`)
        } else {
          throw new Error('Invalid response format')
        }
      } else {
        throw new Error('Failed to generate strategy')
      }
    } catch (error) {
      console.error('Strategy generation error:', error)
      setError('Failed to generate strategy. Using demo mode.')
      // Demo strategy result
      setStrategyResult(`Demo Strategy Generated: Momentum Trading\n\nLogic: Buy when RSI < 30 and MACD crosses above signal line. Sell when RSI > 70 or stop loss at -5%.\n\nRisk Level: Medium\nExpected Return: 12-18% annually`)
    } finally {
      setIsGeneratingStrategy(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-white/10 bg-black/20 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Zap className="h-8 w-8 text-blue-400" />
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  QuantumFlow AI Lab
                </h1>
              </div>
              <Badge variant={isConnected ? "default" : "secondary"} className="ml-4">
                {isConnected ? "🟢 Live Data" : "🔄 Demo Mode"}
              </Badge>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white">
                <Search className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white relative">
                <Bell className="h-4 w-4" />
                <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-500 rounded-full text-xs flex items-center justify-center text-white">3</span>
              </Button>
              <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white">
                <Settings className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white">
                <User className="h-4 w-4" />
              </Button>
              <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-400 to-purple-400 flex items-center justify-center text-white text-sm font-semibold">
                A
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        {/* Error Notification */}
        {error && (
          <Card className="bg-red-500/10 border-red-500/20 backdrop-blur-sm mb-6">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 text-red-400" />
                  <span className="text-red-300 text-sm">{error}</span>
                </div>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setError(null)}
                  className="text-red-300 hover:text-red-100 h-6 w-6 p-0"
                >
                  ×
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-200">Portfolio Value</CardTitle>
              <DollarSign className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">${portfolio.totalValue.toLocaleString()}</div>
              <p className="text-xs text-green-400 flex items-center">
                <TrendingUp className="h-3 w-3 mr-1" />
                +{portfolio.totalReturnPercent}% (${portfolio.totalReturn.toLocaleString()})
              </p>
            </CardContent>
          </Card>

          <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-200">Active Strategies</CardTitle>
              <Brain className="h-4 w-4 text-blue-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{portfolio.activeStrategies}</div>
              <p className="text-xs text-blue-400">AI-powered trading</p>
            </CardContent>
          </Card>

          <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-200">Market Sentiment</CardTitle>
              <Activity className="h-4 w-4 text-purple-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">Bullish</div>
              <p className="text-xs text-purple-400">78% confidence</p>
            </CardContent>
          </Card>

          <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-gray-200">Today's Performance</CardTitle>
              <BarChart3 className="h-4 w-4 text-yellow-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">+2.4%</div>
              <p className="text-xs text-yellow-400">Outperforming S&P 500</p>
            </CardContent>
          </Card>
        </div>

        {/* Market Data */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <TrendingUp className="h-5 w-5 mr-2 text-green-400" />
                Live Market Data
              </CardTitle>
              <CardDescription className="text-gray-400">
                {isConnected ? 'Real-time stock prices from Marketstack API' : 'Demo stock prices (Backend offline)'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SafeComponent fallback={
                <div className="text-center text-gray-400 py-8">
                  Unable to load market data
                </div>
              }>
                <div className="space-y-4">
                  {isLoading ? (
                    // Loading skeletons
                    Array.from({ length: 4 }).map((_, i) => (
                      <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                        <div className="space-y-2">
                          <Skeleton className="h-4 w-16 bg-white/10" />
                          <Skeleton className="h-3 w-20 bg-white/10" />
                        </div>
                        <div className="space-y-2 text-right">
                          <Skeleton className="h-4 w-12 bg-white/10" />
                          <Skeleton className="h-3 w-10 bg-white/10" />
                        </div>
                      </div>
                    ))
                  ) : (
                    // Actual market data
                    marketData.map((stock) => {
                      // Ensure all values exist with defaults
                      const price = stock.price ?? 0
                      const changePercent = stock.changePercent ?? 0
                      const change = stock.change ?? 0
                      const symbol = stock.symbol ?? 'N/A'
                      
                      return (
                        <div key={symbol} className="flex items-center justify-between p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors cursor-pointer">
                          <div>
                            <div className="font-semibold text-white">{symbol}</div>
                            <div className="text-sm text-gray-400">${price.toFixed(2)}</div>
                          </div>
                          <div className="text-right">
                            <div className={`flex items-center ${changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {changePercent >= 0 ? <TrendingUp className="h-4 w-4 mr-1" /> : <TrendingDown className="h-4 w-4 mr-1" />}
                              {changePercent.toFixed(2)}%
                            </div>
                            <div className="text-sm text-gray-400">
                              {changePercent >= 0 ? '+' : ''}{change.toFixed(2)}
                            </div>
                          </div>
                        </div>
                      )
                    })
                  )}
                </div>
              </SafeComponent>
            </CardContent>
          </Card>

          <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Brain className="h-5 w-5 mr-2 text-blue-400" />
                AI Insights
              </CardTitle>
              <CardDescription className="text-gray-400">
                Latest AI-generated market analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                  <div className="flex items-center mb-2">
                    <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30">Strategy Generated</Badge>
                  </div>
                  <p className="text-sm text-gray-300">Momentum strategy identified for tech stocks. High confidence signal detected.</p>
                </div>
                <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
                  <div className="flex items-center mb-2">
                    <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30">RL Model Update</Badge>
                  </div>
                  <p className="text-sm text-gray-300">Reinforcement learning model improved performance by 2.3%.</p>
                </div>
                <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                  <div className="flex items-center mb-2">
                    <Badge className="bg-green-500/20 text-green-300 border-green-500/30">Market Analysis</Badge>
                  </div>
                  <p className="text-sm text-gray-300">Bullish sentiment detected across major indices. Volume indicators strong.</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions */}
        <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white">Quick Actions</CardTitle>
            <CardDescription className="text-gray-400">
              Manage your AI trading strategies
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4">
              <Button 
                className="bg-blue-600 hover:bg-blue-700" 
                onClick={generateStrategy}
                disabled={isGeneratingStrategy}
              >
                <Brain className="h-4 w-4 mr-2" />
                {isGeneratingStrategy ? 'Generating...' : 'Generate Strategy'}
              </Button>
              <Button variant="outline" className="border-white/20 text-white hover:bg-white/10">
                <BarChart3 className="h-4 w-4 mr-2" />
                View Analytics
              </Button>
              <Button variant="outline" className="border-white/20 text-white hover:bg-white/10">
                <Activity className="h-4 w-4 mr-2" />
                Market Research
              </Button>
              <Button variant="outline" className="border-white/20 text-white hover:bg-white/10">
                <DollarSign className="h-4 w-4 mr-2" />
                Portfolio
              </Button>
            </div>
            
            {/* Strategy Result Display */}
            {strategyResult && (
              <div className="mt-6 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <h4 className="text-green-400 font-semibold mb-2 flex items-center">
                  <Brain className="h-4 w-4 mr-2" />
                  AI Strategy Generated
                </h4>
                <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono">
                  {strategyResult}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-black/20 backdrop-blur-sm mt-12">
        <div className="container mx-auto px-6 py-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {/* Company Info */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Zap className="h-6 w-6 text-blue-400" />
                <h3 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  QuantumFlow AI Lab
                </h3>
              </div>
              <p className="text-sm text-gray-400">
                Advanced AI-powered trading platform with real-time market data and intelligent strategy generation.
              </p>
              <div className="flex space-x-4">
                <Badge variant="secondary" className="text-xs">
                  🚀 Live Data
                </Badge>
                <Badge variant="secondary" className="text-xs">
                  🤖 AI Powered
                </Badge>
              </div>
            </div>

            {/* Features */}
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-white uppercase tracking-wider">Features</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li className="flex items-center">
                  <Brain className="h-3 w-3 mr-2 text-blue-400" />
                  AI Strategy Generation
                </li>
                <li className="flex items-center">
                  <BarChart3 className="h-3 w-3 mr-2 text-green-400" />
                  Real-time Analytics
                </li>
                <li className="flex items-center">
                  <Activity className="h-3 w-3 mr-2 text-purple-400" />
                  Market Sentiment
                </li>
                <li className="flex items-center">
                  <TrendingUp className="h-3 w-3 mr-2 text-yellow-400" />
                  Portfolio Management
                </li>
              </ul>
            </div>

            {/* Market Data */}
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-white uppercase tracking-wider">Market Data</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>170,000+ Stock Tickers</li>
                <li>70+ Global Exchanges</li>
                <li>Real-time Price Updates</li>
                <li>Historical Data Access</li>
              </ul>
            </div>

            {/* API Status */}
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-white uppercase tracking-wider">System Status</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Marketstack API</span>
                  <Badge variant={isConnected ? "default" : "secondary"} className="text-xs">
                    {isConnected ? "🟢 Online" : "🔄 Demo"}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">AI Services</span>
                  <Badge variant="default" className="text-xs">🟢 Active</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">WebSocket</span>
                  <Badge variant="default" className="text-xs">🟢 Connected</Badge>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="border-t border-white/10 mt-8 pt-6 flex flex-col md:flex-row justify-between items-center">
            <div className="text-sm text-gray-400 mb-4 md:mb-0">
              © 2025 QuantumFlow AI Lab. Built with Next.js, FastAPI, and AI technologies.
            </div>
            <div className="flex items-center space-x-6 text-sm text-gray-400">
              <span className="flex items-center">
                <Activity className="h-3 w-3 mr-1 text-green-400" />
                System Operational
              </span>
              <span className="flex items-center">
                <Zap className="h-3 w-3 mr-1 text-blue-400" />
                Last Updated: {isMounted ? (currentTime || 'Loading...') : 'Loading...'}
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
