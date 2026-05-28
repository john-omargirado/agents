# Forex Multi-Agent System (MAS)

An educational forex trading recommendation platform powered by multi-agent architecture, designed to help retail and beginner traders grasp forex market dynamics through a gamified learning environment. Combines machine learning inference, technical analysis, and intelligent decision-making to provide trading recommendations and insights.

**Note**: This is an educational and gamified learning platform designed to help traders understand market dynamics, technical analysis, and trading strategies. It provides recommendations based on historical analysis and is not intended as financial advice or for actual trading without proper risk management and trader expertise.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Components](#components)
- [Data Management](#data-management)
- [Development Guide](#development-guide)
- [Contributing](#contributing)

---

## Overview

The Forex Multi-Agent System is an intelligent learning platform that:

- **Analyzes** forex market data and news feeds using multiple specialized agents to generate trading recommendations
- **Educates** retail and beginner traders through explanations of market analysis and trading signals
- **Demonstrates** market dynamics through machine learning models (FinBERT sentiment analysis, technical indicators)
- **Provides trading recommendations** using a consensus-based agent framework
- **Backtests** strategies against historical data to help traders understand performance patterns
- **Offers a gamified dashboard** for practicing trading in a safe, educational environment with performance tracking

### Key Features

- **Multi-Agent Architecture**: Specialized agents for different trading signals (Sentiment, SIV, TTS, Verdict)
- **Market Data Processing**: OHLCV data and news analysis for educational purposes
- **AI-Powered Insights**: FinBERT sentiment analysis, technical indicators, and educational signal generation
- **Comprehensive Backtesting**: Historical analysis to help traders understand strategy performance
- **Gamified Dashboard**: React-based frontend with charting, metrics, and progress tracking
- **Educational Explanations**: AI-generated explanations of market movements and trading rationales
- **Secure Backend**: Flask API with rate limiting and CORS protection

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend Dashboard                 │
│        (Trading Parameters, Charts, Metrics, Backtesting)   │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────▼──────────────────────────────────────┐
│                  Flask Backend API                          │
│         (Rate Limiting, CORS, Request Handling)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
    │  Graph  │  │ Agents  │  │ Utilities│
    │ Engine  │  │ Network │  │          │
    └────┬────┘  └────┬────┘  └────┬────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
    │   LLM   │  │   Data  │  │ Memory  │
    │ Ollama  │  │ Loader  │  │ State   │
    └─────────┘  └─────────┘  └─────────┘
```

### Agent Networkmarket context explanation
- **Chat Agent**: Conversational learning interface and trade explanations
- **SIV Agent**: Strategic Investment Validation for position sizing education
- **TTS Agent**: Technical Trading Signals generation and education
- **Verdict Agent**: Consensus recommendation aggregatotion
- **TTS Agent**: Technical Trading Signals
- **Verdict Agent**: Consensus decision maker

---

## Project Structure

```
Overhaul/
├── app.py                          # Flask backend entry point
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
│
├── agents/                          # Multi-agent implementations
│   ├── ce_agent.py                 # Contextual Evaluation Agent
│   ├── chat_agent.py               # Chat/Explanation Agent
│   ├── siv_agent.py                # Strategic Investment Validation
│   ├── tts_agent.py                # Technical Trading Signals
│   └── verdict_agent.py            # Consensus Verdict Agent
│
├── calibration/                     # Strategy calibration & educational backtesting
│   ├── explanation_pipeline.py     # Explanation generation pipeline
│   ├── run_backtesting.py          # Backtesting runner
│   └── run_calibration.py          # Parameter calibration
│
├── data/                            # Data storage for learning and backtesting
│   ├── backtesting/                # Historical backtesting data
│   │   ├── forex_pairs/            # OHLCV data for currency pairs
│   │   ├── news/                   # Historical news data
│   │   └── news_cleaned/           # Preprocessed news
│   ├── calibration/                # Calibration datasets
│   └── news_backup/                # Backup news data
│
├── frontend/                        # React frontend application
│   ├── index.html
│   ├── package.json                # Node dependencies
│   ├── vite.config.js              # Vite build configuration
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── components/             # React components
│   │   │   ├── Backtesting.jsx
│   │   │   ├── CandlestickChart.jsx
│   │   │   ├── TradingAssistant.jsx
│   │   │   ├── TradingParameters.jsx
│   │   │   └── ...
│   │   ├── services/               # API client
│   │   ├── styles/                 # CSS styling
│   │   └── utils/                  # Frontend utilities
│   └── public/                      # Static assets
│
├── graph/
│   └── build_graph.py              # LangsGraph workflow engine
│
├── llm/                             # LLM integration
│   ├── do_inference.py             # Inference execution
│   └── ollama_client.py            # Ollama API client
│
├── memory/
│   └── state_memory.py             # Trading state memory management
│
├── scripts/                         # Data fetching scripts
│   └── gdelt_news_fetcher.py       # News data collection
│
├── state/                           # State management
│   ├── contracts.py                # Data contracts/schemas
│   ├── trading_state.py            # Trading state definitions
│   └── finbert_cache.json          # FinBERT sentiment cache
│
├── tools/                           # Agent tools/utilities
│   ├── ce_tools.py                 # CE Agent tools
│   ├── siv_tools.py                # SIV Agent tools
│   ├── tts_tools.py                # TTS Agent tools
│   ├── verdict_tools.py            # Verdict Agent tools
│   └── build_news_parquet.py       # News data processing
│
└── utils/                           # General utilities
    ├── credentials.py              # API credentials management
    ├── data_loader.py              # Data loading utilities
    ├── formatters.py               # Output formatting
    ├── logger.py                   # Logging configuration
    └── trade_config.py             # Trading configuration
```

---

## Prerequisites

- **Python**: 3.10+
- **Node.js**: 16+ (for frontend)
- **Ollama**: Latest version (for local LLM inference)
- **Git**: For version control

### External Services (Optional)

- GDELT Project API (for news data)

---

## Installation & Setup

### 1. Clone the Repository

```bash
cd c:\Users\John Omar\Documents\Project_Apps\Overhaul
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate      # Linux/Mac
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Configure Ollama (Optional, for local LLM)

```bash
# Download and install Ollama from https://ollama.ai
# Pull a model (example: mistral)
ollama pull mistral
ollama serve  # Run Ollama server
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# API Configuration
API_PORT=5000
API_HOST=localhost

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=mistral

# Data Configuration
DATA_PATH=./data
BACKTESTING_DATA_PATH=./data/backtesting

# Frontend
VITE_API_URL=http://localhost:5000
```

### Trading Configuration

Edit `utils/trade_config.py` to customize learning parameters:

- Risk management parameters (stop loss, take profit percentages)
- Agent weightings for consensus recommendations
- Technical indicator settings
- News sentiment thresholds for signal generation

## Running the Application

### Option 1: Development Mode (Recommended for Learning)

#### Terminal 1 - Backend

```bash
# Activate virtual environment (if not already active)
.\.venv\Scripts\Activate.ps1

# Run Flask server
python app.py
```

Backend will be available at `http://localhost:5000`

#### Terminal 2 - Frontend

```bash
cd frontend
npm run dev
```

Frontend will be available at `http://localhost:5173` - Start exploring trading recommendations and backtest strategies!

#### Backend

```bash
# Set environment for deployment
$env:FLASK_ENV = 'production'
python app.py
```

#### Frontend

```bash
cd frontend
npm run build
npm run preview
```

### Option 3: Run Backtesting

Educational backtesting to analyze strategy performance:

```bash
python calibration/run_backtesting.py
```

---

## API Documentation

### Base URL

```
http://localhost:5000/api
```

### Key Endpoints
Recommendation

```http
POST /api/trading-recommendation
Content-Type: application/json

{
  "pair": "EURUSD",
  "timestamp": "2024-01-15T14:30:00Z",
  "parameters": {
    "risk_level": "medium",
    "lookback_period": 20
  }
}

Response:
{
  "recommendation": "BUY|SELL|HOLD",
  "confidence": 0.85,
  "agent_signals": {
    "ce_agent": "BUY",
    "siv_agent": "BUY",
    "tts_agent": "HOLD"
  },
  "explanation": "Based on market sentiment (positive news analysis) and technical indicators showing bullish pressure, we recommend considering a BUY position...",
  "educational_insights": {
    "sentiment_analysis": "...",
    "technical_rationale": "..."
  }
  "explanation": "..."
}
```

#### Get Historical Data

```http
GET /api/historical-data?pair=EURUSD&start=2024-01-01&end=2024-01-31

Response:
{
  "pair": "EURUSD",
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 1.1050,
      "high": 1.1080,
      "low": 1.1040,
      "close": 1.1075,
      "volume": 1000000
    }
  ],
  "educational_note": "This historical data is used to demonstrate market movements and help traders understand price action"
}
```

#### Get Backtesting Results

```http
GET /api/backtest-results?pair=EURUSD&strategy=consensus

Response:
{,
  "key_learnings": "This strategy demonstrates the importance of risk management with a max drawdown of 12%..."
  "total_trades": 45,
  "win_rate": 0.64,
  "profit_factor": 1.85,
  "sharpe_ratio": 1.42,
  "max_drawdown": -0.12
}
```

---

## Components Overview

### Backend Agents

| Agent | Purpose | Key Functions |market event explanation |
| **Chat Agent** | Educational Interface | Answering trader questions, explaining market dynamics |
| **SIV Agent** | Investment Education | Teaching portfolio concepts, position sizing principles |
| **TTS Agent** | Technical Signal Generation | Indicator analysis, signal generation for learning |
| **Verdict Agent** | Recommendation Aggregator | Combining signals, providing consensus recommendations with confidence scores
| **TTS Agent** | Technical Signals | Indicator analysis, entry/exit points |
| **Verdict Agent** | Consensus Maker | Decision aggregation, confidence scoring |

### Frontend Components

- **TradingAssistant**: Main learning and recommendation interface
- **TradingParameters**: Configuration panel for learning scenarios
- **Backtesting**: Historical analysis tools for understanding strategy performance
- **CandlestickChart**: Price action visualization for learning technical analysis
- **MetricsSection**: Performance metrics display for educational insights
- **ContextualHelp**: User guidance and learning resources

---

## Data Management

### Data Sources

- **OHLCV Data**: Historical candlestick data for currency pairs used in backtesting
- **News Data**: From GDELT, preprocessed and cached for sentiment analysis
- **Sentiment Data**: FinBERT sentiment scores (cached in `finbert_cache.json`) for educational analysis

### Data Pipeline

```
Raw Data → Validation → Preprocessing → Storage → Retrieval
                                           ↓
                                      Caching Layer
```

### Loading Data

```python
from utils.data_loader import load_ohlcv_data, load_news_for_pair

# Load OHLCV data
df = load_ohlcv_data('EURUSD', start_date='2024-01-01', end_date='2024-01-31')

# Load news for a currency pair
news = load_news_for_pair('EURUSD', date='2024-01-15')
```

---

## Development Guide

### Adding a New Agent

1. Create a new file in `agents/` directory
2. Implement the agent class following the base agent pattern
3. Register the agent in the graph builder (`graph/build_graph.py`)
4. Add corresponding tools in `tools/` directory

Example:

```python
# agents/my_agent.py
from langchain_core.agents import AgentExecutor
from langchain_core.tools import tool

class MyAgent:
    def __init__(self):
        self.name = "my_agent"
    
    @tool
    def analyze_data(self, data: dict) -> dict:
        """Analyze market data and generate educational insights"""
        # Implementation
        return {"recommendation": "BUY", "confidence": 0.85}
```

### Adding a New API Endpoint

```python
# In app.py
@app.route('/api/my-endpoint', methods=['POST'])
def my_endpoint():
    """Handle trading recommendation request"""
    data = request.json
    result = generate_recommendation(data)
    return jsonify(result)
```

### Running Tests

```bash
# Configure test environment
pytest tests/ -v

# With coverage
pytest tests/ --cov=./ --cov-report=html
```

### Code Style

- Follow **PEP 8** for Python code
- Use **ESLint** for JavaScript/React code
- Run formatters before committing:

```bash
# Python
black .
pylint **/*.py

# JavaScript
npm run lint
npm run format
```

---

## Contributing

We welcome contributions from the development team! Please follow these guidelines:

### Contribution Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and test thoroughly
   ```bash
   # Run tests
   pytest tests/
   # Run linters
   black . && pylint **/*.py
   ```

3. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new educational trading signal analysis"
   ```

4. **Push to remote**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** with detailed description

### Development Team

| Name | Role | Focus Area |
|------|------|-----------|
| John Omar Girado | Lead Developer & Backend Lead | Agent Architecture & System Design |
| Isaiah Daniel Lising | Data Engineer | Trading Terms, Backtesting & QA |
| Azer John Valdemoro | QA & Evaluator | QA & Evaluation |

### Contact

For questions or issues, reach out to the development team or check the issues section.

### Code Review Guidelines

- Minimum 2 approvals before merge
- All tests must pass
- Code coverage should not decrease
- Follow the existing code style and patterns
These metrics help educate traders on strategy performance:

- **Win Rate**: Percentage of profitable recommendations in historical analysis
- **Profit Factor**: Gross Profit / Gross Loss ratio for learning risk-reward
- **Sharpe Ratio**: Risk-adjusted returns to understand volatility impact
- **Max Drawdown**: Largest peak-to-trough decline to demonstrate risk exposure
- **API Response Time**: Target < 500ms for smooth user experience
- **Agent Consensus**: % agreement among agents showing recommendation strength
- **Profit Factor**: Gross Profit / Gross Loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **API Response Time**: Target < 500ms
- **Agent Consensus**: % agreement among agents

---

## Troubleshooting

### Common Issues

**Issue**: Flask server won't start
- Check if port 5000 is already in use: `netstat -ano | findstr :5000`
- Kill the process: `taskkill /PID <PID> /F`

**Issue**: Frontend can't connect to backend
- Verify CORS is enabled in `app.py`
- Check `VITE_API_URL` in frontend `.env`
- Ensure backend is running on the correct port

**Issue**: LLM inference is slow
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Use a smaller model for faster inference
- Check system resources (CPU/GPU)

**Issue**: Data loading fails
- Verify data files exist in `data/` directory
- Check file permissions
- Review logs in `logger.py` output

---

## Acknowledgments

- LangChain for agent framework
- Ollama for local LLM inference
- React for frontend framework
- Flask for backend API

---

**Last Updated**: May 28, 2026

This platform is designed as an educational tool for retail and beginner forex traders to learn market dynamics through practical examples, interactive recommendations, and backtested strategies. It is not intended for live trading without proper risk management and trader expertise.
