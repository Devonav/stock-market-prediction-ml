# Project Structure

```
Stock_Market/
│
├── 📁 backend/                 Backend API & ML Models
│   ├── 📁 src/                Core ML modules
│   │   ├── data_collector.py
│   │   ├── feature_engineering.py
│   │   ├── advanced_features.py
│   │   ├── ml_models.py
│   │   ├── deep_learning_models.py
│   │   ├── backtesting.py
│   │   └── visualization.py
│   │
│   ├── 📁 scripts/            Utility scripts
│   │   └── analyze_results.py
│   │
│   ├── api.py                 Flask REST API (PORT 5000)
│   ├── app.py                 Streamlit app
│   ├── main.py                CLI interface
│   ├── requirements.txt       Python dependencies
│   └── requirements-api.txt
│
├── 📁 frontend/               React Frontend
│   ├── 📁 src/
│   │   ├── 📁 components/    React components
│   │   ├── 📁 services/      API services
│   │   └── App.jsx
│   ├── .env                   Environment variables
│   └── package.json           Node dependencies
│
├── 📁 docs/                   Documentation
│   ├── README.md             Full documentation
│   ├── REACT_SETUP.md        Frontend setup
│   └── WEB_APP_GUIDE.md      Web app guide
│
├── 📁 scripts/                Utility scripts
│   ├── run.bat
│   └── test_stocks.bat
│
├── 📁 data/                   Stock data (auto-generated)
├── 📁 models/                 Trained models (auto-generated)
├── 📁 results/                Results (auto-generated)
├── 📁 notebooks/              Jupyter notebooks
│
├── README.md                  Quick start guide
└── .gitignore                Git ignore rules
```

## Quick Commands

**Start Backend:**
```bash
cd backend
python api.py
```

**Start Frontend:**
```bash
cd frontend
npm run dev
```
