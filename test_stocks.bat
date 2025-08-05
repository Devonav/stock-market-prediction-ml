@echo off
echo Testing Stock Prediction System...
echo.

call venv\Scripts\activate.bat

echo Testing Tech Stocks...
python main.py AAPL --compare --save-model
python main.py MSFT --compare --save-model
python main.py GOOGL --compare --save-model

echo.
echo Testing Different Sectors...
python main.py JPM --compare
python main.py JNJ --compare

echo.
echo Testing Market Indices...
python main.py SPY --compare
python main.py QQQ --compare

echo.
echo Testing Different Time Periods...
python main.py AAPL --period 1y --compare
python main.py AAPL --period 5y --compare

echo.
echo Testing Different Prediction Types...
python main.py TSLA --target-type price_change --compare
python main.py AAPL --target-days 3 --compare

echo.
echo All tests completed!