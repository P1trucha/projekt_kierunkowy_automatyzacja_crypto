# 🤖 AI Trading Bot (Binance)

## 📌 Opis projektu

Projekt przedstawia system automatycznego handlu kryptowalutami oparty o analizę techniczną oraz model uczenia maszynowego. Bot analizuje dane rynkowe w czasie rzeczywistym, podejmuje decyzje inwestycyjne oraz zarządza otwartymi pozycjami.

System został zaprojektowany w architekturze modułowej i działa w trybie ciągłym, wykonując cykliczną analizę rynku.

---

## 🎯 Cel projektu

Celem projektu było stworzenie systemu:

- automatyzującego podejmowanie decyzji inwestycyjnych,
- wykorzystującego wskaźniki analizy technicznej,
- wspomaganego przez model Machine Learning,
- zarządzającego ryzykiem i pozycjami w sposób autonomiczny.

---

## 🧠 Główne komponenty systemu

### 1. 📊 Data Feed (`data_feed.py`)
- pobieranie danych rynkowych z Binance
- obsługa danych świecowych (OHLCV)
- przygotowanie danych do analizy

---

### 2. ⚙️ Decision Engine (`decision_engine.py`)
- analiza wskaźników technicznych:
  - RSI
  - MACD
  - EMA
  - Bollinger Bands
- generowanie decyzji:
  - `BUY`
  - `SELL`
  - `HOLD`

---

### 3. 🤖 ML Gating (`ml_gating.py`)
- model: **Logistic Regression (scikit-learn)**
- uczenie na podstawie historycznych transakcji
- przewidywanie prawdopodobieństwa sukcesu (`prob_win`)
- filtracja sygnałów BUY

👉 Bot podejmuje decyzję tylko wtedy, gdy:
```text
prob_win >= threshold
