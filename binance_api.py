# binance_api.py
import time
import logging
import requests
from typing import List, Dict, Any, Optional


class BinanceAPI:
    """
    Источник данных: Binance USDⓈ-M Futures (base_url=fapi.binance.com).

    Используем только публичные эндпоинты:
      • /fapi/v1/exchangeInfo            -> список инструментов
      • /fapi/v1/ticker/24hr             -> 24h tickers (для префильтра волатильности)
      • /fapi/v1/klines                  -> свечи
      • /fapi/v1/openInterest            -> текущий OI
    """

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        sleep_ms: int = 250,
        max_retries: int = 3,
        retry_backoff_sec: int = 2,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.sleep_ms = sleep_ms
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.sess = session or requests.Session()
        self.sess.headers.update(
            {
                "User-Agent": "bybit-macd-bot/1.0 (binance-data)",
                "Accept": "application/json",
            }
        )

    # --- low level request with retry ---
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.sess.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    # rate limit
                    wait = self.retry_backoff_sec * attempt
                    logging.warning(f"429 RateLimit on {path}, sleep {wait}s…")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                wait = self.retry_backoff_sec * attempt
                logging.warning(f"GET {path} failed (attempt {attempt}/{self.max_retries}): {e}. Sleep {wait}s…")
                time.sleep(wait)
        raise RuntimeError(f"GET {path} failed after retries: {last_err}")

    # --- public API wrappers ---
    def get_instruments(self) -> List[Dict[str, Any]]:
        """
        Все торгуемые USDT-перпетуалы.
        """
        data = self._get("/fapi/v1/exchangeInfo")
        syms = []
        for s in data.get("symbols", []):
            try:
                if (
                    s.get("quoteAsset") == "USDT"
                    and s.get("contractType") == "PERPETUAL"
                    and s.get("status") == "TRADING"
                ):
                    syms.append({"symbol": s["symbol"]})
            except Exception:
                continue
        logging.info(f"Найдено {len(syms)} торгуемых USDT-перпетуалов на Binance.")
        time.sleep(self.sleep_ms / 1000.0)
        return syms

    def get_tickers(self) -> List[Dict[str, Any]]:
        """
        24h tickers по всем фьючерсам; фильтруем в main по нужным символам.
        """
        data = self._get("/fapi/v1/ticker/24hr")
        time.sleep(self.sleep_ms / 1000.0)
        return data

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict[str, float]]:
        """
        Свечи фьючерса (symbol пример: BTCUSDT), interval пример: '1h','4h','1d','1w','1M'
        Возвращаем список dict с ключами open/high/low/close (float).
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        arr = self._get("/fapi/v1/klines", params=params)
        out = []
        for x in arr:
            # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
            try:
                out.append(
                    {
                        "open": float(x[1]),
                        "high": float(x[2]),
                        "low": float(x[3]),
                        "close": float(x[4]),
                    }
                )
            except Exception:
                continue
        time.sleep(self.sleep_ms / 1000.0)
        return out

    def get_open_interest(self, symbol: str) -> float:
        """
        Текущий OI по инструменту.
        """
        data = self._get("/fapi/v1/openInterest", params={"symbol": symbol})
        try:
            return float(data.get("openInterest", 0.0))
        except Exception:
            return 0.0
