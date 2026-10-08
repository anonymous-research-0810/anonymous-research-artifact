"""Constants for the paper option-hedging pipeline."""

from __future__ import annotations

DEFAULT_START_YEAR = 2016
DEFAULT_END_YEAR = 2025
SPX_SECID = 108105
FILE_TEMPLATE = "spx_options_{year}.parquet"
MARKET_COLUMNS = ("spot", "zero_rate", "dividend_rate")
OPTION_COLUMNS = (
    "secid",
    "date",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "gamma",
    "vega",
    "theta",
    "optionid",
    "symbol",
)
NUMERIC_COLUMNS = (
    "secid",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "gamma",
    "vega",
    "theta",
    "optionid",
)
STRING_COLUMNS = ("cp_flag", "symbol")
COLUMN_DESCRIPTIONS = {
    "secid": "OptionMetrics underlying identifier; 108105 identifies the S&P 500 index",
    "date": "Trading date of the closing quote",
    "exdate": "Option expiry date",
    "cp_flag": "Option type: C for calls and P for puts",
    "strike_price": "Strike multiplied by 1000; divide by 1000 for index points",
    "best_bid": "Best closing bid in index points",
    "best_offer": "Best closing offer in index points",
    "volume": "Contracts traded on the quote date",
    "open_interest": "Open interest in contracts",
    "impl_volatility": "Annualized Black-Scholes implied volatility",
    "delta": "Option delta",
    "gamma": "Option gamma",
    "vega": "Option vega",
    "theta": "Option theta per calendar day",
    "optionid": "OptionMetrics contract identifier, constant throughout its lifecycle",
    "symbol": "OptionMetrics symbol containing root, expiry, option type, and strike",
    "spot": "Closing index level from the spot table's close field",
    "zero_rate": "Annualized continuously compounded zero rate in decimal units, interpolated by quote date and maturity",
    "dividend_rate": "Annualized continuous dividend yield in decimal units",
}
